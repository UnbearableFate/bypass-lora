"""RB0-only timing proxies for RB5 layouts absent from the RB1 arm matrix.

No RB5 performance experiment or confirmation access. Every case uses the same
R1 data-order/initialization seed as the main RB0 benchmark for comparable shape
costs. Future formal RB5 initialization and controls belong to its later contract.
"""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from bypass_model import BranchModel
from calibrate import MODEL,REVISION,load_base
from mig_profile_guard import main as profile_guard
from rb_data import BASE_SEED,STUDY,plan_batches
from rb_train import adapter_config,load_data,train
from v1_data import derive_seed
from v1_lora import LoRAConfig,LoRAModel
from v1_run import evaluate
from v5_train import plan_metadata,write_json


def cases():
    residual=adapter_config('RB1-02')
    definitions={}
    for fusion,code,parameters in (('C_dual','Cdual',1318912),('C_up','Cup',925696)):
        definitions[code+'32-S4']=(replace(residual,fusion=fusion,steps=4),parameters)
    for fusion,code,parameters in (('F_out','R',663552),('C_dual','Cdual',1318912),('C_up','Cup',925696)):
        definitions['T0-'+code+'32-S4']=(replace(residual,branch='kernel_tokenwise',fusion=fusion,steps=4),parameters)
    for rank,parameters in ((44,916608),(64,1343488)):
        for steps in (1,4):
            definitions[f'R{rank}-S{steps}']=(replace(residual,rank=rank,steps=steps),parameters)
    for rank,parameters in ((1,630784),(2,1261568)):
        definitions[f'LoRA28-r{rank}']=(LoRAConfig(layers=tuple(range(28)),rank=rank,alpha=rank),parameters)
    return definitions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--split-manifest',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if profile_guard(['--output',str(out)]):raise SystemExit(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    tok.pad_token_id=tok.eos_token_id
    splits,excluded=load_data(args.data_dir,args.split_manifest,tok)
    omit=[(x['task'],x['id']) for x in excluded]
    full=plan_batches(splits,tok,'R1',0,omit,2000000)
    rb5=plan_batches(splits,tok,'RB5',0,omit,None)
    write_json(out/'schedules.json',{'R1':plan_metadata(full),'RB5':plan_metadata(rb5)})
    timings={};passed=False
    try:
        for label,(config,expected) in cases().items():
            folder=out/label;folder.mkdir(exist_ok=True)
            tick=time.monotonic()
            base=load_base()
            torch.manual_seed(derive_seed(BASE_SEED,STUDY,'R1','init',0))
            model=LoRAModel(base,config) if isinstance(config,LoRAConfig) else BranchModel(base,config)
            torch.cuda.synchronize();loading=time.monotonic()-tick
            params=list(model.adapter_parameters())
            assert sum(p.numel() for p in params)==expected
            assert all(p.requires_grad and p.dtype==torch.float32 for p in params)
            metrics=train(model,full[:120],tok,.001,folder)
            assert metrics['params.with_grad']==expected
            steps=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()][20:]
            assert len(steps)==100
            seconds=sum(x['seconds'] for x in steps)
            tick=time.monotonic();model.export_adapter(folder/'adapter')
            saving=time.monotonic()-tick
            tick=time.monotonic();evaluate(model,splits,tok,limit=32,split_names=('dev',))
            evaluating=time.monotonic()-tick
            timings[label]={'diagnostic_only':True,'model_config':vars(config),
                'warmup_steps':20,'stable_steps':100,'step_seconds':seconds/100,
                'tokens_per_second':sum(x['input_tokens'] for x in steps)/seconds,
                'load_seconds':loading,'save_seconds':saving,'sample_eval_seconds':evaluating,
                'peak_cuda_bytes':metrics['train.peak_cuda_bytes'],
                'params.stored':sum(p.numel() for p in params),'params.trainable':expected,
                'params.with_grad':metrics['params.with_grad'],'runtime.accounting':metrics['runtime.accounting']}
            write_json(out/'timings.json',timings)
            del model,base,params
            gc.collect();torch.cuda.empty_cache()
        passed=True
    finally:
        write_json(out/'manifest.json',{'status':'complete' if passed else 'partial','contract':'E-0019@1',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':{'cases.completed':len(timings),
            'stable_steps_per_case':100,'diagnostic_only':True,'hardware':torch.cuda.get_device_name()},
            'artifacts':[{'path':str(p),'role':p.name} for p in out.iterdir() if p.is_file() and p.name!='manifest.json'],
            'validator':{'name':'rb5_layout_cost_only','passed':passed}})


if __name__=='__main__':main()
