"""RB0: real-stream20+100-step timing, complete dev/save cost and RB budget estimates.

All resulting adapters are diagnostic timing artifacts,never formal checkpoints.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoTokenizer

from bypass_model import BranchModel
from calibrate import MODEL,REVISION
from rb_data import plan_batches
from rb_train import PARAMETERS,make_model,load_data,train
from rb_latency import measure as measure_latency
from v1_run import evaluate
from v5_latency import merge_lora
from v5_train import write_json,plan_metadata
from v1_lora import LoRAModel
from mig_profile_guard import main as profile_guard


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--split-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',action='store_true',help='Uniform20+4-step engineering startup only;not a stable timing result')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # This job compares time,so the D0019 2g exception applies.
    if profile_guard(['--output',str(out)]):raise SystemExit(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    tok.pad_token_id=tok.eos_token_id
    splits,excluded=load_data(args.data_dir,args.split_manifest,tok)
    exclusions=[(x['task'],x['id']) for x in excluded]
    plans={name:plan_batches(splits,tok,phase,0,exclusions,budget) for name,phase,budget in
           [('RBc','V0',200000),('RB1','R1',2000000),('RB5','RB5',None)]}
    schedule={name:plan_metadata(batches) for name,batches in plans.items()}
    schedule['excluded']=excluded
    write_json(out/'schedules.json',schedule)
    count=24 if args.probe else 120
    selected=plans['RB1'][:count]
    timings={};passed=False
    try:
        cases=[(arm,'prefix') for arm in PARAMETERS]+[('RB1-02','reference')]
        for arm,impl in cases:
            label=arm+('-reference' if impl=='reference' else '')
            folder=out/label;folder.mkdir(exist_ok=True)
            tick=time.monotonic()
            model=make_model(arm,impl=impl)
            torch.cuda.synchronize();load_seconds=time.monotonic()-tick
            metrics=train(model,selected,tok,.001,folder)
            assert metrics['params.with_grad']==PARAMETERS[arm]
            steps=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
            stable=steps[20:]
            assert len(stable)==(4 if args.probe else 100)
            stable_seconds=sum(x['seconds'] for x in stable)
            stable_tokens=sum(x['input_tokens'] for x in stable)
            tick=time.monotonic();model.export_adapter(folder/'adapter')
            save_seconds=time.monotonic()-tick
            tick=time.monotonic()
            # One complete dev measurement;other arms use the same32rows/task
            # sample,scaled against its measured full/sample ratio.
            values,pred=evaluate(model,splits,tok,limit=2 if args.probe else 32,split_names=('dev',))
            sample_eval_seconds=time.monotonic()-tick
            full_eval_seconds=None
            if arm=='RB1-02' and impl=='prefix' and not args.probe:
                tick=time.monotonic();evaluate(model,splits,tok,split_names=('dev',))
                full_eval_seconds=time.monotonic()-tick
            result={'arm':arm,'impl':impl,'diagnostic_only':True,'warmup_steps':20,
                'stable_steps':len(stable),'step_seconds':stable_seconds/len(stable),
                'tokens_per_second':stable_tokens/stable_seconds,'measured_input_tokens':stable_tokens,
                'load_seconds':load_seconds,'save_seconds':save_seconds,'sample_eval_seconds':sample_eval_seconds,
                'full_eval_seconds':full_eval_seconds,'peak_cuda_bytes':metrics['train.peak_cuda_bytes'],
                'params.stored':sum(p.numel() for p in model.adapter_parameters()),
                'params.trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'params.with_grad':metrics['params.with_grad'],'runtime.accounting':metrics['runtime.accounting']}
            if impl=='prefix' and arm in ('RB1-01','RB1-02','RB1-04','RB1-07','RB1-08') and not args.probe:
                result['latency']=measure_latency(model)
                if isinstance(model,LoRAModel):
                    merged=merge_lora(model)
                    model._targets.clear()
                    del model
                    gc.collect();torch.cuda.empty_cache()
                    result['latency_merged']=measure_latency(merged)
                    del merged
                    model=None
            timings[label]=result
            write_json(out/'timings.json',timings)
            del model
            gc.collect();torch.cuda.empty_cache()
        estimates={}
        if not args.probe:
            anchor=timings['RB1-02']
            factor=anchor['full_eval_seconds']/anchor['sample_eval_seconds']
            for arm in PARAMETERS:
                measured=timings[arm]
                evaluation=measured['sample_eval_seconds']*factor
                estimates[arm]={}
                for stage,evaluations in [('RBc',2),('RB1',5),('RB5',5)]:
                    # RB1:25/50/75%dev,finaldev,fresh reload. RB5 estimate
                    # assumes the same monitoring;its later contract can refine.
                    seconds=2*measured['load_seconds']+schedule[stage]['plan.steps']*measured['step_seconds']+evaluations*evaluation+measured['save_seconds']
                    estimates[arm][stage]={'estimated_seconds':seconds,'estimated_node_h':seconds/14400,
                        'steps':schedule[stage]['plan.steps'],'eval_count':evaluations,
                        'eval_seconds_per_pass_estimate':evaluation,
                        'basis':'direct100-step timing;full/sample dev scaling anchored toR_res S1;two loads+save;excludes later interventions/bootstrap and queue wait'}
        write_json(out/'cost-estimates.json',{'schedules':schedule,'arms':estimates,
            'RB1_training_monitor_reload_node_h':sum(v['RB1']['estimated_node_h'] for v in estimates.values()),
            'needs_profile':any(v['step_seconds']>.4 for v in timings.values() if v['impl']=='prefix' and v['arm']!='RB1-07'),
            'note':'All estimates are predictions,not billed use. Later intervention/analysis costs must be added before release.'})
        passed=True
    finally:
        write_json(out/'manifest.json',{'status':'complete' if passed else 'partial','contract':'E-0019@1',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':{'cases.completed':len(timings),'stable_steps_per_case':4 if args.probe else 100,
                'diagnostic_only':True,'hardware':torch.cuda.get_device_name()},
            'artifacts':[{'path':str(p),'role':p.name} for p in out.iterdir() if p.is_file() and p.name!='manifest.json'],
            'validator':{'name':'rb0_training_timing','passed':passed}})


if __name__=='__main__':main()
