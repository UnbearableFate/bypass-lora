"""RB training entry: locked A2 split, V1 objective/scoring, paired initialization.

RB0 executes only smoke/timing. RBc/RB1 release belongs to the coordinator.
This module never reads an official validation or RB confirmation example.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from bypass_model import AdapterConfig, BranchModel
from calibrate import MODEL,REVISION,load_base
from rb_data import BASE_SEED,STUDY,apply_confirmation,plan_batches
from v1_data import derive_seed,audit_training_exclusions
from v1_lora import LoRAConfig,LoRAModel
from v1_run import (prepare,evaluate,train as v1_train,TrainingDiverged,EvaluationNonfinite,
                    preserved_evaluation_state)
from v5_train import file_sha256,write_json,plan_metadata

LAYERS=(6,13,20,27)
# E-0026 width probe: W-01 r128 T0, W-02 r128 R_res KernelMatrix S1, W-03 the
# same-four-layer all-7 LoRA whose implementation count is nearest to W-02.
WIDTH_ARMS=('W-01','W-02','W-03')
WIDTH_RANK=128
TAP_ARMS=('X-01','X-02','X-03')
ETA_GRID=(.0625,.125,.25,.5,1.,2.)
ETA_EXTENSIONS=(.03125,4.)
ETA_ARMS={f'ETA-{tap}-{eta:g}':(tap,eta) for tap in ('QK','X')
          for eta in (*ETA_GRID,*ETA_EXTENSIONS) if eta!=.25}
E0028_ARMS=(*TAP_ARMS,*ETA_ARMS)
LORA_RANKS={'RB1-07':7,'W-03':31}
PARAMETERS={**{f'RB1-{i:02}':663552 for i in range(1,7)},
            'RB1-07':630784,'RB1-08':1318912,'RB1-09':925696,
            'W-01':2752512,'W-02':2752512,'W-03':2793472}
PARAMETERS.update({'X-01':794624,'X-02':794624,'X-03':925696,
                   **{arm:794624 if tap=='X' else 925696 for arm,(tap,eta) in ETA_ARMS.items()}})
FAMILIES={**{f'RB1-{i:02}':'RBc-R' for i in range(2,7)},'RB1-01':'RBc-T',
          'RB1-07':'RBc-L','RB1-08':'RBc-C','RB1-09':'RBc-C',
          'W-01':'RBc-T','W-02':'RBc-R','W-03':'RBc-L'}
FAMILIES.update({arm:'RBc-C' for arm in E0028_ARMS})
REPRESENTATIVES={'RBc-T':'RB1-01','RBc-R':'RB1-02','RBc-L':'RB1-07','RBc-C':'RB1-08'}
GRIDS={'RBc-T':(1e-4,3e-4,1e-3),'RBc-L':(1e-4,3e-4,1e-3),
       'RBc-R':(3e-4,1e-3,3e-3),'RBc-C':(3e-4,1e-3,3e-3)}
EXTENSIONS={'RBc-T':(3e-5,3e-3),'RBc-L':(3e-5,3e-3),
            'RBc-R':(1e-4,1e-2),'RBc-C':(1e-4,1e-2)}


def contract(arm,stage):
    if arm in E0028_ARMS:
        return 'E-0028@1'
    if arm in WIDTH_ARMS:
        return 'E-0026@1'
    return 'E-0019@1' if stage=='smoke' else 'E-0020@1' if stage=='calibration' else 'E-0021@1'


def adapter_config(arm,impl='prefix'):
    if arm not in PARAMETERS or arm in LORA_RANKS:
        raise ValueError('not an RB side-branch arm')
    tap,eta=ETA_ARMS.get(arm,('X' if arm in ('X-01','X-02') else 'QK',.25))
    return AdapterConfig(layers=LAYERS,branch='kernel_tokenwise' if arm in ('RB1-01','W-01','X-02','X-03') else 'memory',
        fusion='C_dual' if arm=='RB1-08' else 'C_up' if arm=='RB1-09' or arm in E0028_ARMS else 'F_out',
        tap='direct_hidden' if tap=='X' else 'qknorm_pre_rope',
        rank=WIDTH_RANK if arm in WIDTH_ARMS else 32,
        steps=2 if arm=='RB1-03' else 4 if arm in ('RB1-04','RB1-05','RB1-06') else 1,
        fixed_query=arm=='RB1-05',eta=.0625 if arm=='RB1-06' else eta,impl=impl)


def make_model(arm,phase='R1',seed=0,checkpoint=None,impl='prefix'):
    if arm not in ('Base',*PARAMETERS):
        raise ValueError('unknown RB arm')
    base=load_base()
    torch.manual_seed(derive_seed(BASE_SEED,STUDY,phase,'init',seed))
    if arm=='Base':
        return base
    if arm in LORA_RANKS:
        config=LoRAConfig(layers=LAYERS,rank=LORA_RANKS[arm],alpha=LORA_RANKS[arm])
        model=LoRAModel.load_adapter(base,checkpoint) if checkpoint else LoRAModel(base,config)
    else:
        config=adapter_config(arm,impl)
        model=BranchModel.load_adapter(base,checkpoint) if checkpoint else BranchModel(base,config)
    if model.config!=config:
        raise ValueError('checkpoint differs from registered RB arm')
    params=list(model.adapter_parameters())
    assert sum(p.numel() for p in params)==PARAMETERS[arm]
    assert all(p.requires_grad and p.dtype==torch.float32 for p in params)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==PARAMETERS[arm]
    return model


class TrainingAccounting:
    """Accumulate side counts on device; serialize only after the training loop."""
    def __init__(self,model):
        self.counts={}
        self.states={}
        self.handle=model.register_forward_hook(self.observe) if isinstance(model,BranchModel) else None

    def observe(self,model,inputs,output):
        if not model.training:
            return
        for index,counts in output.accounting.items():
            current=self.counts.setdefault(index,{'attention':0,'ffn':0,'writes':0,'reads':0,'valid_tokens':0})
            for name,value in counts['base'].items():
                current[name]+=value
            for name in ('writes','reads','valid_tokens'):
                value=counts['side'][name]
                current[name]+=value.detach() if torch.is_tensor(value) else value
            state=output.cache.memory[index]
            if state is not None:
                self.states[index]={'M.shape':list(state.M.shape),'c.shape':list(state.c.shape),
                    'dtype':str(state.M.dtype),'M_c_bytes':(state.M.numel()+state.c.numel())*4,
                    'counts_bytes':state.counts.numel()*state.counts.element_size()}

    def finish(self):
        if self.handle:
            self.handle.remove()
        return {'layers':{str(i):{k:int(v) for k,v in values.items()} for i,values in self.counts.items()},
                'states':self.states,'state_M_c_bytes':sum(x['M_c_bytes'] for x in self.states.values())}


def load_data(data_dir,split_manifest,tok):
    original=prepare(data_dir)
    split=json.loads(Path(split_manifest).read_text())
    splits=apply_confirmation(original,split)
    excluded=audit_training_exclusions(splits,tok)
    return splits,excluded


def train(model,batches,tok,lr,output,monitor=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    accounting=TrainingAccounting(model)
    try:
        metrics=v1_train(model,batches,tok,lr,output,intermediate=monitor)
    finally:
        counts=accounting.finish()
        write_json(output/'accounting.json',counts)
    if isinstance(model,BranchModel):
        valid=metrics['train.input_tokens']
        for values in counts['layers'].values():
            assert values['valid_tokens']==valid
            expected=0 if model.config.branch=='kernel_tokenwise' else valid
            assert values['writes']==expected
            assert values['reads']==expected*model.config.steps
            assert values['attention']==values['ffn']==metrics['train.examples']
    return {**metrics,'runtime.accounting':counts}


def extension_lr(family,candidates):
    grid=GRIDS[family]
    if set(candidates)!=set(grid):
        raise ValueError('extension requires exactly the initial family grid')
    finite=[(lr,nll) for lr,nll in candidates.items() if nll is not None and math.isfinite(nll)]
    if not finite:
        raise ValueError('family has no finite calibration candidate')
    best=min(finite,key=lambda p:(p[1],p[0]))[0]
    return EXTENSIONS[family][0] if best==grid[0] else EXTENSIONS[family][1] if best==grid[-1] else None


def lock_lrs(manifests):
    families={f:{} for f in GRIDS}
    inputs=[]
    split_hash=None
    order_hash=None
    for path in manifests:
        path=Path(path).resolve()
        value=json.loads(path.read_text())
        m=value.get('metrics',{})
        family=value.get('family')
        if family not in GRIDS or value.get('stage')!='calibration' or value.get('phase')!='V0' or value.get('seed')!=0 or value.get('arm')!=REPRESENTATIVES[family]:
            raise ValueError('unexpected calibration family/arm/phase/seed')
        if value.get('status')!='complete' or value.get('validator',{}).get('passed') is not True:
            raise ValueError('incomplete calibration manifest')
        lr=m.get('learning_rate')
        if type(lr) not in (int,float) or lr<=0 or not math.isfinite(lr) or lr in families[family]:
            raise ValueError('invalid or duplicate family LR')
        if split_hash is not None and split_hash!=m.get('split_manifest.sha256'):
            raise ValueError('calibration split mismatch')
        split_hash=m.get('split_manifest.sha256')
        if not isinstance(split_hash,str) or len(split_hash)!=64:
            raise ValueError('calibration split hash missing')
        order=m.get('plan.order_sha256')
        if not isinstance(order,str) or len(order)!=64 or order_hash is not None and order_hash!=order:
            raise ValueError('calibration sample order mismatch')
        order_hash=order
        failed=m.get('failed',False)
        nll=m.get('calibration.nll')
        if not failed:
            if type(nll) not in (int,float) or not math.isfinite(nll) or nll<=0:
                raise ValueError('nonfailed calibration requires finite positive NLL')
            if m.get('train.input_tokens',0)<200000 or m.get('train.steps')!=m.get('plan.steps'):
                raise ValueError('incomplete calibration budget')
        families[family][float(lr)]=None if failed else float(nll)
        inputs.append({'path':str(path),'sha256':file_sha256(path),'bytes':path.stat().st_size})
    choices,boundaries={},{}
    for family,candidates in families.items():
        initial={lr:candidates[lr] for lr in GRIDS[family] if lr in candidates}
        extension=extension_lr(family,initial)
        expected=set(GRIDS[family])|({extension} if extension is not None else set())
        if set(candidates)!=expected:
            raise ValueError('missing/unregistered endpoint extension')
        finite={lr:nll for lr,nll in candidates.items() if nll is not None}
        minimum=min(finite.values())
        choices[family]=min(lr for lr,nll in finite.items() if (nll-minimum)/minimum<=.01)
        boundaries[family]=extension is not None and finite.get(extension)==minimum
    return {'experiment':'E-0020@1','study_id':STUDY,'split_manifest_sha256':split_hash,
            'locked_lrs':choices,'boundary_minimum':boundaries,'inputs':inputs}


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    metrics,passed={},False
    phase='V0' if args.stage=='calibration' else args.phase
    start=time.monotonic()
    try:
        if not math.isfinite(args.lr) or args.lr<=0:
            raise ValueError('learning rate must be finite and positive')
        if args.stage=='calibration' and args.seed!=0:
            raise ValueError('RBc uses seed0 only')
        if args.stage=='train' and (args.phase!='R1' or args.three_epochs):
            raise ValueError('RB1 formal budget is2M/3epochs;RB5 needs its own later release/contract')
        if args.arm in E0028_ARMS and (args.seed!=0 or args.phase!='R1' or args.lr!=1e-3):
            raise ValueError('E-0028 fixes R1 seed0 and LR1e-3')
        if args.stage=='calibration' and args.arm not in REPRESENTATIVES.values():
            raise ValueError('calibration requires family representative')
        if args.stage!='smoke' and (args.eval_limit is not None or args.smoke_steps is not None):
            raise ValueError('partial budgets/evaluation allowed only in smoke')
        if args.stage=='calibration' and args.three_epochs:
            raise ValueError('calibration has a fixed200k token budget')
        if args.stage=='train':
            lock=json.loads(Path(args.lr_lock).read_text()) if args.lr_lock else {}
            if lock.get('experiment')!='E-0020@1' or lock.get('locked_lrs',{}).get(FAMILIES[args.arm])!=args.lr:
                raise ValueError('formal training requires the committed family LR lock')
            if lock.get('split_manifest_sha256')!=file_sha256(args.split_manifest):
                raise ValueError('formal training split differs from calibration lock')
            metrics['lr_lock.sha256']=file_sha256(args.lr_lock)
        tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        tok.pad_token_id=tok.eos_token_id
        splits,excluded=load_data(args.data_dir,args.split_manifest,tok)
        write_json(out/'excluded-train.json',excluded)
        write_json(out/'config.json',vars(args))
        budget=200000 if args.stage=='calibration' else None if args.three_epochs else 2000000
        batches=plan_batches(splits,tok,phase,args.seed,[(x['task'],x['id']) for x in excluded],budget)
        if args.stage=='smoke':
            batches=batches[:args.smoke_steps or 4]
        metrics.update(plan_metadata(batches))
        metrics['split_manifest.sha256']=file_sha256(args.split_manifest)
        metrics['learning_rate']=args.lr
        metrics['failed']=False
        metrics.update({'model_revision':REVISION,'tokenizer_revision':REVISION,
            'source.commit':os.environ.get('SR_CODE_COMMIT','unknown'),
            'environment.lock_sha256':file_sha256(Path(__file__).resolve().parents[1]/'uv.lock'),
            'torch':torch.__version__,'cuda':torch.version.cuda})
        tick=time.monotonic()
        model=make_model(args.arm,phase,args.seed)
        metrics['load.seconds']=time.monotonic()-tick
        metrics['hardware']=torch.cuda.get_device_name()
        metrics['params.stored']=sum(p.numel() for p in model.adapter_parameters())
        metrics['params.trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.arm in E0028_ARMS:
            from rb_tap_check import initial_state, step0_diagnostics
            metrics['init.parameters.sha256']=initial_state(model)[0]
            with preserved_evaluation_state(model):
                write_json(out/'step0-rms.json',step0_diagnostics(model,batches[0][0],tok))
        def monitor(step,current):
            values,preds=evaluate(current,splits,tok,args.eval_limit,split_names=('dev',))
            write_json(out/f'dev-step-{step}.json',{'step':step,'metrics':values,'predictions':preds})
        if args.arm in (*WIDTH_ARMS,*E0028_ARMS):
            # E-0026 rule 1(b): the zero-exit initialization is scored before any update.
            tick=time.monotonic()
            with preserved_evaluation_state(model):
                monitor(0,model)
            metrics['init_eval.seconds']=time.monotonic()-tick
        metrics.update(train(model,batches,tok,args.lr,out,monitor if args.stage!='calibration' else None))
        assert metrics['params.with_grad']==PARAMETERS[args.arm]
        if args.stage!='calibration':
            tick=time.monotonic()
            model.export_adapter(out/'adapter')
            metrics['save.seconds']=time.monotonic()-tick
        else:
            # RBc selects an LR from NLL and RB1 starts from scratch. Checkpoint
            # identity is covered by smoke/formal runs, not repeated per LR.
            metrics['calibration.export_reload_skipped']=True
        split_names=(('calibration',) if args.stage=='calibration' else
                     ('calibration','dev') if args.stage=='smoke' else ('dev',))
        tick=time.monotonic()
        values,preds=evaluate(model,splits,tok,args.eval_limit,split_names=split_names)
        metrics.update(values)
        metrics['eval.seconds']=time.monotonic()-tick
        write_json(out/'predictions.json',preds)
        if args.stage!='calibration':
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model=make_model(args.arm,phase,args.seed,out/'adapter')
            values,reloaded=evaluate(model,splits,tok,args.eval_limit,split_names=split_names)
            metrics['reload.identical']=preds==reloaded
            assert metrics['reload.identical'] and all(metrics[k]==v for k,v in values.items())
            write_json(out/'checkpoint.json',{name:{'path':str(out/'adapter'/name),
                'sha256':file_sha256(out/'adapter'/name),'bytes':(out/'adapter'/name).stat().st_size}
                for name in ('config.json','adapter.pt')})
            if args.arm in E0028_ARMS:
                tick=time.monotonic()
                values,calibration=evaluate(model,splits,tok,args.eval_limit,split_names=('calibration',))
                metrics.update(values)
                metrics['calibration.seconds']=time.monotonic()-tick
                write_json(out/'calibration-predictions.json',calibration)
        passed=True
    except (TrainingDiverged,EvaluationNonfinite) as exc:
        if isinstance(exc,TrainingDiverged):
            metrics.update(exc.metrics)
        metrics.update(failed=True,failure_reason=str(exc))
        if isinstance(exc,TrainingDiverged) and args.arm in ETA_ARMS:
            # Rule 5 excludes a genuinely non-finite TRAINING candidate. Other
            # engineering/evaluation failures remain failed jobs and must retry.
            metrics['failure_kind']='training_nonfinite'
            passed=True
        elif args.stage!='calibration':
            raise
        else:
            metrics['calibration.nll']=None
            passed=True
    finally:
        metrics['elapsed_s']=time.monotonic()-start
        write_json(out/'manifest.json',{'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'contract':contract(args.arm,args.stage),
            'arm':args.arm,'family':FAMILIES[args.arm],'seed':args.seed,'phase':phase,'stage':args.stage,
            'metrics':metrics,'artifacts':[{'path':str(p),'role':p.name} for p in out.iterdir() if p.is_file() and p.name!='manifest.json'],
            'validator':{'name':'rb_training_runtime','passed':passed}})


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run')
    p.add_argument('--arm',choices=PARAMETERS,required=True)
    p.add_argument('--stage',choices=('smoke','calibration','train'),default='smoke')
    p.add_argument('--phase',choices=('R1','RB5'),default='R1')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lr',type=float,required=True)
    p.add_argument('--lr-lock',type=Path)
    p.add_argument('--split-manifest',type=Path,required=True)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--three-epochs',action='store_true')
    p.add_argument('--eval-limit',type=int)
    p.add_argument('--smoke-steps',type=int)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('lock')
    p.add_argument('--manifests',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='lock':
        write_json(args.output,lock_lrs(args.manifests))
    else:
        run(args)


if __name__=='__main__':
    main()
