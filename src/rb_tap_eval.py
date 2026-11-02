"""E-0028 RB1-09 final-checkpoint calibration and compatibility evaluation."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL,REVISION
from rb1_artifacts import read,reference,validate_formal_run
from rb_eta_select import validate_calibration
from rb_tap_check import initial_state,step0_diagnostics
from rb_tap_identity import LEGACY_COMMIT,reference_check
from rb_train import load_data,make_model
from v1_run import evaluate
from v5_train import write_json


def predictions(rows):
    return [(r['task'],r['id'],r['gold'],r['prediction']) for r in rows]


def evaluate_legacy_source(adapter,data_dir,split_manifest,out,limit=None):
    """Run the accepted E-0021 source in an isolated interpreter in THIS job."""
    source=Path('/work/xg24i002/x10041/bypass-lora/snapshots')/LEGACY_COMMIT
    names=('rb_train.py','bypass_model.py','bypass_core.py','calibrate.py','v1_run.py','v1_data.py','rb_data.py')
    refs={name:reference(source/'src'/name) for name in names}
    result_path=out/'legacy-source-dev.json'
    program='''
import json,os,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from calibrate import MODEL,REVISION
from rb_train import load_data,make_model
from v1_run import evaluate
adapter,data_dir,split_manifest,output,commit,limit=sys.argv[1:]
limit=None if limit=='None' else int(limit)
tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
tok.pad_token_id=tok.eos_token_id
splits,_=load_data(data_dir,Path(split_manifest),tok)
model=make_model('RB1-09',checkpoint=Path(adapter))
metrics,rows=evaluate(model,splits,tok,limit,split_names=('dev',))
with open(output,'x') as handle:
    json.dump({'source_commit':commit,'hardware':torch.cuda.get_device_name(),
        'job_id':os.environ['PBS_JOBID'],'metrics':metrics,'predictions':rows},handle)
'''
    env=dict(os.environ,PYTHONPATH=str(source/'src'))
    with (out/'legacy-evaluation.log').open('w') as log:
        subprocess.run([sys.executable,'-c',program,str(adapter),str(data_dir),str(split_manifest),
                        str(result_path),LEGACY_COMMIT,str(limit)],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return read(result_path),refs


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    passed=False;metrics={};start=time.monotonic()
    try:
        origin,refs,adapter=validate_formal_run(args.training_run,'RB1-09')
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        tokenizer.pad_token_id=tokenizer.eos_token_id
        splits,excluded=load_data(args.data_dir,args.split_manifest,tokenizer)
        if reference(args.split_manifest)['sha256']!=origin['metrics']['split_manifest.sha256']:
            raise ValueError('re-evaluation split differs from RB1-09')
        metrics={k:v for k,v in origin['metrics'].items() if k.startswith(('train.','plan.','params.')) or
                 k in ('split_manifest.sha256','learning_rate','reload.identical','failed')}
        model=make_model('RB1-09')
        metrics['init.parameters.sha256']=initial_state(model)[0]
        from rb_data import plan_batches
        batches=plan_batches(splits,tokenizer,'R1',0,[(x['task'],x['id']) for x in excluded],2000000)
        write_json(out/'step0-rms.json',step0_diagnostics(model,batches[0][0],tokenizer))
        del model,batches
        gc.collect();torch.cuda.empty_cache()
        model=make_model('RB1-09',checkpoint=adapter)
        metrics['hardware']=torch.cuda.get_device_name()
        metrics['exec.pbs_job_id']=os.environ.get('PBS_JOBID','unknown')
        expected=dev_rows=None
        if not args.calibration_only:
            tick=time.monotonic()
            values,rows=evaluate(model,splits,tokenizer,split_names=('dev',))
            metrics.update(values);metrics['dev_eval.seconds']=time.monotonic()-tick
            dev_rows=rows
            write_json(out/'predictions.json',rows)
            expected=read(Path(args.training_run)/'predictions.json')
            metrics['dev_predictions_identical_to_E0021']=predictions(rows)==predictions(expected)
            metrics['dev_rows_identical_to_E0021']=rows==expected
        tick=time.monotonic()
        values,rows=evaluate(model,splits,tokenizer,split_names=('calibration',))
        metrics.update(values);metrics['calibration.seconds']=time.monotonic()-tick
        validate_calibration(rows,metrics)
        write_json(out/'calibration-predictions.json',rows)
        provenance={'training_run':refs,'evaluation_source_commit':os.environ['SR_CODE_COMMIT'],
            'model_revision':REVISION,'tokenizer_revision':REVISION,
            'environment_lock':reference(Path(__file__).resolve().parents[1]/'uv.lock')}
        metrics['source.commit']=os.environ['SR_CODE_COMMIT']
        metrics['model_revision']=metrics['tokenizer_revision']=REVISION
        metrics['environment.lock_sha256']=reference(Path(__file__).resolve().parents[1]/'uv.lock')['sha256']
        if not args.calibration_only:
            local=None
            if metrics['hardware']!=origin['metrics']['hardware']:
                del model
                gc.collect();torch.cuda.empty_cache()
                tick=time.monotonic()
                legacy,legacy_refs=evaluate_legacy_source(adapter,args.data_dir,args.split_manifest,out)
                metrics['legacy_source_dev.seconds']=time.monotonic()-tick
                if legacy['hardware']!=metrics['hardware'] or legacy['job_id']!=metrics['exec.pbs_job_id'] or legacy['source_commit']!=LEGACY_COMMIT:
                    raise ValueError('legacy reference did not execute in the same allocation at the accepted source')
                local=legacy['predictions']
                provenance['legacy_source_files']=legacy_refs
                provenance['legacy_source_dev']=reference(out/'legacy-source-dev.json')
            metrics['compatibility']=reference_check(dev_rows,expected,metrics['hardware'],
                                                     origin['metrics']['hardware'],local)
        write_json(out/'provenance.json',provenance)
        # Calibration has been written before any prediction-gate exception.
        if not args.calibration_only and not metrics['compatibility']['passed']:
            raise ValueError('rule 1(g): new-source predictions differ from the D-0060 reference')
        passed=True
    finally:
        metrics['elapsed_s']=time.monotonic()-start
        write_json(out/'manifest.json',{'status':'complete' if passed else 'partial','contract':'E-0028@1',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'arm':'RB1-09','family':'RBc-C','seed':0,'phase':'R1',
            'stage':'smoke' if args.calibration_only else 'checkpoint_evaluation','metrics':metrics,
            'artifacts':[{'path':str(p),'role':p.name} for p in out.iterdir() if p.is_file() and p.name!='manifest.json'],
            'validator':{'name':'rb1_09_compatibility_calibration','passed':passed}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--split-manifest',type=Path,required=True)
    parser.add_argument('--calibration-only',action='store_true',help='pre-review timing smoke; not a selection input')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
