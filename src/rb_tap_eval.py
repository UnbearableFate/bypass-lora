"""E-0028 RB1-09 final-checkpoint calibration and compatibility evaluation."""
import argparse
import gc
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL,REVISION
from rb1_artifacts import read,reference,validate_formal_run
from rb_eta_select import validate_calibration
from rb_tap_check import initial_state,step0_diagnostics
from rb_train import load_data,make_model
from v1_run import evaluate
from v5_train import write_json


def predictions(rows):
    return [(r['task'],r['id'],r['gold'],r['prediction']) for r in rows]


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
        if not args.calibration_only:
            tick=time.monotonic()
            values,rows=evaluate(model,splits,tokenizer,split_names=('dev',))
            metrics.update(values);metrics['dev_eval.seconds']=time.monotonic()-tick
            write_json(out/'predictions.json',rows)
            expected=read(Path(args.training_run)/'predictions.json')
            metrics['dev_predictions_identical_to_E0021']=predictions(rows)==predictions(expected)
            metrics['dev_rows_identical_to_E0021']=rows==expected
            if not metrics['dev_predictions_identical_to_E0021']:
                raise ValueError('rule 1(g): RB1-09 dev predictions changed from E-0021')
        tick=time.monotonic()
        values,rows=evaluate(model,splits,tokenizer,split_names=('calibration',))
        metrics.update(values);metrics['calibration.seconds']=time.monotonic()-tick
        validate_calibration(rows,metrics)
        write_json(out/'calibration-predictions.json',rows)
        write_json(out/'provenance.json',{'training_run':refs,'evaluation_source_commit':os.environ['SR_CODE_COMMIT'],
            'model_revision':REVISION,'tokenizer_revision':REVISION,
            'environment_lock':reference(Path(__file__).resolve().parents[1]/'uv.lock')})
        metrics['source.commit']=os.environ['SR_CODE_COMMIT']
        metrics['model_revision']=metrics['tokenizer_revision']=REVISION
        metrics['environment.lock_sha256']=reference(Path(__file__).resolve().parents[1]/'uv.lock')['sha256']
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
