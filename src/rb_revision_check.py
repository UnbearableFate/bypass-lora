"""PBS regression and actual trainer CLI smoke for the four RV-0060 findings."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text())


def ref(path,role):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes':path.stat().st_size,'role':role}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--split-manifest',required=True)
    parser.add_argument('--timing-jobs',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    metrics={};artifacts=[];passed=False;started=time.monotonic()
    try:
        subprocess.run([sys.executable,'src/rb_check.py','--suite','all','--output',str(out/'checks')],check=True)
        checks=read(out/'checks'/'manifest.json')
        assert checks['status']=='complete' and checks['validator']['passed'] is True
        assert checks['metrics']['checks.failed']==checks['metrics']['checks.skipped']==0
        metrics['checks']=checks['metrics'];artifacts.append(ref(out/'checks'/'manifest.json','regression_manifest'))
        smoke={}
        for arm in ('RB1-02','RB1-01','RB1-07'):
            folder=out/arm
            subprocess.run([sys.executable,'src/rb_train.py','run','--stage','smoke',
                '--arm',arm,'--lr','.001','--seed','0','--smoke-steps','4','--eval-limit','2',
                '--data-dir',args.data_dir,'--split-manifest',args.split_manifest,
                '--output',str(folder)],check=True)
            value=read(folder/'manifest.json');m=value['metrics']
            assert value['status']=='complete' and value['validator']['passed'] is True
            assert value['contract']=='E-0019@1' and value['arm']==arm
            assert value['stage']=='smoke' and value['phase']=='R1' and value['seed']==0
            assert value['family']=={'RB1-02':'RBc-R','RB1-01':'RBc-T','RB1-07':'RBc-L'}[arm]
            assert m['failed'] is False and m['reload.identical'] is True
            assert m['train.steps']==m['plan.steps']==4 and m['train.examples']==m['plan.examples']==8
            assert m['train.input_tokens']==m['plan.input_tokens']>0
            assert m['params.stored']==m['params.trainable']==m['params.with_grad']==(630784 if arm=='RB1-07' else 663552)
            assert m['learning_rate']==.001 and math.isfinite(m['calibration.nll']) and m['calibration.nll']>0
            assert m['split_manifest.sha256']==hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest()
            assert len(m['plan.order_sha256'])==64
            assert m['source.commit']==os.environ['SR_CODE_COMMIT']
            for split in ('calibration','dev'):
                for task in ('piqa','boolq'):assert m[f'{split}.{task}.examples']==2
            for step in (1,2,3):assert (folder/f'dev-step-{step}.json').is_file()
            inventory=read(folder/'checkpoint.json')
            for name in ('config.json','adapter.pt'):
                actual=ref(folder/'adapter'/name,name)
                assert inventory[name]=={k:actual[k] for k in ('path','sha256','bytes')}
            smoke[arm]={'stage':'smoke','lock_schema_keys_present':True,
                'not_a_calibration_candidate':True,'reload.identical':True,
                'train.steps':m['train.steps'],'train.input_tokens':m['train.input_tokens']}
            artifacts.append(ref(folder/'manifest.json',arm+'_smoke_manifest'))
        metrics['smoke']=smoke
        subprocess.run([sys.executable,'src/rb_report.py','--jobs',args.timing_jobs,
            '--probe','/work/xg24i002/x10041/bypass-lora/runs/ATT-0188-1',
            '--output',str(out/'cost-report')],check=True)
        report=read(out/'cost-report'/'manifest.json')
        assert report['status']=='complete' and report['validator']['passed'] is True
        costs=read(out/'cost-report'/'cost-summary.json')
        for row in costs['cases']:
            policy=row['estimates']['RBc']
            assert policy['eval_passes']==policy['model_loads']==1 and policy['checkpoint_exports']==0
        metrics['costs']=report['metrics']
        metrics['RBc_12candidate_node_h']=costs['RBc_12candidate_node_h']
        metrics['RBc_16candidate_node_h']=costs['RBc_16candidate_node_h']
        artifacts.append(ref(out/'cost-report'/'manifest.json','updated_cost_manifest'))
        artifacts.extend(ref(out/'cost-report'/name,name) for name in ('cost-summary.json','cost-tables.md'))
        passed=True
    finally:
        metrics['elapsed_s']=time.monotonic()-started
        (out/'manifest.json').write_text(json.dumps({'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'contract':'E-0019@1',
            'source_commit':os.environ['SR_CODE_COMMIT'],'review_fixes':'RV-0060 lows1-4',
            'metrics':metrics,'artifacts':artifacts,
            'validator':{'name':'rb0_review_revision','passed':passed}},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
