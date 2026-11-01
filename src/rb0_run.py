"""PBS RB0 probe/timing orchestration with an application-level terminal manifest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('probe','timing'),default='probe')
    p.add_argument('--data-dir',required=True)
    p.add_argument('--split-manifest',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    commands=[]
    if args.mode=='probe':
        commands.append(('checks',[sys.executable,'src/rb_check.py','--suite','all','--output',str(out/'checks')]))
    commands.append(('benchmark',[sys.executable,'src/rb_benchmark.py','--data-dir',args.data_dir,
        '--split-manifest',args.split_manifest,'--output',str(out/'benchmark')]+(['--probe'] if args.mode=='probe' else [])))
    metrics={};artifacts=[];passed=False;start=time.monotonic()
    try:
        for name,command in commands:
            result=subprocess.run(command)
            metrics[name+'.exit_code']=result.returncode
            child=out/name if name=='checks' else out/'benchmark'
            path=child/'manifest.json'
            if path.is_file():
                value=json.loads(path.read_text())
                metrics[name]=value['metrics']
                artifacts.append({'path':str(path),'role':name+'_manifest'})
            if result.returncode:
                raise SystemExit(result.returncode)
        passed=True
    finally:
        metrics['elapsed_s']=time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'contract':'E-0019@1','mode':args.mode,
            'source_commit':os.environ['SR_CODE_COMMIT'],'metrics':metrics,'artifacts':artifacts,
            'validator':{'name':'rb0_application','passed':passed}},indent=2)+'\n')


if __name__=='__main__':main()
