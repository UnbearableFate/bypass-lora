"""RB0 PBS validation driver, JUnit error summary and optional IDs-only split creation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',choices=('core','model','all'),default='core')
    parser.add_argument('--data-dir')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    core=['src/test_core.py','src/test_rb_core.py','src/test_rb_data.py','src/test_v1_data.py']
    model=['src/test_model.py','src/test_rb_model.py','src/test_v1_lora.py','src/test_v5_train.py','src/test_v2_memory.py']
    files=core if args.suite=='core' else model if args.suite=='model' else core+model
    start=time.monotonic()
    with (out/'checks.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest',*files,'-q','--junitxml='+str(out/'checks.xml')],stdout=log,stderr=subprocess.STDOUT)
    metrics={'checks.exit_code':result.returncode,'elapsed_s':time.monotonic()-start}
    artifacts=[{'path':str(out/name),'role':name} for name in ('checks.log','checks.xml') if (out/name).is_file()]
    if (out/'checks.xml').is_file():
        root=ET.parse(out/'checks.xml').getroot()
        cases=list(root.iter('testcase'))
        metrics.update({'checks.cases':len(cases),
            'checks.failed':sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
            'checks.skipped':sum(c.find('skipped') is not None for c in cases)})
        errors={}
        for prop in root.iter('property'):
            name,value=prop.get('name'),prop.get('value')
            if name.endswith(('.max_abs','.relative_inf')):
                errors[name]=max(errors.get(name,0),float(value))
        metrics['errors']=errors
    if result.returncode==0 and args.data_dir:
        from rb_data import confirmation_manifest,apply_confirmation
        from v1_run import prepare
        splits=prepare(args.data_dir)
        split=confirmation_manifest(splits)
        kept=apply_confirmation(splits,split)
        (out/'rb-confirmation.json').write_text(json.dumps(split,indent=2)+'\n')
        artifacts.append({'path':str(out/'rb-confirmation.json'),'role':'sealed_confirmation_ids'})
        metrics.update({'confirmation.rows':sum(split['counts'].values()),'confirmation.ids_sha256':split['ids_sha256'],
                        'rb.train_counts':kept['metadata']['rb_train_counts'],'model_evaluated':False})
    passed=result.returncode==0
    (out/'manifest.json').write_text(json.dumps({'status':'complete' if passed else 'partial',
        'contract':'E-0019@1','spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':metrics,
        'artifacts':artifacts,'validator':{'name':'rb0_'+args.suite,'passed':passed}},indent=2)+'\n')
    print(json.dumps(metrics),flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
