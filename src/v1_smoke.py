"""V0 regression, V1 invariants and measured three-family smoke, one registered attempt."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True,exist_ok=True)
    src = Path(__file__).resolve().parent
    start = time.monotonic()
    metrics = {}
    passed = False
    try:
        code = pytest.main(['-v','-p','no:cacheprovider','--basetemp='+str(out/'tmp'),
                            '--junitxml='+str(out/'checks.xml'),str(src/'test_core.py'),
                            str(src/'test_model.py'),str(src/'test_v1_data.py'),
                            str(src/'test_v1_lora.py'),str(src/'test_v1_run.py')])
        metrics['pytest.exit_code'] = int(code)
        if code:
            raise RuntimeError('V0/V1 invariant tests failed')
        for arm in ('lora','F_out','C_dual'):
            arm_out = out/arm
            result = subprocess.run([sys.executable,str(src/'v1_run.py'),'--arm',arm,
                '--smoke-steps','20','--eval-limit','32','--data-dir',
                '/work/xg24i002/x10041/bypass-lora/data/v1','--output',str(arm_out)])
            manifest = json.loads((arm_out/'manifest.json').read_text())
            metrics.update({arm+'.'+k:v for k,v in manifest['metrics'].items()})
            if result.returncode or not manifest['validator']['passed']:
                raise RuntimeError(f'{arm} smoke failed')
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        metrics['train.input_tokens'] = sum(metrics.get(a+'.train.input_tokens',0) for a in ('lora','F_out','C_dual'))
        (out/'manifest.json').write_text(json.dumps({'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':metrics,
            'artifacts':[{'path':str(out),'role':'smoke_outputs'}],
            'validator':{'name':'v1_smoke','passed':passed}},indent=2)+'\n')
    return 0 if passed else 1

if __name__=='__main__':
    sys.exit(main())
