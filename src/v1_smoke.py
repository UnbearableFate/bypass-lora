"""V0 regression, V1 invariants and measured three-family smoke, one registered attempt."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest


def valid_smoke(manifest, steps=5):
    metrics = manifest.get('metrics', {})
    if (manifest.get('status') != 'complete' or not manifest.get('validator', {}).get('passed')
            or metrics.get('train.diverged') or metrics.get('run.failed_lr')
            or metrics.get('train.steps') != steps or metrics.get('reload.identical') is not True):
        return False
    for split in ('calibration', 'dev'):
        for task in ('piqa', 'boolq'):
            for metric in ('nll', 'accuracy'):
                value = metrics.get(f'{split}.{task}.{metric}')
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    return False
        value = metrics.get(f'{split}.nll')
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
    return True


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
                '--smoke-steps','5','--eval-limit','32','--data-dir',
                '/work/xg24i002/x10041/bypass-lora/data/v1','--output',str(arm_out)])
            manifest = json.loads((arm_out/'manifest.json').read_text())
            metrics.update({arm+'.'+k:v for k,v in manifest['metrics'].items()})
            if result.returncode or not valid_smoke(manifest):
                raise RuntimeError(f'{arm} smoke failed')
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        metrics['train.input_tokens'] = sum(metrics.get(a+'.train.input_tokens',0) for a in ('lora','F_out','C_dual'))
        (out/'manifest.json').write_text(json.dumps({'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':metrics,
            'artifacts':[{'path':str(out/'checks.xml'),'role':'pytest_results'}],
            'validator':{'name':'v1_smoke','passed':passed}},indent=2)+'\n')
    return 0 if passed else 1

if __name__=='__main__':
    sys.exit(main())
