"""Registered V5 regression and three-arm end-to-end smoke in one allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from v5_train import ARMS, write_json


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    passed, metrics, artifacts = False, {}, []
    start = time.monotonic()
    try:
        code = pytest.main(['-q', '-p', 'no:cacheprovider', '--basetemp='+str(out/'tmp'),
                           '--junitxml='+str(out/'checks.xml'),
                           *[str(src/name) for name in ('test_v1_data.py','test_v1_lora.py',
                                                        'test_v1_run.py','test_v5_train.py')]])
        metrics['pytest.exit_code'] = int(code)
        if code:
            raise RuntimeError('V1/V5 regression suite failed')
        order = None
        for arm, expected in ARMS.items():
            result = subprocess.run([sys.executable, str(src/'v5_train.py'), 'run', '--phase','smoke',
                                     '--arm',arm,'--seed','0','--data-dir',
                                     '/work/xg24i002/x10041/bypass-lora/data/v1',
                                     '--output',str(out/arm)], check=True)
            path = out/arm/'manifest.json'
            manifest = json.loads(path.read_text())
            m = manifest['metrics']
            assert manifest['status'] == 'complete' and manifest['validator']['passed']
            assert m['reload.identical'] and m['init.identical']
            assert m['params.trainable'] == m['params.with_grad'] == expected
            assert m['plan.steps'] == 29165 and m['plan.examples'] == 58329
            assert m['plan.epoch_end_steps'] == {'1':9722,'2':19443,'3':29165}
            assert m['train.steps'] == m['smoke.steps']
            if order is None:
                order = m['plan.order_sha256']
            assert m['plan.order_sha256'] == order
            for epoch in range(1,4):
                assert (out/arm/f'dev-epoch-{epoch}.json').is_file()
            with (out/arm/'predictions.jsonl').open() as handle:
                assert sum(1 for _ in handle) == 64
            metrics.update({f'{arm}.{key}':value for key,value in m.items()})
            artifacts.append({'path':str(path),'role':arm+'_manifest'})
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        artifacts.append({'path':str(out/'checks.xml'),'role':'regressions'})
        write_json(out/'manifest.json', {'status':'complete' if passed else 'partial',
                   'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':metrics,
                   'artifacts':artifacts,'validator':{'name':'v5_smoke','passed':passed}})


if __name__ == '__main__':
    main()
