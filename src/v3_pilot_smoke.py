"""PBS regression entry point plus four-step V3 F-line checkpoint roundtrip."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import pytest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    src = Path(__file__).parent
    code = pytest.main(['-v', '-p', 'no:cacheprovider', '--basetemp=' + str(args.output / 'tmp'),
                       '--junitxml=' + str(args.output / 'checks.xml'),
                       *[str(src / name) for name in ('test_v2_fline.py', 'test_v3_data.py', 'test_v3_pilot.py')]])
    if code:
        raise RuntimeError('V2 compatibility or V3 pilot regressions failed')
    import v2_fline
    sys.argv = ['v2_fline', '--data-dir', str(args.data_dir), '--output', str(args.output / 'pilot'),
                '--lr', '1e-3', '--tokens', '200000', '--data-phase', 'V3', '--smoke']
    v2_fline.main()
    report = json.loads((args.output / 'pilot' / 'metrics.json').read_text())
    metrics = report['metrics']
    if report['data_phase'] != 'V3' or metrics['train.steps'] != 4 or not metrics['reload.identical']:
        raise ValueError('V3 smoke did not complete its exact checkpoint roundtrip')
    metrics.update({'pytest.exit_code': int(code), 'smoke.elapsed_s': time.monotonic() - tick})
    (args.output / 'metrics.json').write_text(json.dumps({'metrics': metrics}, indent=2) + '\n')
    artifacts = [{'path': str(p.resolve()), 'role': str(p.relative_to(args.output))}
                 for p in args.output.rglob('*') if p.is_file() and p.name != 'manifest.json']
    manifest = {'status': 'complete', 'spec_digest': os.environ['SR_SPEC_DIGEST'],
                'metrics': metrics, 'artifacts': artifacts,
                'validator': {'name': 'v3_pilot_smoke', 'passed': True}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'tests_exit': int(code), 'train_steps': metrics['train.steps'],
                      'train_tokens': metrics['train.tokens'], 'reload_identical': metrics['reload.identical'],
                      'elapsed_s': metrics['smoke.elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
