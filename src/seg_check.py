"""Post-review PBS regressions: protocol identities and synthetic analysis only.

No training run or SEG dev quality is read. The unchanged training/scoring
runtime already has ATT-0271-1's seven-arm cost smoke. Re-run the deterministic
protocol at this exact source so later rule-1(c) provenance has the formal pin.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from v5_train import write_json

FILES = ['src/test_seg_rules.py', 'src/test_seg_data.py', 'src/test_seg_model.py',
         'src/test_seg_artifacts.py', 'src/test_seg_analysis.py', 'src/test_core.py', 'src/test_rb_core.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    metrics, passed, tick = {}, False, time.monotonic()
    try:
        with (out/'checks.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'pytest', *FILES, '-q', '-p', 'no:cacheprovider',
                '-o', 'junit_family=legacy', '--junitxml='+str(out/'checks.xml')], stdout=log, stderr=subprocess.STDOUT)
        cases = list(ET.parse(out/'checks.xml').getroot().iter('testcase'))
        metrics.update({'checks.cases': len(cases),
            'checks.failed': sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
            'checks.skipped': sum(c.find('skipped') is not None for c in cases),
            'no_channel_identity': True, 'differentiable_state_carry': True,
            'synthetic_reducer_boundaries': True, 'formal_SEG_results_read': False,
            'confirmation_evaluated': False})
        passed = result.returncode == 0 and bool(cases) and metrics['checks.failed'] == metrics['checks.skipped'] == 0
        if not passed:
            raise RuntimeError('segmented PBS regression failed; inspect checks.log/checks.xml')
    finally:
        metrics['elapsed_s'] = time.monotonic()-tick
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0025@1',
            'source_commit': os.environ['SR_CODE_COMMIT'], 'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'stage': 'deterministic_regression', 'metrics': metrics,
            'artifacts': [{'path': str(p), 'role': p.name} for p in (out/'checks.log', out/'checks.xml') if p.exists()],
            'validator': {'name': 'segmented_post_review_checks', 'passed': passed}})


if __name__ == '__main__':
    main()
