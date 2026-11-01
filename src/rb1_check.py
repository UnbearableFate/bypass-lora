"""PBS phase-one checks; only synthetic examples/outcomes, no RB1 result input."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    files = ['src/test_core.py', 'src/test_rb_core.py', 'src/test_rb_data.py', 'src/test_rb_train.py',
             'src/test_rb_read_diagnostics.py', 'src/test_rb1_rules.py', 'src/test_rb1_artifacts.py',
             'src/test_rb1_latency.py', 'src/test_rb1_analysis.py', 'src/test_rb1_intervene.py']
    start = time.monotonic(); passed = False; metrics = {}
    try:
        with (out/'checks.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'pytest', *files, '-q', '-p', 'no:cacheprovider',
                '-o', 'junit_family=legacy', '--junitxml='+str(out/'checks.xml')], stdout=log, stderr=subprocess.STDOUT)
        root = ET.parse(out/'checks.xml').getroot()
        cases = list(root.iter('testcase'))
        metrics = {'checks.exit_code': result.returncode, 'checks.cases': len(cases),
                   'checks.failed': sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
                   'checks.skipped': sum(c.find('skipped') is not None for c in cases),
                   'scope': 'synthetic exact-rule/full-artifact analysis;normalized read slots;actual shared loop;real model synthetic-text V1 identity,all four-layer interventions,Base/off and T0 no-op',
                   'formal_RB1_results_read': False, 'confirmation_evaluated': False}
        passed = result.returncode == 0 and bool(cases) and metrics['checks.failed'] == metrics['checks.skipped'] == 0
        if not passed:
            raise RuntimeError('RB1 phase-one PBS checks did not pass; see checks.log/checks.xml')
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({'status': 'complete' if passed else 'partial',
            'contract': 'E-0021@1', 'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'source_commit': os.environ['SR_CODE_COMMIT'], 'metrics': metrics,
            'artifacts': [{'path': str(p), 'role': p.name} for p in (out/'checks.xml', out/'checks.log') if p.is_file()],
            'validator': {'name': 'rb1_phase_one_checks', 'passed': passed}}, indent=2)+'\n')


if __name__ == '__main__':
    main()
