"""PBS entry point for the E-0010 analysis regressions, with a result manifest."""
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import pytest


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started, passed = time.monotonic(), False
    metrics = {}
    try:
        code = pytest.main(['-v', '-p', 'no:cacheprovider', '--basetemp='+str(out/'tmp'),
                           '--junitxml='+str(out/'checks.xml'),
                           str(Path(__file__).with_name('test_v2_factorial_analysis.py'))])
        metrics['pytest.exit_code'] = int(code)
        suite = ET.parse(out/'checks.xml').getroot().find('testsuite')
        metrics.update({'pytest.'+k: int(suite.attrib[k]) for k in ('tests', 'failures', 'errors', 'skipped')})
        passed = code == 0
    finally:
        metrics['elapsed_s'] = time.monotonic()-started
        (out/'manifest.json').write_text(json.dumps(dict(status='complete' if passed else 'failed',
            spec_digest=os.environ['SR_SPEC_DIGEST'], metrics=metrics,
            artifacts=[dict(path=str(out/'checks.xml'), role='pytest_results')],
            validator=dict(name='E-0010_analysis_regression', passed=passed)), indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
