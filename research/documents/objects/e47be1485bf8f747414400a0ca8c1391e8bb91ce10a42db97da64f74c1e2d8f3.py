"""PBS regression entry point for the RV-0007 formal-input validation fix."""
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
    src = Path(__file__).resolve().parent
    start = time.monotonic()
    passed = False
    metrics = {'train.input_tokens': 0}
    try:
        code = pytest.main(['-v', '-p', 'no:cacheprovider',
                            '--basetemp='+str(out/'tmp'), '--junitxml='+str(out/'checks.xml'),
                            *[str(src/name) for name in ('test_core.py','test_model.py',
                              'test_v1_data.py','test_v1_lora.py','test_v1_run.py',
                              'test_v1_analysis.py')]])
        metrics['pytest.exit_code'] = int(code)
        suite = ET.parse(out/'checks.xml').getroot().find('testsuite')
        metrics.update({'pytest.'+key: int(suite.attrib[key])
                        for key in ('tests','failures','errors','skipped')})
        passed = code == 0
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({
            'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'], 'metrics':metrics,
            'artifacts':[{'path':str(out/'checks.xml'),'role':'pytest_results'}],
            'validator':{'name':'v1_analysis_regression','passed':passed}},indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
