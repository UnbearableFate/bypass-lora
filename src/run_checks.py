"""Run checks and always emit a result manifest, including on pytest failures."""
import json
import os
from pathlib import Path
import sys
import time
import pytest

class Results:
    def __init__(self):
        self.metrics = {}
    def pytest_runtest_logreport(self, report):
        key = 'check.' + report.nodeid
        if report.when == 'call' or report.failed:
            self.metrics[key + '.pass'] = report.passed
            for name,value in report.user_properties:
                self.metrics[key+'.'+name] = value

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=True)
results = Results()
start = time.monotonic()
code = pytest.main(['-v','-p','no:cacheprovider', '--junitxml='+str(out/'checks.xml'),
                    '--basetemp='+str(out/'tmp'), str(Path(__file__).parent/'test_core.py'),
                    str(Path(__file__).parent/'test_model.py'),
                    *sys.argv[2:]], plugins=[results])
results.metrics['elapsed_s'] = time.monotonic()-start
results.metrics['pytest.exit_code'] = int(code)
(out/'manifest.json').write_text(json.dumps({
    'status':'complete','spec_digest':os.environ['SR_SPEC_DIGEST'],
    'metrics':results.metrics,
    'artifacts':[{'path':str(out/'checks.xml'),'role':'pytest_results'}],
    'validator':{'name':'pytest','passed':code==0}},indent=2)+'\n')
sys.exit(code)
