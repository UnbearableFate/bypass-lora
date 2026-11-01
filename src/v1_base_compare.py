"""Read-only PBS comparison of the full E-0003 and E-0002 Base dev artifacts."""
import json
import os
from pathlib import Path
import sys
import time

from v1_analysis import _read_run


def main():
    old, new, out = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    metrics = {'train.input_tokens': 0}
    passed = False
    try:
        new_rows, new_manifest = _read_run(new, 'Base', 'base')
        old_manifest = json.loads((old/'manifest.json').read_text())
        assert old_manifest['status'] == 'complete' and old_manifest['validator']['passed']
        assert old_manifest['metrics']['reload.identical'] is True
        old_rows = [row for row in json.loads((old/'predictions.json').read_text())
                    if row['split'] == 'dev']
        metrics['dev.predictions_identical'] = old_rows == new_rows
        old_metrics = {k:v for k,v in old_manifest['metrics'].items() if k.startswith('dev.')}
        new_metrics = {k:v for k,v in new_manifest['metrics'].items() if k.startswith('dev.')}
        metrics['dev.metrics_identical'] = old_metrics == new_metrics
        metrics['split_map.identical'] = json.loads((old/'split-map.json').read_text()) == json.loads((new/'split-map.json').read_text())
        metrics['dev.rows'] = len(new_rows)
        metrics.update(new_metrics)
        metrics['dev.macro_accuracy'] = (new_metrics['dev.piqa.accuracy'] + new_metrics['dev.boolq.accuracy'])/2
        assert metrics['dev.predictions_identical'] and metrics['dev.metrics_identical'] and metrics['split_map.identical']
        assert metrics['dev.rows'] == 2000
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({
            'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'], 'metrics':metrics,
            'artifacts':[{'path':str(new/'manifest.json'),'role':'new_base'},
                         {'path':str(old/'manifest.json'),'role':'old_base'}],
            'validator':{'name':'base_dev_exact_comparison','passed':passed}},indent=2)+'\n')


if __name__ == '__main__':
    main()
