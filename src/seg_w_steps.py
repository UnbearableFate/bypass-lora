"""E-0032 diagnostic (read-only): training-loss trajectories of the seed-0 arms.

Reads each run's steps.jsonl and manifest.json and reports, per run: the step
count, 100-step window means of example_mean_nll, the largest jump between
consecutive windows, the preclip gradient-norm maximum and the windows where it
exceeds SPIKE_NORM, the mean NLL over the last 2,000 steps, and the dev
accuracies/NLL recorded in the manifest. Runs that share a data order (same
seed) are also compared window by window against a named partner. No rule
reads these numbers; they describe training only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

from rb1_artifacts import read, reference
from v5_train import write_json

WINDOW = 100
LAST = 2000
SPIKE_NORM = 50.0


def windows(values):
    return [sum(values[i:i+WINDOW])/len(values[i:i+WINDOW]) for i in range(0, len(values), WINDOW)]


def describe(root):
    steps = [json.loads(line) for line in (root/'steps.jsonl').read_text().splitlines() if line.strip()]
    numbers = [s['step'] for s in steps]
    if numbers != list(range(1, len(steps)+1)):
        raise ValueError(f'{root}: steps are not contiguous from 1')
    nll = [float(s['example_mean_nll']) for s in steps]
    norm = [float(s['preclip_norm']) for s in steps]
    if not all(math.isfinite(x) for x in nll+norm):
        raise ValueError(f'{root}: nonfinite step record')
    w = windows(nll)
    jumps = [(w[i+1]-w[i], i+1) for i in range(len(w)-1)]
    jump, at = max(jumps)
    peak = max(range(len(norm)), key=norm.__getitem__)
    manifest = read(root/'manifest.json'); m = manifest['metrics']
    spikes = sorted({i//WINDOW*WINDOW+1 for i, x in enumerate(norm) if x > SPIKE_NORM})
    return {'arm': manifest.get('arm'), 'seed': manifest.get('seed'), 'contract': manifest.get('contract'),
            'hardware': m.get('hardware'), 'steps': len(steps),
            'window_mean_nll': w,
            'largest_window_jump': {'nats': jump, 'from_window_start': (at-1)*WINDOW+1, 'to_window_start': at*WINDOW+1,
                                    'before': w[at-1], 'after': w[at]},
            'preclip_norm_max': {'value': norm[peak], 'step': peak+1},
            f'windows_with_preclip_norm_gt_{int(SPIKE_NORM)}': spikes,
            'last2000_mean_nll': sum(nll[-LAST:])/len(nll[-LAST:]),
            'dev': {k: m.get(k) for k in ('dev.boolq.accuracy', 'dev.piqa.accuracy', 'dev.boolq.nll', 'dev.piqa.nll')},
            'steps_ref': reference(root/'steps.jsonl'), 'manifest_ref': reference(root/'manifest.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--data-dir')    # accepted for the shared PBS stage interface; unused
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, values, start = False, {}, time.monotonic()
    try:
        spec = read(args.input); values['input'] = reference(args.input)
        runs = {label: describe(Path(root)) for label, root in spec['runs'].items()}
        pairs = {}
        for label, partner in spec.get('pairs', {}).items():
            a, b = runs[label]['window_mean_nll'], runs[partner]['window_mean_nll']
            if len(a) != len(b):
                raise ValueError('paired runs differ in length')
            diff = [x-y for x, y in zip(a, b)]
            pairs[f'{label} - {partner}'] = {'window_diff_nll': diff,
                                            'first_window_start_diff_gt_0.5': next(((i*WINDOW+1) for i, d in enumerate(diff) if d > .5), None),
                                            'last2000_diff_nll': runs[label]['last2000_mean_nll']-runs[partner]['last2000_mean_nll']}
        write_json(out/'steps-diagnostic.json', {'definition': __doc__, 'window': WINDOW, 'last': LAST,
                                                 'spike_norm': SPIKE_NORM, 'runs': runs, 'pairs': pairs})
        values['runs'] = len(runs)
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0032@1',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'source_commit': os.environ['SR_CODE_COMMIT'],
            'stage': 'steps_diagnostic', 'metrics': values,
            'artifacts': [{'path': str(out/'steps-diagnostic.json'), 'role': 'steps-diagnostic.json'}]
                         if (out/'steps-diagnostic.json').exists() else [],
            'validator': {'name': 'e0032_steps_diagnostic', 'passed': passed}})


if __name__ == '__main__':
    main()
