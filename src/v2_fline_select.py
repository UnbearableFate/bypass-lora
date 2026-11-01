"""E-0008 calibration-only LR choice and single endpoint extension decision."""
import argparse
import json
import math
from pathlib import Path
from v2_schedule import select_learning_rate

INITIAL = (1e-4, 3e-4, 1e-3)


def candidate(report):
    metrics = report['metrics']
    if report['smoke'] or report['tokens_budget'] != 200000 or report['eval_batch'] != 16:
        raise ValueError('only full-budget calibration runs may select LR')
    failed = metrics['failed_lr']
    if failed:
        if metrics.get('failure.stage') not in ('training', 'original_evaluation'):
            raise ValueError('invalid scientific failure stage')
        nll = float('inf')
    else:
        if (metrics.get('train.steps') != 300 or metrics.get('train.tokens') != 200170
                or metrics.get('calibration.all.examples') != 2048 or not metrics.get('reload.identical')):
            raise ValueError('incomplete or unverified calibration')
        nll = metrics['calibration.all.nll']
        if not math.isfinite(nll):
            raise ValueError('finite run has nonfinite NLL')
    return {'lr': report['lr'], 'nll': nll, 'finite_loss': not failed}


def select(reports, extension=None):
    candidates = [candidate(r) for r in reports]
    if sorted(r['lr'] for r in candidates) != list(INITIAL):
        raise ValueError('require exactly the three initial rates')
    best = select_learning_rate(candidates)
    extra = best['lr'] / 3 if best['lr'] == INITIAL[0] else best['lr'] * 3 if best['lr'] == INITIAL[-1] else None
    if extension is not None:
        other = candidate(extension)
        if extra is None or not math.isclose(other['lr'], extra, rel_tol=1e-12):
            raise ValueError('unexpected endpoint extension')
        candidates.append(other)
        best = select_learning_rate(candidates)
    return {'selected_lr': best['lr'], 'selected_calibration_nll': best['nll'],
            'extension_lr': extra, 'extension_required': extra is not None and extension is None,
            'selection_final': extra is None or extension is not None,
            'candidates': [{**r, 'nll': r['nll'] if math.isfinite(r['nll']) else None} for r in candidates]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs=3, type=Path, required=True)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = select([json.loads(p.read_text()) for p in args.reports],
                    json.loads(args.extension.read_text()) if args.extension else None)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
