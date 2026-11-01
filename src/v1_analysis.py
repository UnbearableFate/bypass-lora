"""Read-only paired analysis for the eight E-0003 V1 runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


ARM_IDS = tuple(f'V1-{index:02d}' for index in range(1, 9))
ARM_NAMES = {
    'V1-01': 'F_out', 'V1-02': 'F_pre',
    'V1-03': 'C_gate', 'V1-04': 'C_up', 'V1-05': 'C_dual',
    'V1-06': 'linear', 'V1-07': 'lora4_r10', 'V1-08': 'lora28_r2',
}
TRAINABLE_PARAMS = {
    'V1-01': 262144, 'V1-02': 262144,
    'V1-03': 524288, 'V1-04': 524288, 'V1-05': 917504,
    'V1-06': 917504, 'V1-07': 901120, 'V1-08': 1261568,
}
TASKS = ('piqa', 'boolq')
BOOTSTRAP_SEED = 20261029
BOOTSTRAP_REPEATS = 10000
NEAR_TIE = .01


class AnalysisError(ValueError):
    """Input runs cannot support a paired comparison."""


def _read_run(path: Path, run_id: str) -> tuple[list[dict], dict]:
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
        rows = json.loads((path / 'predictions.json').read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f'{run_id}: cannot read manifest/predictions in {path}: {exc}') from exc
    if manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True:
        raise AnalysisError(f'{run_id}: run is not complete and validator-passed')
    if manifest.get('metrics', {}).get('reload.identical') is False:
        raise AnalysisError(f'{run_id}: reload evaluation is not identical')
    for key, value in manifest.get('metrics', {}).items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise AnalysisError(f'{run_id}: manifest metric {key} is nonfinite')
    if not isinstance(rows, list):
        raise AnalysisError(f'{run_id}: predictions.json must contain a list')
    dev = [row for row in rows if row.get('split') == 'dev']
    if not dev:
        raise AnalysisError(f'{run_id}: no dev prediction rows')
    for row in dev:
        if row.get('task') not in TASKS:
            raise AnalysisError(f'{run_id}: unexpected task {row.get("task")!r}')
        for key in ('id', 'gold', 'prediction', 'gold_nll_sum', 'gold_loss_tokens'):
            if key not in row:
                raise AnalysisError(f'{run_id}: dev row missing {key}')
        if row['gold'] not in (0, 1) or row['prediction'] not in (0, 1):
            raise AnalysisError(f'{run_id}: gold and prediction must be binary')
        if not isinstance(row['gold_loss_tokens'], int) or row['gold_loss_tokens'] <= 0:
            raise AnalysisError(f'{run_id}: gold_loss_tokens must be a positive integer')
        if not math.isfinite(float(row['gold_nll_sum'])) or float(row['gold_nll_sum']) < 0:
            raise AnalysisError(f'{run_id}: gold_nll_sum must be finite and nonnegative')
        if not isinstance(row.get('scores'), list) or not row['scores'] or any(
                not math.isfinite(float(score)) for score in row['scores']):
            raise AnalysisError(f'{run_id}: candidate scores must be a nonempty finite list')
    identities = [(row['task'], row['id'], row['gold']) for row in dev]
    if len(set((task, ident) for task, ident, _ in identities)) != len(identities):
        raise AnalysisError(f'{run_id}: duplicate (task, id) in dev predictions')
    if set(row['task'] for row in dev) != set(TASKS):
        raise AnalysisError(f'{run_id}: dev rows must contain both tasks')
    return dev, manifest


def _arrays(rows: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for task in TASKS:
        subset = [row for row in rows if row['task'] == task]
        result[task] = {
            'correct': np.asarray([row['prediction'] == row['gold'] for row in subset], dtype=np.float64),
            'nll_sum': np.asarray([row['gold_nll_sum'] for row in subset], dtype=np.float64),
            'tokens': np.asarray([row['gold_loss_tokens'] for row in subset], dtype=np.float64),
        }
    return result


def _point_metrics(data: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, float]:
    result = {}
    for task in TASKS:
        result[f'{task}_accuracy'] = float(data[task]['correct'].mean())
        result[f'{task}_nll'] = float(data[task]['nll_sum'].sum() / data[task]['tokens'].sum())
    result['macro_accuracy'] = (result['piqa_accuracy'] + result['boolq_accuracy']) / 2
    result['macro_nll'] = (result['piqa_nll'] + result['boolq_nll']) / 2
    return result


def _paired_bootstrap(arrays: list[dict], repeats: int, seed: int, chunk_size: int) -> dict:
    """Paired, task-stratified row bootstrap; every arm gets the same sampled indices."""
    if repeats <= 0 or chunk_size <= 0:
        raise ValueError('repeats and chunk_size must be positive')
    pair_indices = [(left, right) for left in range(len(arrays)) for right in range(left + 1, len(arrays))]
    metric_names = ('macro_accuracy', 'piqa_accuracy', 'boolq_accuracy',
                    'piqa_nll', 'boolq_nll', 'macro_nll')
    boot_diffs = np.empty((repeats, len(pair_indices), len(metric_names)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    sizes = {task: len(arrays[0][task]['correct']) for task in TASKS}
    for idx, data in enumerate(arrays):
        for task in TASKS:
            if len(data[task]['correct']) != sizes[task]:
                raise AnalysisError('paired bootstrap arrays have inconsistent task sizes')
    for begin in range(0, repeats, chunk_size):
        end = min(repeats, begin + chunk_size)
        count = end - begin
        sampled = {task: rng.integers(0, sizes[task], size=(count, sizes[task])) for task in TASKS}
        # At most one chunk of sampled indices and its metrics live in memory.
        values = np.empty((len(arrays), count, len(metric_names)), dtype=np.float64)
        for arm_index, data in enumerate(arrays):
            task_accuracy = {}
            task_nll = {}
            for task in TASKS:
                ix = sampled[task]
                task_accuracy[task] = data[task]['correct'][ix].mean(axis=1)
                numerator = data[task]['nll_sum'][ix].sum(axis=1)
                denominator = data[task]['tokens'][ix].sum(axis=1)
                task_nll[task] = numerator / denominator
            values[arm_index, :, :] = np.column_stack((
                (task_accuracy['piqa'] + task_accuracy['boolq']) / 2,
                task_accuracy['piqa'], task_accuracy['boolq'],
                task_nll['piqa'], task_nll['boolq'],
                (task_nll['piqa'] + task_nll['boolq']) / 2,
            ))
        for pair_index, (left, right) in enumerate(pair_indices):
            boot_diffs[begin:end, pair_index, :] = values[left] - values[right]
    comparisons = {}
    for pair_index, (left, right) in enumerate(pair_indices):
        comparison = {}
        for metric_index, metric in enumerate(metric_names):
            samples = boot_diffs[:, pair_index, metric_index]
            comparison[metric] = {
                'difference': float(_point_metrics(arrays[left])[metric] - _point_metrics(arrays[right])[metric]),
                'ci95': [float(v) for v in np.quantile(samples, [.025, .975])],
            }
        comparisons[f'{left}:{right}'] = comparison
    return comparisons


def analyze(run_paths: Mapping[str, str | Path], base_path: str | Path, *,
            repeats: int = BOOTSTRAP_REPEATS, seed: int = BOOTSTRAP_SEED,
            chunk_size: int = 128) -> dict:
    """Analyze eight V1 run directories and the E-0002 Base reference directory."""
    missing = set(ARM_IDS) - set(run_paths)
    extra = set(run_paths) - set(ARM_IDS)
    if missing or extra:
        raise AnalysisError(f'run IDs must be exactly {ARM_IDS}; missing={sorted(missing)}, extra={sorted(extra)}')
    ordered_ids = (*ARM_IDS, 'Base')
    paths = {arm: Path(run_paths[arm]) for arm in ARM_IDS}
    paths['Base'] = Path(base_path)
    loaded = {arm: _read_run(paths[arm], arm) for arm in ordered_ids}
    reference_rows = loaded[ARM_IDS[0]][0]
    reference_identity = [(row['task'], row['id'], row['gold']) for row in reference_rows]
    for arm in ordered_ids[1:]:
        identity = [(row['task'], row['id'], row['gold']) for row in loaded[arm][0]]
        if identity != reference_identity:
            raise AnalysisError(f'{arm}: dev task/id/gold rows or order differ from {ARM_IDS[0]}')
    arrays = [_arrays(loaded[arm][0]) for arm in ordered_ids]
    metrics = {arm: _point_metrics(data) for arm, data in zip(ordered_ids, arrays)}
    comparisons = _paired_bootstrap(arrays, repeats, seed, chunk_size)
    # Pair labels are IDs in stable input order, avoiding opaque list-index references.
    comparisons = {
        f'{ordered_ids[int(pair.split(":")[0])]} - {ordered_ids[int(pair.split(":")[1])]}': value
        for pair, value in comparisons.items()
    }

    f_diff = metrics['V1-01']['macro_accuracy'] - metrics['V1-02']['macro_accuracy']
    f_star = 'V1-01' if f_diff > -NEAR_TIE else 'V1-02'
    c_ids = ('V1-03', 'V1-04', 'V1-05')
    c_scores = {arm: metrics[arm]['macro_accuracy'] for arm in c_ids}
    c_best = max(c_scores.values())
    c_near_best = [arm for arm in c_ids if c_best - c_scores[arm] < NEAR_TIE]
    smallest_params = min(TRAINABLE_PARAMS[arm] for arm in c_near_best)
    simplest_near_best = [arm for arm in c_near_best if TRAINABLE_PARAMS[arm] == smallest_params]
    return {
        'experiment': 'E-0003', 'arms': {arm: {'name': ARM_NAMES.get(arm, 'Base'),
            'path': str(paths[arm]), 'metrics': metrics[arm],
            'dev_examples': {task: len(arrays[index][task]['correct']) for task in TASKS}}
            for index, arm in enumerate(ordered_ids)},
        'comparisons': comparisons,
        'bootstrap': {'method': 'paired task-stratified row resampling', 'repeats': repeats,
                      'seed': seed, 'interval': 'percentile 95%', 'chunk_size': chunk_size,
                      'difference_sign': 'left arm minus right arm'},
        'selection': {
            'near_tie_threshold': NEAR_TIE,
            'F_star': {'selected': f_star, 'rule': 'higher macro accuracy; if within 1 pp select F_out'},
            'C_star': None,
            'C_star_status': 'coordinator_definition_needed',
            'C_star_reason': 'E-0003 says effective, but does not define an effectiveness criterion.',
            'C_scores': c_scores,
            'C_near_best_within_1pp': c_near_best,
            'C_simplest_near_best': simplest_near_best,
            'C_parameter_counts': {arm: TRAINABLE_PARAMS[arm] for arm in c_ids},
            'C_gate_up_tie_convention': 'If a downstream report requires one equal-parameter gate/up label, choose V1-03 by ascending run ID; this is a disclosed reporting convention, not an effectiveness decision.',
        },
        'notes': ['Single-seed screening results; bootstrap intervals describe paired dev-row variation and are not confirmatory.',
                  'Task NLL is total gold NLL sum divided by total gold loss tokens, matching v1_run metrics.'],
    }


def _parse_run(value: str) -> tuple[str, Path]:
    if '=' not in value:
        raise argparse.ArgumentTypeError('expected ID=PATH')
    run_id, path = value.split('=', 1)
    if run_id not in ARM_IDS or not path:
        raise argparse.ArgumentTypeError(f'expected one of {ARM_IDS}=PATH')
    return run_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Base run directory')
    parser.add_argument('--run', action='append', type=_parse_run, required=True, help='V1-ID=run-directory (repeat 8 times)')
    parser.add_argument('--output', type=Path, help='write JSON report here; defaults to stdout')
    parser.add_argument('--repeats', type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument('--seed', type=int, default=BOOTSTRAP_SEED)
    parser.add_argument('--chunk-size', type=int, default=128)
    args = parser.parse_args()
    run_paths = dict(args.run)
    report = analyze(run_paths, args.base, repeats=args.repeats, seed=args.seed, chunk_size=args.chunk_size)
    rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
