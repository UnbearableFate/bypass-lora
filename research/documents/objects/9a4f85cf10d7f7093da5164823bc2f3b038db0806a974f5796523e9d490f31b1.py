"""Read-only paired analysis for the eight E-0003 V1 runs."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
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
CONFIG_ARMS = {
    'V1-01': 'F_out', 'V1-02': 'F_pre', 'V1-03': 'C_gate', 'V1-04': 'C_up',
    'V1-05': 'C_dual', 'V1-06': 'linear', 'V1-07': 'lora', 'V1-08': 'lora28',
}
TASKS = ('piqa', 'boolq')
BOOTSTRAP_SEED = 20261029
BOOTSTRAP_REPEATS = 10000
NEAR_TIE = .01
LOCKED_SPLIT_SHA256 = '47cf5c777d07863b8def6903487697eafc970b0cd5b5d80babb7f20ded98571b'
TRAIN_ORDER_SHA256 = 'fcfed383efd9a1672f6e210b8eb52fc2a6039db5c6d868926fb2ddec0e49d376'
FULL_STEPS = 13077
FULL_INPUT_TOKENS = 2000040
FULL_EXAMPLES = 26154
SOURCE_HASHES = {
    'piqa': 'f805f9c96518d7b59f2f870d90f712457cda949979469d9fa3fd706ce22e0a1a',
    'piqa_labels': 'df8ae080c41f02f88e81f6932d425f4b7ee72f433ad552e6a30d45e358e8f8bb',
    'boolq': '4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623',
}
SOURCE_METADATA = {
    'piqa': {'repo': 'ybisk/ybisk.github.io', 'revision': '21edab439af693b961be2f069e8690a88d3b4e37'},
    'boolq': {'repo': 'google/boolq', 'revision': '35b264d03638db9f4ce671b711558bf7ff0f80d5'},
}


class AnalysisError(ValueError):
    """Input runs cannot support a paired comparison."""


def _read_json(path: Path, run_id: str, label: str):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f'{run_id}: cannot read {label} at {path}: {exc}') from exc


def _validate_formal_artifacts(path: Path, run_id: str, expected_arm: str):
    config = _read_json(path / 'config.json', run_id, 'config.json')
    metadata = _read_json(path / 'data-metadata.json', run_id, 'data-metadata.json')
    split_ids = _read_json(path / 'split-map.json', run_id, 'split-map.json')
    if not isinstance(config, dict) or not isinstance(metadata, dict) or not isinstance(split_ids, dict):
        raise AnalysisError(f'{run_id}: config, metadata and split-map must be JSON objects')
    if config.get('arm') != expected_arm:
        raise AnalysisError(f'{run_id}: config arm {config.get("arm")!r} != expected {expected_arm!r}')
    if config.get('e3') is not True:
        raise AnalysisError(f'{run_id}: config must have e3=true')
    if config.get('smoke_steps') is not None or config.get('eval_limit') is not None:
        raise AnalysisError(f'{run_id}: smoke_steps and eval_limit must be unset')
    if config.get('batch') != 2 or config.get('tokens') != 2000000:
        raise AnalysisError(f'{run_id}: config batch/tokens do not match E-0003 full run')
    if run_id != 'Base' and config.get('lr') != .0003:
        raise AnalysisError(f'{run_id}: config learning rate must be 0.0003')

    try:
        canonical_split = json.dumps(split_ids, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f'{run_id}: split-map is not canonical JSON data') from exc
    split_hash = hashlib.sha256(canonical_split).hexdigest()
    if split_hash != LOCKED_SPLIT_SHA256:
        raise AnalysisError(f'{run_id}: split-map hash is not the locked E-0003 mapping')
    if metadata.get('split_mapping_sha256') != LOCKED_SPLIT_SHA256:
        raise AnalysisError(f'{run_id}: metadata split_mapping_sha256 is not locked')
    if metadata.get('split_ids') != split_ids:
        raise AnalysisError(f'{run_id}: metadata split_ids differ from split-map.json')
    if metadata.get('source_files', {}) == {}:
        raise AnalysisError(f'{run_id}: metadata source_files is missing')
    actual_source_hashes = {name: metadata.get('source_files', {}).get(name, {}).get('sha256')
                            for name in SOURCE_HASHES}
    if actual_source_hashes != SOURCE_HASHES:
        raise AnalysisError(f'{run_id}: source file hashes differ from v1_run.prepare')
    if metadata.get('source_metadata') != SOURCE_METADATA:
        raise AnalysisError(f'{run_id}: source repository revisions differ from v1_run.prepare')
    if metadata.get('dev_size') != 1000:
        raise AnalysisError(f'{run_id}: metadata dev_size must be 1000')
    for task in TASKS:
        if len(split_ids.get(task, {}).get('dev', [])) != 1000:
            raise AnalysisError(f'{run_id}: split map must contain 1000 {task} dev IDs')
    return config, metadata, split_ids


def _read_run(path: Path, run_id: str, expected_arm: str) -> tuple[list[dict], dict]:
    config, metadata, split_ids = _validate_formal_artifacts(path, run_id, expected_arm)
    manifest = _read_json(path / 'manifest.json', run_id, 'manifest.json')
    rows = _read_json(path / 'predictions.json', run_id, 'predictions.json')
    if not isinstance(manifest, dict):
        raise AnalysisError(f'{run_id}: manifest.json must contain an object')
    if manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True:
        raise AnalysisError(f'{run_id}: run is not complete and validator-passed')
    metrics = manifest.get('metrics', {})
    if not isinstance(metrics, dict):
        raise AnalysisError(f'{run_id}: manifest metrics must be an object')
    if metrics.get('reload.identical') is not True:
        raise AnalysisError(f'{run_id}: reload.identical must be true')
    if metrics.get('train.diverged') is True or metrics.get('run.failed_lr') is True:
        raise AnalysisError(f'{run_id}: training diverged or failed its learning rate')
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise AnalysisError(f'{run_id}: manifest metric {key} is nonfinite')
    for key, expected in (('plan.steps', FULL_STEPS), ('plan.input_tokens', FULL_INPUT_TOKENS),
                          ('plan.order_sha256', TRAIN_ORDER_SHA256)):
        if metrics.get(key) != expected:
            raise AnalysisError(f'{run_id}: {key} does not match the locked full schedule')
    if run_id == 'Base':
        for key in ('train.steps', 'train.input_tokens', 'train.examples'):
            if metrics.get(key, 0) not in (0, None):
                raise AnalysisError(f'Base: unexpected training consumption {key}={metrics[key]}')
        if metrics.get('params.trainable') != 0:
            raise AnalysisError('Base: params.trainable must be zero')
        for key in ('params.adapter_stored', 'params.with_grad'):
            if metrics.get(key, 0) not in (0, None):
                raise AnalysisError(f'Base: {key} must be zero when present')
    else:
        for key, expected in (('train.steps', FULL_STEPS), ('train.input_tokens', FULL_INPUT_TOKENS),
                              ('train.examples', FULL_EXAMPLES),
                              ('params.adapter_stored', TRAINABLE_PARAMS[run_id]),
                              ('params.trainable', TRAINABLE_PARAMS[run_id]),
                              ('params.with_grad', TRAINABLE_PARAMS[run_id])):
            if metrics.get(key) != expected:
                raise AnalysisError(f'{run_id}: {key}={metrics.get(key)!r}, expected {expected}')
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
    for task in TASKS:
        actual_ids = [row['id'] for row in dev if row['task'] == task]
        if actual_ids != split_ids[task]['dev']:
            raise AnalysisError(f'{run_id}: {task} dev IDs/membership/order differ from locked split map')
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


def _macro_accuracy_fraction(data: Mapping[str, Mapping[str, np.ndarray]]) -> Fraction:
    """Exact macro accuracy from integer correct counts and task sample sizes."""
    task_scores = [Fraction(int(data[task]['correct'].sum()), len(data[task]['correct']))
                   for task in TASKS]
    return sum(task_scores, Fraction()) / len(TASKS)


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
    """Validate formal E-0003 artifacts, then analyze the eight arms and Base."""
    missing = set(ARM_IDS) - set(run_paths)
    extra = set(run_paths) - set(ARM_IDS)
    if missing or extra:
        raise AnalysisError(f'run IDs must be exactly {ARM_IDS}; missing={sorted(missing)}, extra={sorted(extra)}')
    ordered_ids = (*ARM_IDS, 'Base')
    paths = {arm: Path(run_paths[arm]) for arm in ARM_IDS}
    paths['Base'] = Path(base_path)
    try:
        resolved = {arm: paths[arm].resolve(strict=True) for arm in ordered_ids}
    except OSError as exc:
        raise AnalysisError(f'run path does not exist or cannot be resolved: {exc}') from exc
    if len(set(resolved.values())) != len(ordered_ids):
        raise AnalysisError('each V1 arm and Base must use a unique resolved run directory')
    if any(not path.is_dir() for path in resolved.values()):
        raise AnalysisError('all run paths must resolve to directories')
    loaded = {arm: _read_run(resolved[arm], arm,
                             'base' if arm == 'Base' else CONFIG_ARMS[arm])
              for arm in ordered_ids}
    return _analyze_loaded(loaded, resolved, repeats=repeats, seed=seed, chunk_size=chunk_size)


def _analyze_loaded(loaded: Mapping[str, tuple[list[dict], dict]], paths: Mapping[str, Path], *,
                    repeats: int = BOOTSTRAP_REPEATS, seed: int = BOOTSTRAP_SEED,
                    chunk_size: int = 128) -> dict:
    """Numerical analysis for already-loaded rows; internal fixture helper, no I/O validation."""
    ordered_ids = (*ARM_IDS, 'Base')
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

    exact_scores = {arm: _macro_accuracy_fraction(data) for arm, data in zip(ordered_ids, arrays)}
    one_pp = Fraction(1, 100)
    f_delta = exact_scores['V1-01'] - exact_scores['V1-02']
    f_star = 'V1-01' if abs(f_delta) < one_pp or f_delta > 0 else 'V1-02'
    c_ids = ('V1-03', 'V1-04', 'V1-05')
    c_scores = {arm: metrics[arm]['macro_accuracy'] for arm in c_ids}
    c_best_exact = max(exact_scores[arm] for arm in c_ids)
    c_near_best = [arm for arm in c_ids if c_best_exact - exact_scores[arm] < one_pp]
    effective = [arm for arm in c_ids if exact_scores[arm] >= exact_scores['Base'] + one_pp]
    effective_best_exact = max((exact_scores[arm] for arm in effective), default=None)
    effective_near_best = ([arm for arm in effective
                            if effective_best_exact - exact_scores[arm] < one_pp]
                           if effective_best_exact is not None else [])
    smallest_params = min((TRAINABLE_PARAMS[arm] for arm in effective_near_best), default=None)
    simplest_effective = ([arm for arm in effective_near_best
                           if TRAINABLE_PARAMS[arm] == smallest_params]
                          if smallest_params is not None else [])
    # Sequential deterministic tie rules: highest exact macro accuracy, then lower
    # point-estimate macro NLL, then C_gate. The parameter filter above precedes them.
    c_star = (min(simplest_effective,
                  key=lambda arm: (-exact_scores[arm], metrics[arm]['macro_nll'],
                                   0 if arm == 'V1-03' else 1))
              if simplest_effective else None)
    best_concat = max(exact_scores[arm] for arm in c_ids)
    concat_wins = c_star is not None and best_concat >= exact_scores[f_star] + one_pp
    dual_better_than_gate = exact_scores['V1-05'] >= exact_scores['V1-03'] + one_pp
    base_piqa_accuracy = Fraction(int(arrays[-1]['piqa']['correct'].sum()),
                                  len(arrays[-1]['piqa']['correct']))
    piqa_no_discrimination = all(
        abs(Fraction(int(data['piqa']['correct'].sum()), len(data['piqa']['correct']))
            - base_piqa_accuracy) <= one_pp
        for data in arrays[:-1])
    return {
        'experiment': 'E-0003', 'arms': {arm: {'name': ARM_NAMES.get(arm, 'Base'),
            'path': str(paths[arm]), 'metrics': metrics[arm],
            'manifest_metrics': loaded[arm][1].get('metrics', {}),
            'dev_examples': {task: len(arrays[index][task]['correct']) for task in TASKS}}
            for index, arm in enumerate(ordered_ids)},
        'comparisons': comparisons,
        'bootstrap': {'method': 'paired task-stratified row resampling', 'repeats': repeats,
                      'seed': seed, 'interval': 'percentile 95%', 'chunk_size': chunk_size,
                      'difference_sign': 'left arm minus right arm'},
        'selection': {
            'near_tie_threshold': NEAR_TIE,
            'PIQA_no_discrimination': piqa_no_discrimination,
            'PIQA_no_discrimination_rule': 'true iff every training arm PIQA accuracy is within inclusive ±1 pp of Base',
            'F_star': {'selected': f_star, 'rule': 'higher macro accuracy; if within 1 pp select F_out'},
            'C_star': c_star,
            'C_effective': effective,
            'C_effectiveness_rule': 'macro accuracy >= Base + 1 pp',
            'C_scores': c_scores,
            'C_near_best_within_1pp': c_near_best,
            'C_effective_near_best_within_1pp': effective_near_best,
            'C_simplest_effective_near_best': simplest_effective,
            'C_parameter_counts': {arm: TRAINABLE_PARAMS[arm] for arm in c_ids},
            'C_star_tie_rules': ['among effective arms, macro accuracy within strictly <1 pp of the best',
                                 'fewest trainable parameters', 'higher macro accuracy',
                                 'lower macro NLL', 'C_gate'],
            'concat_wins': concat_wins,
            'concat_wins_rule': 'C_star exists and best concat macro accuracy >= F_star + 1 pp',
            'C_dual_better_than_C_gate': dual_better_than_gate,
            'C_dual_better_rule': 'C_dual macro accuracy >= C_gate + 1 pp',
            'decision_accuracy_arithmetic': 'exact rational correct-count/sample-size fractions; thresholds are inclusive unless stated strictly <1 pp',
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
