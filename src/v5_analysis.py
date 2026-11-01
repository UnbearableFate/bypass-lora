"""Paired analysis for the E-0013 V5 confirmation (and dev rehearsal)."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ('V5-C', 'V5-F', 'V5-L')
TASKS = ('piqa', 'boolq')
SEEDS = (1, 2, 3)
BOOTSTRAP_SEED = 20261029
DEFAULT_REPETITIONS = 10000


class AnalysisError(ValueError):
    """Raised when prediction artifacts cannot support paired analysis."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f'cannot read JSON {path}: {exc}') from exc


def _read_predictions(path: Path) -> list[dict]:
    try:
        content = path.read_text()
    except OSError as exc:
        raise AnalysisError(f'cannot read predictions {path}: {exc}') from exc
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = []
        for line_no, line in enumerate(content.splitlines(), 1):
            if line.strip():
                try:
                    value.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AnalysisError(f'{path}:{line_no}: invalid JSONL: {exc}') from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise AnalysisError(f'{path}: predictions must be a JSON list or JSONL objects')
    return value


def _fraction(value: Fraction | int) -> dict[str, int]:
    f = Fraction(value)
    return {'numerator': f.numerator, 'denominator': f.denominator}


def _as_fraction(value: Fraction) -> str:
    return f'{value.numerator}/{value.denominator}'


def _resolve(root: Path, path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else root / result


def _load_manifest(manifest: Any) -> tuple[dict, Path]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        obj = _read_json(path)
        root = path.resolve().parent
    elif isinstance(manifest, dict):
        obj, root = manifest, Path.cwd()
    else:
        raise AnalysisError('manifest must be a path or object')
    entries = obj.get('entries', obj.get('runs'))
    if not isinstance(entries, list):
        raise AnalysisError('manifest requires an entries list')
    return {**obj, 'entries': entries}, root


def _load_metrics(path: Path | None) -> dict:
    if path is None:
        return {}
    obj = _read_json(path)
    if not isinstance(obj, dict):
        raise AnalysisError(f'{path}: metrics must be a JSON object')
    if 'metrics' in obj:
        if not isinstance(obj['metrics'], dict):
            raise AnalysisError(f'{path}: nested metrics must be an object')
        return {**{k: v for k, v in obj.items() if k != 'metrics'}, **obj['metrics']}
    return obj


def _get_wiki_nll(manifest: dict, root: Path, entry: dict, metrics: dict) -> float | None:
    arm = entry['arm']
    seed = entry['seed']
    wiki = manifest.get('wikitext', {})
    if isinstance(wiki, dict):
        arm_data = wiki.get(arm)
        if isinstance(arm_data, dict):
            direct = arm_data.get(seed, arm_data.get(str(seed))) if seed is not None else arm_data.get('nll')
            if isinstance(direct, dict):
                direct = direct.get('nll')
            if direct is not None:
                return float(direct)
    value = metrics.get('wikitext.nll')
    if value is None:
        value = metrics.get('wikitext', {}).get('nll') if isinstance(metrics.get('wikitext'), dict) else None
    return None if value is None else float(value)


def _parse_artifacts(manifest: dict, root: Path, split: str) -> tuple[dict, dict, dict]:
    entries = manifest['entries']
    indexed: dict[tuple[str, int | None], dict] = {}
    data, wiki = {}, {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise AnalysisError('each manifest entry must be an object')
        arm, seed = entry.get('arm'), entry.get('seed')
        if arm not in (*ARMS, 'Base'):
            raise AnalysisError(f'unknown arm {arm!r}')
        if arm == 'Base':
            if seed is not None:
                raise AnalysisError('Base entry must have seed null')
        elif type(seed) is not int or seed not in SEEDS:
            raise AnalysisError(f'{arm}: seed must be one of {SEEDS}')
        key = (arm, seed)
        if key in indexed:
            raise AnalysisError(f'duplicate manifest entry {key}')
        if not isinstance(entry.get('predictions'), str):
            raise AnalysisError(f'{key}: predictions path is required')
        indexed[key] = entry
        metrics_path = _resolve(root, entry['metrics']) if entry.get('metrics') else None
        metrics = _load_metrics(metrics_path)
        wiki_nll = _get_wiki_nll(manifest, root, entry, metrics)
        if wiki_nll is not None:
            if not math.isfinite(wiki_nll) or wiki_nll < 0:
                raise AnalysisError(f'{key}: wikitext NLL must be finite and nonnegative')
            wiki[key] = wiki_nll
        rows = _read_predictions(_resolve(root, entry['predictions']))
        chosen = [r for r in rows if r.get('split') == split]
        if not chosen:
            raise AnalysisError(f'{key}: no rows for split {split!r}')
        by_task = {}
        for row in chosen:
            task = row.get('task')
            if task not in TASKS:
                raise AnalysisError(f'{key}: unexpected task {task!r}')
            for field in ('id', 'gold', 'prediction', 'gold_nll_sum', 'gold_loss_tokens'):
                if field not in row:
                    raise AnalysisError(f'{key}: row missing {field}')
            ident = row['id']
            try:
                hash(ident)
            except TypeError as exc:
                raise AnalysisError(f'{key}: unhashable row id') from exc
            if ident in by_task.setdefault(task, {}):
                raise AnalysisError(f'{key}: duplicate {task} id {ident!r}')
            if row['gold'] not in (0, 1) or row['prediction'] not in (0, 1):
                raise AnalysisError(f'{key}: gold and prediction must be binary')
            tokens = row['gold_loss_tokens']
            nll = row['gold_nll_sum']
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                raise AnalysisError(f'{key}: gold_loss_tokens must be a positive integer')
            try:
                nll = float(nll)
            except (TypeError, ValueError) as exc:
                raise AnalysisError(f'{key}: invalid gold_nll_sum') from exc
            if not math.isfinite(nll) or nll < 0:
                raise AnalysisError(f'{key}: gold_nll_sum must be finite and nonnegative')
            by_task[task][ident] = (int(row['gold']), int(row['prediction']), nll, tokens)
        if set(by_task) != set(TASKS):
            raise AnalysisError(f'{key}: split must contain both tasks')
        data[key] = by_task
        if split == 'validation':
            if metrics_path is None:
                raise AnalysisError(f'{key}: validation requires a metrics sidecar with exclusion audit')
            entry['_validation_audit'] = _validation_audit(metrics, key)
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    missing = expected - set(indexed)
    if missing:
        raise AnalysisError(f'missing required arm/seed entries: {sorted(missing)}')
    # Base is optional, but if present it must be represented once with null seed.
    if ('Base', None) not in indexed and any(arm == 'Base' for arm, _ in indexed):
        raise AnalysisError('Base entries require one null-seed entry')
    if split == 'validation' and ('Base', None) not in indexed:
        raise AnalysisError('validation requires a Base entry with seed null')
    if split == 'validation':
        audit_values = [indexed[key]['_validation_audit'] for key in expected | {('Base', None)}]
        if any(audit != audit_values[0] for audit in audit_values[1:]):
            raise AnalysisError('validation exclusion metadata differs across the ten model results')
    wiki_keys = set(wiki)
    expected_wiki = expected | {('Base', None)}
    if split == 'validation':
        if wiki_keys != expected_wiki:
            missing_wiki = expected_wiki - wiki_keys
            extra_wiki = wiki_keys - expected_wiki
            raise AnalysisError(f'validation WikiText NLL requires Base and all nine arm/seed results; missing={sorted(missing_wiki)}, extra={sorted(extra_wiki)}')
    elif wiki_keys and wiki_keys != {('Base', None)}:
        raise AnalysisError('dev manifest may contain Base WikiText NLL only, or omit WikiText')
    return indexed, data, wiki


def _validation_audit(metrics: dict, key: tuple) -> dict:
    """Read the locked per-task validation counts/exclusion audit sidecar fields.

    Required schema: metrics['validation'][task] has raw_examples,
    examples (kept count), excluded_ids (list), and protocol_deviation (bool).
    """
    validation = metrics.get('validation')
    if not isinstance(validation, dict):
        raise AnalysisError(f'{key}: metrics sidecar needs a validation object')
    expected_raw = {'piqa': 1838, 'boolq': 3270}
    audit = {}
    for task in TASKS:
        record = validation.get(task)
        if not isinstance(record, dict):
            raise AnalysisError(f'{key}: validation.{task} audit missing')
        raw = record.get('raw_examples')
        kept = record.get('examples')
        excluded = record.get('excluded_ids')
        deviation = record.get('protocol_deviation')
        if type(raw) is not int or raw != expected_raw[task]:
            raise AnalysisError(f'{key}: validation.{task}.raw_examples must be {expected_raw[task]}')
        if type(kept) is not int or kept < 0:
            raise AnalysisError(f'{key}: validation.{task}.examples must be a nonnegative integer')
        if not isinstance(excluded, list) or len(set(map(_id_key, excluded))) != len(excluded):
            raise AnalysisError(f'{key}: validation.{task}.excluded_ids must be a unique ID list')
        if type(deviation) is not bool:
            raise AnalysisError(f'{key}: validation.{task}.protocol_deviation must be boolean')
        if kept + len(excluded) != raw:
            raise AnalysisError(f'{key}: validation.{task} kept plus excluded count must equal raw count')
        if deviation != (len(excluded) / raw > .005):
            raise AnalysisError(f'{key}: validation.{task}.protocol_deviation disagrees with the 0.5% exclusion rule')
        audit[task] = {'raw_examples': raw, 'examples': kept,
                       'excluded_ids': sorted(excluded, key=_id_key),
                       'protocol_deviation': deviation}
    return audit


def _id_key(value: Any) -> str:
    """Stable typed serialization so numeric and string IDs remain distinct."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _align(data: dict) -> dict[str, list[Any]]:
    reference_key = ('V5-C', 1)
    reference = data[reference_key]
    ids = {}
    for task in TASKS:
        ids[task] = list(reference[task])
    for key, task_data in data.items():
        for task in TASKS:
            ref_ids, got = ids[task], list(task_data[task])
            if set(ref_ids) != set(got):
                raise AnalysisError(f'{key}: {task} IDs/exclusions differ from paired rows')
            if any(reference[task][ident][0] != task_data[task][ident][0] for ident in ref_ids):
                raise AnalysisError(f'{key}: {task} gold labels do not align')
    return ids


def _row_arrays(data: dict, ids: dict) -> dict:
    arrays = {}
    for key, tasks in data.items():
        arrays[key] = {}
        for task in TASKS:
            rows = [tasks[task][ident] for ident in ids[task]]
            arrays[key][task] = {
                'correct': np.asarray([pred == gold for gold, pred, _, _ in rows], dtype=np.float64),
                'nll': np.asarray([nll for _, _, nll, _ in rows], dtype=np.float64),
                'tokens': np.asarray([tokens for _, _, _, tokens in rows], dtype=np.float64),
            }
    return arrays


def _point(arrays: dict, key: tuple, task: str) -> dict:
    d = arrays[key][task]
    return {'accuracy': float(d['correct'].mean()), 'correct': int(d['correct'].sum()),
            'n': len(d['correct']), 'gold_nll_sum': float(d['nll'].sum()),
            'gold_loss_tokens': int(d['tokens'].sum()),
            'nll': float(d['nll'].sum() / d['tokens'].sum())}


def _sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _paired_bootstrap(arrays: dict, repetitions: int, chunk_size: int = 128) -> dict:
    if repetitions <= 0 or chunk_size <= 0:
        raise ValueError('repetitions and chunk_size must be positive')
    pairs = {'primary': ('V5-C', 'V5-L'), 'S1': ('V5-C', 'V5-F'), 'S2': ('V5-F', 'V5-L')}
    samples = {name: np.empty(repetitions, dtype=np.float64) for name in pairs}
    task_samples = {name: {task: np.empty(repetitions, dtype=np.float64) for task in TASKS}
                    for name in pairs}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sizes = {task: len(arrays[('V5-C', 1)][task]['correct']) for task in TASKS}
    row_deltas = {}
    for name, (left, right) in pairs.items():
        for task in TASKS:
            row_deltas[name, task] = np.mean([
                arrays[(left, seed)][task]['correct'] - arrays[(right, seed)][task]['correct']
                for seed in SEEDS], axis=0)
    for begin in range(0, repetitions, chunk_size):
        end = min(repetitions, begin + chunk_size)
        for replicate in range(begin, end):
            # Draw in replicate-major, then PIQA/BoolQ order. This is the
            # reference RNG stream regardless of chunk_size.
            task_delta = {}
            for task in TASKS:
                ix = rng.integers(0, sizes[task], size=sizes[task])
                for name in pairs:
                    task_delta[name, task] = float(row_deltas[name, task][ix].mean())
            for name in pairs:
                for task in TASKS:
                    task_samples[name][task][replicate] = task_delta[name, task]
                samples[name][replicate] = (task_delta[name, 'piqa'] + task_delta[name, 'boolq']) / 2
    return {name: {'samples': samples[name], 'task_samples': task_samples[name]} for name in pairs}


def _ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [.025, .975])]


def _holm(pvalues: dict[str, float], alpha: float = .05) -> dict:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted, reject = {}, {}
    running = 0.0
    still_rejecting = True
    for rank, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * pvalue))
        adjusted[name] = running
        threshold = alpha / (m - rank)
        reject[name] = bool(still_rejecting and pvalue <= threshold)
        if pvalue > threshold:
            still_rejecting = False
    return {'alpha': alpha, 'adjusted_p': adjusted, 'reject': reject,
            'order': [name for name, _ in ordered]}


def classify_primary(delta, ci95, seed_deltas, task_deltas):
    """Apply the four point-estimate gates and the two strict CI boundaries."""
    if delta >= Fraction(1, 100) and ci95[0] > 0 and all(v > 0 for v in seed_deltas) and all(
            value >= Fraction(-1, 100) for value in task_deltas):
        return 'supported'
    if ci95[1] < .01:
        return 'challenged'
    return 'inconclusive'


def analyze(manifest: Any, split: str, repetitions: int = DEFAULT_REPETITIONS) -> dict:
    """Analyze one split from a manifest path or loaded manifest object."""
    if split not in ('dev', 'validation'):
        raise ValueError("split must be 'dev' or 'validation'")
    obj, root = _load_manifest(manifest)
    indexed, data, wiki = _parse_artifacts(obj, root, split)
    ids = _align(data)
    if split == 'validation':
        for task in TASKS:
            kept_ids = {_id_key(ident) for ident in ids[task]}
            audit = indexed[('V5-C', 1)]['_validation_audit'][task]
            excluded = set(map(_id_key, audit['excluded_ids']))
            if len(ids[task]) != audit['examples']:
                raise AnalysisError(f'validation {task}: reported kept count differs from paired prediction rows')
            if kept_ids & excluded:
                raise AnalysisError(f'validation {task}: excluded IDs overlap kept prediction IDs')
    arrays = _row_arrays(data, ids)
    metrics = {}
    for arm in ARMS:
        seed_results = {}
        for seed in SEEDS:
            key = (arm, seed)
            task_results = {task: _point(arrays, key, task) for task in TASKS}
            task_results['macro_accuracy'] = (task_results['piqa']['accuracy'] + task_results['boolq']['accuracy']) / 2
            task_results['macro_nll'] = (task_results['piqa']['nll'] + task_results['boolq']['nll']) / 2
            seed_results[str(seed)] = task_results
        summarized = {}
        for task in (*TASKS, 'macro_accuracy', 'macro_nll'):
            if task in TASKS:
                task_summary = {}
                for metric in ('accuracy', 'nll'):
                    vals = [seed_results[str(s)][task][metric] for s in SEEDS]
                    task_summary[metric] = {'seed_mean': float(np.mean(vals)),
                                            'seed_sample_sd': _sample_sd(vals)}
                summarized[task] = task_summary
            else:
                metric = 'accuracy' if task == 'macro_accuracy' else 'nll'
                vals = [seed_results[str(s)][task] for s in SEEDS]
                summarized[task] = {'seed_mean': float(np.mean(vals)),
                                    'seed_sample_sd': _sample_sd(vals)}
        metrics[arm] = {'seeds': seed_results, 'summary': summarized}
    per_seed_deltas = {}
    for arm_a, arm_b, name in (('V5-C', 'V5-L', 'primary'), ('V5-C', 'V5-F', 'S1'), ('V5-F', 'V5-L', 'S2')):
        ds = {}
        for seed in SEEDS:
            av = metrics[arm_a]['seeds'][str(seed)]['macro_accuracy']
            bv = metrics[arm_b]['seeds'][str(seed)]['macro_accuracy']
            ds[str(seed)] = av - bv
        per_seed_deltas[name] = ds
    primary_seed = per_seed_deltas['primary']
    delta = float(np.mean(list(primary_seed.values())))
    task_deltas = {}
    for task in TASKS:
        vals = []
        for seed in SEEDS:
            left = metrics['V5-C']['seeds'][str(seed)][task]['accuracy']
            right = metrics['V5-L']['seeds'][str(seed)][task]['accuracy']
            vals.append(left - right)
        task_deltas[task] = {'by_seed': {str(s): vals[i] for i, s in enumerate(SEEDS)},
                             'seed_mean': float(np.mean(vals))}
    boot = _paired_bootstrap(arrays, repetitions)
    primary_ci = _ci(boot['primary']['samples'])
    # Count ratios define the inclusive point-estimate gates exactly. Subtracting
    # floating-point accuracies can turn an exact -1 pp into -1.000000000000001 pp.
    exact_task = {}
    exact_seed = {}
    for task in TASKS:
        exact_task[task] = {}
        for seed in SEEDS:
            left, right = (_point(arrays, (arm, seed), task) for arm in ('V5-C', 'V5-L'))
            exact_task[task][seed] = Fraction(left['correct'], left['n']) - Fraction(right['correct'], right['n'])
    for seed in SEEDS:
        exact_seed[seed] = sum((exact_task[t][seed] for t in TASKS), Fraction()) / 2
    exact_primary = sum(exact_seed.values(), Fraction()) / 3
    exact_task_mean = {task: sum(values.values(), Fraction()) / 3 for task, values in exact_task.items()}
    decision = classify_primary(exact_primary, primary_ci, exact_seed.values(), exact_task_mean.values())
    comparisons = {}
    pvalues = {}
    for name in ('S1', 'S2'):
        vals = boot[name]['samples']
        p = min(1.0, 2 * min(float(np.mean(vals <= 0)), float(np.mean(vals >= 0))))
        pvalues[name] = p
        left, right = (('V5-C', 'V5-F') if name == 'S1' else ('V5-F', 'V5-L'))
        point = float(np.mean(list(per_seed_deltas[name].values())))
        comparisons[name] = {'contrast': f'{left} - {right}', 'seed_mean_difference': point,
                             'ci95': _ci(vals), 'two_sided_empirical_p': p,
                             'task_ci95': {t: _ci(boot[name]['task_samples'][t]) for t in TASKS},
                             'by_seed': per_seed_deltas[name]}
        comparisons[name]['task_deltas'] = {
            task: {'by_seed': {str(seed): metrics[left]['seeds'][str(seed)][task]['accuracy'] -
                                      metrics[right]['seeds'][str(seed)][task]['accuracy'] for seed in SEEDS},
                   'seed_mean': float(np.mean([
                       metrics[left]['seeds'][str(seed)][task]['accuracy'] -
                       metrics[right]['seeds'][str(seed)][task]['accuracy'] for seed in SEEDS]))}
            for task in TASKS}
    holm = _holm(pvalues)
    for name in ('S1', 'S2'):
        comparisons[name]['holm_adjusted_p'] = holm['adjusted_p'][name]
        comparisons[name]['holm_reject_at_0_05'] = holm['reject'][name]
    wiki_result = None
    base = wiki.get(('Base', None))
    if base is not None:
        arms = {}
        for arm in ARMS:
            deltas = {str(seed): wiki[(arm, seed)] - base for seed in SEEDS if (arm, seed) in wiki}
            if len(deltas) == 3:
                values = list(deltas.values())
                arms[arm] = {'by_seed': deltas, 'seed_mean': float(np.mean(values)),
                             'seed_sample_sd': _sample_sd(values)}
        wiki_result = {'base_nll': base, 'delta_nll_vs_base': arms}
    # Convert point estimates to exact rational count ratios for machine output.
    exact = {}
    for arm in ARMS:
        exact[arm] = {}
        for task in TASKS:
            exact[arm][task] = {str(seed): _fraction(Fraction(_point(arrays, (arm, seed), task)['correct'],
                                                                _point(arrays, (arm, seed), task)['n']))
                                for seed in SEEDS}
    exact_deltas = {}
    for name, (arm_a, arm_b) in {'primary': ('V5-C', 'V5-L'), 'S1': ('V5-C', 'V5-F'),
                                 'S2': ('V5-F', 'V5-L')}.items():
        per_seed = {}
        for seed in SEEDS:
            task_diff = []
            for task in TASKS:
                a = _point(arrays, (arm_a, seed), task)
                b = _point(arrays, (arm_b, seed), task)
                task_diff.append(Fraction(a['correct'], a['n']) - Fraction(b['correct'], b['n']))
            per_seed[str(seed)] = (task_diff[0] + task_diff[1]) / 2
        exact_deltas[name] = _fraction(sum(per_seed.values(), Fraction()) / len(SEEDS))
    return {
        'contract': 'E-0013@3', 'split': split,
        'evidence_status': ('dev rehearsal only; not evidence for dev' if split == 'dev' else
                            'validation analysis; exact seeds required'),
        'procedural_seed_note': 'Duplicated seed data used for rehearsal is procedural only, not evidence.',
        'bootstrap': {'method': 'paired task-stratified row bootstrap', 'repetitions': repetitions,
                      'rng': 'numpy.random.default_rng', 'seed': BOOTSTRAP_SEED,
                      'seed_resampling': False, 'shared_indices_across_arms_and_seeds': True},
        'counts_by_task': {task: len(ids[task]) for task in TASKS},
        'validation_audit': ({task: indexed[('V5-C', 1)]['_validation_audit'][task]
                              for task in TASKS} if split == 'validation' else None),
        'accuracy_fractions': exact,
        'delta_fractions': exact_deltas,
        'arms': metrics,
        'base': ({task: _point(arrays, ('Base', None), task) for task in TASKS}
                 if ('Base', None) in arrays else None),
        'primary': {'contrast': 'V5-C - V5-L', 'delta': delta, 'delta_seed_mean': delta,
                    'delta_seed_sample_sd': _sample_sd(list(primary_seed.values())),
                    'by_seed': primary_seed, 'ci95': primary_ci,
                    'task_deltas': task_deltas, 'decision': decision},
        'secondary': comparisons, 'holm': holm, 'wikitext': wiki_result,
        'procedural_exclusions': ('Validation requires identical explicit exclusion audits and complete raw counts; any mismatch in paired IDs is rejected.'
                                  if split == 'validation' else 'Rows absent from all rehearsal arms are outside the compared set; mismatched paired IDs are rejected.'),
        'interpretation': {
            'intervals': 'Bootstrap intervals reflect data uncertainty conditional on these three trained seeds; seeds are fixed and not resampled.',
            'thresholds': '+1 pp and per-task -1 pp are point-estimate gates. Supported does not guarantee a 95% interval of at least +1 pp or taskwise noninferiority; challenged is not an equivalence finding.',
            'secondary_intervals': 'Secondary confidence intervals are raw, not Holm-adjusted simultaneous intervals.',
            'S1': 'V5-C minus V5-F is a budget comparison, not an isolated fusion-location intervention.',
        },
    }


def render_markdown(result: dict) -> str:
    pp = lambda value: f'{100 * value:+.2f} pp'
    lines = [f"# V5 analysis ({result['split']})", '',
             f"**Status:** {result['evidence_status']}", '',
             result['procedural_seed_note'], '',
             f"Rows: PIQA {result['counts_by_task']['piqa']}, BoolQ {result['counts_by_task']['boolq']}.", '',
             '## Arm metrics', '',
             '| Arm | Task | Seed mean accuracy ± sample SD | Seed mean NLL ± sample SD |',
             '|---|---|---:|---:|']
    for arm in ARMS:
        for task in TASKS:
            rs = [result['arms'][arm]['seeds'][str(seed)][task] for seed in SEEDS]
            acc, nll = [r['accuracy'] for r in rs], [r['nll'] for r in rs]
            lines.append(f"| {arm} | {task} | {np.mean(acc):.4f} ± {_sample_sd(acc):.4f} | {np.mean(nll):.4f} ± {_sample_sd(nll):.4f} |")
    if result['base']:
        for task, values in result['base'].items():
            lines.append(f"| Base | {task} | {values['accuracy']:.4f} | {values['nll']:.4f} |")
    if result['validation_audit']:
        lines += ['', '## Validation exclusions', '',
                  '| Task | Raw rows | Kept | Excluded | Protocol deviation (>0.5%) |',
                  '|---|---:|---:|---:|---:|']
        for task, audit in result['validation_audit'].items():
            lines.append(f"| {task} | {audit['raw_examples']} | {audit['examples']} | {len(audit['excluded_ids'])} | {audit['protocol_deviation']} |")
    p = result['primary']
    lines += ['', '## Primary contrast', '',
              f"V5-C − V5-L: **{pp(p['delta'])}**; 95% bootstrap CI [{pp(p['ci95'][0])}, {pp(p['ci95'][1])}]; decision: **{p['decision']}**.",
              '', 'Per-seed differences: ' + ', '.join(f"seed {s}: {pp(v)}" for s, v in p['by_seed'].items()) + '.',
              '', 'Per-task seed-mean differences: ' + ', '.join(f"{t}: {pp(v['seed_mean'])}" for t, v in p['task_deltas'].items()) + '.',
              '', '## Secondary contrasts', '',
              '| Contrast | Seed mean | 95% CI | Empirical p | Holm adjusted p | Reject at 0.05 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, comp in result['secondary'].items():
        lines.append(f"| {name} {comp['contrast']} | {pp(comp['seed_mean_difference'])} | [{pp(comp['ci95'][0])}, {pp(comp['ci95'][1])}] | {comp['two_sided_empirical_p']:.4g} | {comp['holm_adjusted_p']:.4g} | {comp['holm_reject_at_0_05']} |")
    wiki = result['wikitext']
    if wiki:
        lines += ['', '## WikiText NLL change vs Base', '', f"Base NLL: {wiki['base_nll']:.6f}.", '']
        for arm, values in wiki['delta_nll_vs_base'].items():
            lines.append(f"- {arm}: {values['seed_mean']:+.4f} ± {values['seed_sample_sd']:.4f}")
    lines += ['', result['interpretation']['intervals'],
              result['interpretation']['thresholds'],
              result['interpretation']['secondary_intervals'],
              result['interpretation']['S1'],
              '', 'Bootstrap resamples rows within PIQA and BoolQ, shares indices across all arms and seeds, and keeps the three fixed seeds fixed.',
              'Accuracy and differences in the JSON are represented as fractions where exact count ratios are available.', '']
    return '\n'.join(lines)


def write_outputs(result: dict, output: Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'analysis.md').write_text(render_markdown(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'validation'))
    parser.add_argument('--output', required=True, type=Path, help='output directory')
    args = parser.parse_args()
    result = analyze(args.manifest, args.split)
    write_outputs(result, args.output)


if __name__ == '__main__':
    main()
