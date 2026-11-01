"""Pure numerical analysis for the E-0021 RB1 screening rules.

The module deliberately does not load artifacts or assign hypothesis status. The
caller supplies exact correctness counts, task NLL aggregates, and (optionally)
paired row outcomes for descriptive bootstrap intervals.

``counts`` has shape ``{label: {task: (correct, total)}}``. Required labels are
``Base`` and ``RB1-01`` through ``RB1-09``; optional extra labels such as
``RB1-02/normal`` and ``RB1-02/iv_swap`` represent intervention conditions.
``nll`` has the same
shape, with each task value either ``(nll_sum, loss_tokens)`` or a scalar mean
NLL. Counts determine all accuracy decisions through :class:`Fraction`.

Bootstrap rows have shape ``{label: {task: sequence_of_0_or_1}}``. The same
task-stratified indices are used for every label in each replicate.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np


ARM_IDS = tuple(f'RB1-{i:02d}' for i in range(1, 10))
TASKS = ('piqa', 'boolq')
PARAMS = {
    'RB1-08': 1_318_912,  # C_dual
    'RB1-09': 925_696,    # C_up
}
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20261101
ONE_PP = Fraction(1, 100)
HALF_PP = Fraction(1, 200)


class RulesError(ValueError):
    """Input metrics cannot support the requested paired analysis."""


def counts_from_rows(rows: Sequence[Mapping]) -> dict[str, dict[str, tuple[int, int]]]:
    """Convert prediction rows to exact ``(correct, total)`` counts by task."""
    result: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        task = row.get('task')
        if task not in TASKS:
            raise RulesError(f'unexpected or missing task {task!r}')
        if 'gold' not in row or 'prediction' not in row:
            raise RulesError('prediction rows require gold and prediction')
        bucket = result.setdefault(task, [0, 0])
        bucket[0] += int(row['gold'] == row['prediction'])
        bucket[1] += 1
    if set(result) != set(TASKS):
        raise RulesError(f'rows must include both tasks {TASKS}')
    return {task: (result[task][0], result[task][1]) for task in TASKS}


def _fractional_counts(counts: Mapping, label: str) -> dict[str, tuple[int, int]]:
    try:
        per_task = counts[label]
    except KeyError as exc:
        raise RulesError(f'missing counts for {label}') from exc
    result = {}
    for task in TASKS:
        try:
            correct, total = per_task[task]
        except (KeyError, TypeError, ValueError) as exc:
            raise RulesError(f'{label}: {task} count must be (correct, total)') from exc
        if (not isinstance(correct, (int, np.integer)) or
                not isinstance(total, (int, np.integer)) or total <= 0 or
                correct < 0 or correct > total):
            raise RulesError(f'{label}: invalid {task} count {(correct, total)!r}')
        result[task] = (int(correct), int(total))
    return result


def _task_accuracy(counts: Mapping[str, tuple[int, int]], task: str) -> Fraction:
    correct, total = counts[task]
    return Fraction(correct, total)


def _macro(counts: Mapping[str, tuple[int, int]]) -> Fraction:
    return sum((_task_accuracy(counts, task) for task in TASKS), Fraction()) / len(TASKS)


def _task_nll(nll: Mapping | None, label: str, task: str) -> float | None:
    if nll is None or label not in nll or task not in nll[label]:
        return None
    value = nll[label][task]
    if isinstance(value, (tuple, list)):
        total, tokens = value
        if tokens <= 0:
            raise RulesError(f'{label}: {task} NLL loss-token count must be positive')
        score = float(total) / int(tokens)
    else:
        score = float(value)
    if not np.isfinite(score):
        raise RulesError(f'{label}: {task} NLL must be finite')
    return score


def _macro_nll(nll: Mapping | None, label: str) -> float | None:
    values = [_task_nll(nll, label, task) for task in TASKS]
    return None if any(value is None for value in values) else sum(values) / len(values)


def _json_fraction(value: Fraction) -> dict[str, int | float]:
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'value': float(value), 'percentage_points': float(value * 100)}


def paired_interval(comparisons: Mapping[str, Mapping], left: str, right: str,
                    metric: str = 'macro_accuracy') -> Mapping:
    """Return a requested paired comparison, reversing sign and CI if needed."""
    direct = comparisons.get(f'{left} - {right}')
    if direct is not None:
        return direct[metric]
    reverse = comparisons.get(f'{right} - {left}')
    if reverse is None:
        raise RulesError(f'no paired comparison for {left} and {right}')
    result = reverse[metric]
    low, high = result['ci95']
    return {'difference': -result['difference'], 'ci95': [-high, -low]}


def paired_bootstrap(rows: Mapping[str, Mapping[str, Sequence[int]]], *,
                     repeats: int = BOOTSTRAP_REPEATS,
                     seed: int = BOOTSTRAP_SEED,
                     chunk_size: int = 128) -> dict:
    """Bootstrap every supplied label using shared task-stratified row draws."""
    if repeats <= 0 or chunk_size <= 0:
        raise RulesError('bootstrap repeats and chunk_size must be positive')
    labels = list(rows)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    sizes = None
    for label in labels:
        if label not in rows:
            raise RulesError(f'bootstrap rows missing {label}')
        arrays[label] = {}
        current = {}
        for task in TASKS:
            try:
                values = np.asarray(rows[label][task], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise RulesError(f'{label}: bootstrap rows missing valid {task} outcomes') from exc
            if values.ndim != 1 or len(values) == 0 or not np.isin(values, (0, 1)).all():
                raise RulesError(f'{label}: {task} bootstrap outcomes must be a nonempty 0/1 vector')
            arrays[label][task] = values
            current[task] = len(values)
        if sizes is None:
            sizes = current
        elif current != sizes:
            raise RulesError(f'{label}: bootstrap task row sizes differ')

    pairs = [(left, right) for i, left in enumerate(labels) for right in labels[i + 1:]]
    samples = {pair: {metric: np.empty(repeats, dtype=np.float64)
                      for metric in ('macro_accuracy', 'piqa_accuracy', 'boolq_accuracy')}
               for pair in pairs}
    rng = np.random.default_rng(seed)
    start = 0
    while start < repeats:
        end = min(repeats, start + chunk_size)
        n = end - start
        draws = {task: rng.integers(0, sizes[task], size=(n, sizes[task])) for task in TASKS}
        scores = {}
        for label in labels:
            task_scores = {task: arrays[label][task][draws[task]].mean(axis=1) for task in TASKS}
            scores[label] = {
                'piqa_accuracy': task_scores['piqa'],
                'boolq_accuracy': task_scores['boolq'],
                'macro_accuracy': (task_scores['piqa'] + task_scores['boolq']) / 2,
            }
        for left, right in pairs:
            for metric in samples[(left, right)]:
                samples[(left, right)][metric][start:end] = scores[left][metric] - scores[right][metric]
        start = end
    output = {}
    for left, right in pairs:
        output[f'{left} - {right}'] = {
            metric: {'ci95': [float(x) for x in np.quantile(values, [.025, .975])],
                     'difference': float(
                         (arrays[left]['piqa'].mean() - arrays[right]['piqa'].mean() +
                          arrays[left]['boolq'].mean() - arrays[right]['boolq'].mean()) / 2
                         if metric == 'macro_accuracy' else
                         arrays[left][metric.removesuffix('_accuracy')].mean() -
                         arrays[right][metric.removesuffix('_accuracy')].mean())}
            for metric, values in samples[(left, right)].items()
        }
    return output


def analyze_counts(counts: Mapping[str, Mapping[str, tuple[int, int]]],
                   nll: Mapping | None = None, *,
                   bootstrap_rows: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
                   bootstrap_repeats: int = BOOTSTRAP_REPEATS,
                   bootstrap_seed: int = BOOTSTRAP_SEED,
                   chunk_size: int = 128,
                   gradient_started: bool | None = None) -> dict:
    """Apply E-0021 rules to exact counts and optionally compute paired intervals.

    Missing intervention outcomes leave GH unassessed (``None``). They never
    turn a progression gate into a hypothesis direction. ``gradient_started``
    is included as context for Rule 9, not used to alter its trigger.
    """
    labels = ('Base', *ARM_IDS)
    count_data = {label: _fractional_counts(counts, label) for label in labels}
    valid_modes = {'normal', 'iv_off', 'iv_window1', 'iv_window8', 'iv_strict', 'iv_swap', 'iv_permute'}
    for label in sorted(set(counts) - set(labels)):
        try:
            arm, mode = label.split('/', 1)
        except (AttributeError, ValueError) as exc:
            raise RulesError(f'invalid extra count label {label!r}') from exc
        if arm not in ARM_IDS or mode not in valid_modes or (arm == 'RB1-01' and mode not in ('normal', 'iv_swap')):
            raise RulesError(f'invalid intervention count label {label!r}')
        count_data[label] = _fractional_counts(counts, label)

    score = {label: _macro(data) for label, data in count_data.items()}
    base = score['Base']
    # G0 is a progression gate and is checked before further branch decisions.
    g0 = score['RB1-07'] >= base + ONE_PP
    g0_margin = score['RB1-07'] - base
    rb2_effective = score['RB1-02'] >= base + ONE_PP
    gr_margin = score['RB1-02'] - score['RB1-07']
    gr = rb2_effective and gr_margin >= -ONE_PP

    history_margin = score['RB1-02'] - score['RB1-01']
    history_reference = 'RB1-02/normal' if 'RB1-02/normal' in score else 'RB1-02'
    history_drops = {
        mode: score[history_reference] - score[f'RB1-02/{mode}']
        for mode in ('iv_swap', 'iv_permute') if f'RB1-02/{mode}' in score
    }
    gh_assessed = all(mode in history_drops for mode in ('iv_swap', 'iv_permute'))
    gh = (history_margin >= ONE_PP and all(v >= ONE_PP for v in history_drops.values())) if gh_assessed else None

    # Select S2 on a strict sub-0.5 pp tie; otherwise choose the higher score.
    s2_s4_delta = score['RB1-03'] - score['RB1-04']
    s_star = 'RB1-03' if abs(s2_s4_delta) < HALF_PP or s2_s4_delta > 0 else 'RB1-04'
    loop_margin = score[s_star] - score['RB1-02']
    query_margin = score['RB1-04'] - score['RB1-05']
    gl_point = loop_margin >= ONE_PP
    gl = gl_point and (s_star != 'RB1-04' or query_margin >= ONE_PP)

    concat_ids = ('RB1-08', 'RB1-09')
    effective_concat = [arm for arm in concat_ids if score[arm] >= base + ONE_PP]
    concat_best = max((score[arm] for arm in concat_ids))
    near_best = [arm for arm in effective_concat if concat_best - score[arm] < ONE_PP]
    c_star = (min(near_best, key=lambda arm: (PARAMS[arm], -score[arm], _macro_nll(nll, arm)))
              if near_best else None)
    gc_margin = concat_best - score['RB1-02']
    gc_raw = gc_margin >= ONE_PP
    # A GC progression/release needs a selected, effective concat arm. The
    # raw contrast remains available for Rule 11 even when C*_RB is absent.
    gc = gc_raw and c_star is not None

    training_within_base = all(
        base - ONE_PP <= score[arm] < base + ONE_PP for arm in ARM_IDS)
    rb1_t0_effective = score['RB1-01'] >= base + ONE_PP
    rule9_gate_condition = g0 and not gr and not gc
    rule9 = rule9_gate_condition
    rule9_trigger = {
        'triggered': rule9,
        'gate_condition_met': rule9_gate_condition,
        't0_effective': rb1_t0_effective if rule9 else None,
        'gradient_started': gradient_started if rule9 else None,
        'next_check': (
            'inspect_T0_positive_control_and_gradient_start'
            if rule9 and (not rb1_t0_effective or gradient_started is False)
            else 'await_implementation_and_gradient_check'
            if rule9 and gradient_started is None
            else 'at_most_one_probe_if_checks_pass' if rule9 else None
        ),
        'probe_budget': 'at most one capacity or feature probe' if rule9 else None,
    }

    pilot = bool(g0 and not gl and gc and c_star)
    if not g0:
        progression = 'stop_at_implementation_and_learnability_check'
    elif pilot:
        progression = 'C_S1_S4_pilot_then_draft_RB5_and_review_conclusion'
    elif gr or gc:
        progression = 'draft_RB5_and_review_conclusion'
    elif rule9:
        progression = 'check_T0_and_gradient_then_at_most_one_probe'
    else:
        progression = 'no_additional_gate_triggered'

    # Rule 11 directions are independent of progression and stay screening-only.
    task_main_deltas = {task: _task_accuracy(count_data['RB1-02'], task) -
                        _task_accuracy(count_data['RB1-07'], task) for task in TASKS}
    main_support = (score['RB1-02'] >= base + ONE_PP and
                    gr_margin >= ONE_PP and all(delta >= -ONE_PP for delta in task_main_deltas.values()))
    main_challenge = (score['RB1-02'] < base + ONE_PP or gr_margin <= 0)
    gh_direction = _direction(gh, history_margin)
    gl_direction = _direction(gl, loop_margin)
    gc_direction = _direction(gc_raw, gc_margin)
    directions = {
        'main_utility': _direction(main_support, score['RB1-02'] - score['RB1-07'], main_challenge),
        'history': gh_direction,
        'loop': gl_direction,
        'concat': gc_direction,
    }

    rb5_gate_condition = g0 and (gr or gc)
    rb5_released = rb5_gate_condition
    rb5_matrix = None
    if rb5_released:
        rb5_matrix = ['RB5-01', 'RB5-05', 'RB5-08']
        if gl:
            rb5_matrix.append('RB5-02')
        if gc:
            rb5_matrix.extend(['RB5-03', 'RB5-06', 'RB5-07', 'RB5-09'])
            if gl:
                rb5_matrix.append('RB5-04')
    rb5_matrix_status = 'pending_C_S1_S4_pilot' if pilot else 'final'
    pilot_s4_alternative = ('all nine RB5 groups at fixed S4'
                            if pilot else None)
    pilot_s1_alternative = ('retain provisional S1 matrix'
                            if pilot else None)
    rules = {
        '1_G0_positive_control': {'passed': g0, 'margin': _json_fraction(g0_margin),
                                  'threshold': 'RB1-07 >= Base + 1 pp'},
        '2_effective_threshold': {arm: score[arm] >= base + ONE_PP for arm in ARM_IDS},
        '3_GR_residual_single_read': {'passed': gr, 'RB1-02_vs_Base': _json_fraction(score['RB1-02'] - base),
                                      'RB1-02_vs_RB1-07': _json_fraction(gr_margin)},
        '4_GH_side_history': {'passed': gh, 'assessed': gh_assessed,
                              'normal_reference': history_reference,
                              'RB1-02_vs_RB1-01': _json_fraction(history_margin),
                              'drop_vs_normal': {m: _json_fraction(v) for m, v in history_drops.items()}},
        '5_GL_loop': {'passed': gl, 'selected': s_star,
                      'S2_minus_S4': _json_fraction(s2_s4_delta),
                      'selected_minus_RB1-02': _json_fraction(loop_margin),
                      'S4_minus_fixed_query': _json_fraction(query_margin),
                      'fixed_query_gate_required': s_star == 'RB1-04',
                      'secondary_descriptive_contrasts': {
                          'RB1-06_minus_RB1-02': _json_fraction(score['RB1-06'] - score['RB1-02']),
                          'RB1-04_minus_RB1-06': _json_fraction(score['RB1-04'] - score['RB1-06']),
                      }},
        '6_GC_concat': {'passed': gc_raw, 'best_concat_minus_RB1-02': _json_fraction(gc_margin),
                        'raw_gate_met': gc_raw, 'resolved_gate_met': gc,
                        'effective_concat_arms': effective_concat, 'C_star_RB': c_star,
                        'near_best_effective': near_best},
        '7_RB5_release': {'passed': bool(rb5_released),
                          'gate_condition_met': bool(rb5_gate_condition),
                          'matrix': rb5_matrix,
                          'matrix_status': rb5_matrix_status,
                          'if_pilot_C_S4_minus_C_S1_ge_1pp': pilot_s4_alternative,
                          'if_pilot_C_S4_minus_C_S1_lt_1pp': pilot_s1_alternative,
                          'loop_depth': 1 if not gl else (2 if s_star == 'RB1-03' else 4)},
        '8_C_S1_S4_pilot': {'raw_gate_condition_met': bool(g0 and not gl and gc_raw),
                            'gate_condition_met': bool(g0 and not gl and gc and c_star),
                            'triggered': pilot,
                            'blocked_without_C_star': bool(g0 and not gl and gc_raw and c_star is None),
                            'pilot_result': None, 'condition': 'C*_RB+KernelMatrix C-S4 minus C-S1 >= 1 pp'},
        '9_no_residual_or_concat_signal': rule9_trigger,
        '10_all_arms_within_Base_plus_minus_1pp': {'passed': training_within_base,
                                                   'interval': 'Base - 1 pp <= arm < Base + 1 pp (D-0032)',
                                                   'outcome': 'condition_met_no_gate_override'
                                                   if training_within_base else None},
        '11_screening_directions': directions,
    }
    metrics = {label: {task + '_accuracy': _json_fraction(_task_accuracy(data, task))
                       for task in TASKS} | {'macro_accuracy': _json_fraction(_macro(data)),
                                             'macro_nll': _macro_nll(nll, label)}
               for label, data in count_data.items()}
    report = {
        'experiment': 'E-0021', 'metrics': metrics, 'rules': rules,
        'progression': progression,
        'raw_gates': {'G0': g0, 'GR': gr, 'GH': gh, 'GL': gl, 'GC': gc_raw},
        'resolved_gates': {'G0': g0, 'GR': gr, 'GL': gl, 'GC': gc},
        'resolved_actions': {'progression': progression, 'RB5_matrix': rb5_matrix,
                             'RB5_matrix_status': rb5_matrix_status,
                             'if_pilot_C_S4_minus_C_S1_ge_1pp': pilot_s4_alternative,
                             'if_pilot_C_S4_minus_C_S1_lt_1pp': pilot_s1_alternative,
                             'C_S1_S4_pilot': pilot},
        'selection': {'S_F_star': s_star, 'C_star_RB': c_star,
                      'effective_concat_arms': effective_concat,
                      'near_best_effective_concat_arms': near_best},
        'screening_directions': directions,
        'notes': ['Single-seed screening; intervals are descriptive, not confirmatory.',
                  'D-0032: Rule 10 uses Base - 1 pp <= arm < Base + 1 pp and never overrides G0/GR/GC/GL.',
                  'Rule 11 reports screening directions only and does not change hypothesis status.',
                  'No same-depth T0 comparison is made; E-0021 only tests T0 at S1.'],
    }
    if bootstrap_rows is not None:
        bootstrap_labels = list(count_data)
        for label in bootstrap_labels:
            if label not in bootstrap_rows:
                raise RulesError(f'bootstrap rows missing {label}')
            for task in TASKS:
                try:
                    observed = np.asarray(bootstrap_rows[label][task], dtype=np.float64)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RulesError(f'{label}: bootstrap rows missing valid {task} outcomes') from exc
                correct, total = count_data[label][task]
                if len(observed) != total or int(observed.sum()) != correct:
                    raise RulesError(f'{label}: {task} bootstrap row count/sum differs from supplied counts')
        report['bootstrap'] = {
            'method': 'paired task-stratified row resampling',
            'repeats': bootstrap_repeats, 'seed': bootstrap_seed,
            'interval': 'percentile 95%', 'chunk_size': chunk_size,
            'difference_sign': 'left condition minus right condition',
            'comparisons': paired_bootstrap(bootstrap_rows,
                                            repeats=bootstrap_repeats,
                                            seed=bootstrap_seed,
                                            chunk_size=chunk_size),
        }
    return report


def _direction(support: bool | None, primary_delta: Fraction,
               challenge: bool | None = None) -> str:
    if support is True:
        return 'supports'
    if challenge is True or (challenge is None and primary_delta <= 0):
        return 'challenges'
    return 'descriptive'

