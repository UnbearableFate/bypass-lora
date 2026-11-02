"""Exact point-estimate rules 1-9, E-0025@1 with pre-result D-0044.

Returns rule outcomes, never changes a hypothesis status or starts work.
"""
from fractions import Fraction

TASKS = ('piqa', 'boolq')
PP = Fraction(1, 100)


def number(x):
    return {'numerator': x.numerator, 'denominator': x.denominator, 'pp': float(100*x)}


def analyze(counts, checks, *, require_interventions=True):
    required = [f'SEG-{i:02}' for i in range(5)]+['RB1-07']
    optional = [f'SEG-{i:02}' for i in (5, 6, 7)]
    if any(x not in counts for x in required):
        raise ValueError('rules require Base_seg, all four r32 arms and full-context LoRA')
    present = set(optional).intersection(counts)
    if present and present != set(optional):
        raise ValueError('r128 comparison requires exactly all three arms')
    score = {}
    for label, tasks in counts.items():
        score[label] = {}
        for task in TASKS:
            correct, total = tasks[task]
            if type(correct) is not int or type(total) is not int or not 0 <= correct <= total or total <= 0:
                raise ValueError('invalid exact correctness counts')
            if total != counts['SEG-00'][task][1]:
                raise ValueError('paired task row counts differ')
            score[label][task] = Fraction(correct, total)
    required_checks = ('off_matches_base', 'initial_matches_base', 'no_channel_identity', 'reload_identical')
    implementation_ok = all(checks.get(k) is True for k in required_checks)
    informative = {t: score['RB1-07'][t]-score['SEG-04'][t] >= 3*PP for t in TASKS}
    primary = 'piqa' if informative['piqa'] else 'boolq' if informative['boolq'] else None
    r128 = bool(present)
    per_task = {}
    for task in TASKS:
        gain32 = score['SEG-02'][task]-max(score[a][task] for a in ('SEG-01', 'SEG-03'))
        swap32 = None if 'SEG-02/iv_swap' not in score else score.get('SEG-02/normal', score['SEG-02'])[task]-score['SEG-02/iv_swap'][task]
        release = 0 < gain32 < 3*PP
        outcome, width = 'descriptive', 32
        if not informative[task]:
            outcome = 'uninformative'
        elif gain32 <= 0:
            outcome = 'challenges'
        elif gain32 >= 3*PP and swap32 is not None and swap32 >= gain32/2:
            outcome = 'supports'
        gain128, swap128 = None, None
        if release and r128:
            gain128 = score['SEG-05'][task]-max(score[a][task] for a in ('SEG-06', 'SEG-07'))
            swap128 = None if 'SEG-05/iv_swap' not in score else score.get('SEG-05/normal', score['SEG-05'])[task]-score['SEG-05/iv_swap'][task]
            if informative[task]:
                if gain128 >= 3*PP and swap128 is not None and swap128 >= gain128/2:
                    outcome, width = 'supports', 128
                elif gain32 < PP and gain128 < PP:
                    outcome = 'challenges'
                else:
                    outcome = 'descriptive_inconclusive_D0044c'
        per_task[task] = {'informative': informative[task], 'gain32': number(gain32),
            'swap_drop32': None if swap32 is None else number(swap32),
            'gain128': None if gain128 is None else number(gain128),
            'swap_drop128': None if swap128 is None else number(swap128),
            'outcome': outcome, 'support_width': width if outcome == 'supports' else None,
            'release_range': release}
    release = primary is not None and per_task[primary]['release_range']
    if r128 and not release:
        raise ValueError('r128 supplied without the primary-task release gate')
    if require_interventions:
        for arm in ('SEG-02', *(['SEG-05'] if r128 else [])):
            if any(arm+'/'+kind not in score for kind in ('iv_off', 'iv_swap', 'iv_reset')):
                raise ValueError('complete intervention grid required')
    learned = None if primary is None else {a: score[a][primary] >= score['SEG-00'][primary]+PP
        for a in (*required[1:5], *(optional if r128 else ()))}
    controls_unlearned = primary is not None and not learned['SEG-01'] and not learned['SEG-04']
    secondary = None if primary is None else next(t for t in TASKS if t != primary)
    direction = 'descriptive' if primary is None else per_task[primary]['outcome']
    if primary and release and not r128:
        direction = 'pending_r128'
    secondary_only = (secondary is not None and per_task[secondary]['outcome'] == 'supports' and
                      direction not in ('supports', 'pending_r128'))
    if secondary_only:
        direction = 'inconclusive_secondary_only'
    if not implementation_ok:
        direction = 'invalid_implementation'
    edge = direction if direction in ('supports', 'challenges') and not controls_unlearned else None
    rules = {
        '1': {'level': 'diagnostic', 'passed': implementation_ok, 'checks': checks,
              'action': 'interpret' if implementation_ok else 'repair_no_conclusion'},
        '2': {'level': 'screening', 'informative': informative,
              'full_lora_minus_segment_lora': {t: number(score['RB1-07'][t]-score['SEG-04'][t]) for t in TASKS}},
        '3': {'level': 'screening', 'primary': primary,
              'action': 'stop_no_usable_information_removed' if primary is None else 'compare'},
        '4': {'level': 'diagnostic' if controls_unlearned else 'screening', 'learned': learned,
              'no_history_controls_unlearned': controls_unlearned},
        '5': {'level': 'diagnostic' if controls_unlearned else 'screening',
              'primary_r32': None if primary is None else per_task[primary]},
        '6': {'level': 'diagnostic' if controls_unlearned else 'screening',
              'release_once': bool(release and implementation_ok), 'completed': r128, 'decision_pin': 'D-0044',
              'no_more_width_tap_loop_after_r128': r128},
        '7': {'level': 'diagnostic' if controls_unlearned else 'screening',
              'challenge_r32': bool(primary and per_task[primary]['gain32']['numerator'] <= 0)},
        '8': {'level': 'diagnostic' if controls_unlearned else 'screening', 'secondary': secondary,
              'task_result': None if secondary is None else per_task[secondary],
              'secondary_only': secondary_only, 'two_new_seed_replications_needed': secondary_only},
        '9': {'level': 'screening', 'macro_nll': 'descriptive_only',
              'hypothesis_status_changed': False, 'intervals': 'descriptive conditional on trained seed0; no seed/selection uncertainty or simultaneous guarantee'}
    }
    return {'rules': rules, 'primary': primary, 'per_task': per_task,
            'direction': direction, 'eligible_edge': edge,
            'release_r128': bool(release and implementation_ok and not r128), 'r128_completed': r128}


def gain_intervals(rows, repeats=10000, seed=20261101, chunk_size=128):
    """Bootstrap the max-control estimand, using the same draws as RB1 pairs.

    Max is recomputed within each replicate. This is a descriptive interval;
    it does not include seed variation or selecting the task/width on this dev.
    """
    import numpy as np
    arrays = {a: {t: np.asarray(rows[a][t], dtype=float) for t in TASKS} for a in rows}
    sizes = {t: len(arrays['SEG-00'][t]) for t in TASKS}
    for a, tasks in arrays.items():
        for t in TASKS:
            if tasks[t].shape != (sizes[t],) or sizes[t] == 0 or not np.isin(tasks[t], (0, 1)).all():
                raise ValueError('gain bootstrap requires aligned nonempty binary task rows')
    widths = {32: ('SEG-02', 'SEG-01', 'SEG-03')}
    if 'SEG-05' in arrays:
        widths[128] = ('SEG-05', 'SEG-06', 'SEG-07')
    samples = {str(w): {t: np.empty(repeats) for t in (*TASKS, 'macro')} for w in widths}
    rng = np.random.default_rng(seed)
    for start in range(0, repeats, chunk_size):
        end = min(start+chunk_size, repeats)
        draws = {t: rng.integers(0, sizes[t], (end-start, sizes[t])) for t in TASKS}
        for width, (keep, control1, control2) in widths.items():
            gains = {}
            for t in TASKS:
                scores = {a: arrays[a][t][draws[t]].mean(1) for a in (keep, control1, control2)}
                gains[t] = scores[keep]-np.maximum(scores[control1], scores[control2])
                samples[str(width)][t][start:end] = gains[t]
            # Descriptive macro gain uses the max of macro control scores,
            # which can differ from the mean of taskwise maxima.
            macro = {a: sum(arrays[a][t][draws[t]].mean(1) for t in TASKS)/2
                     for a in (keep, control1, control2)}
            samples[str(width)]['macro'][start:end] = macro[keep]-np.maximum(macro[control1], macro[control2])
    return {w: {t: {'ci95_pp': [float(v*100) for v in np.quantile(x, [.025, .975])]}
                for t, x in tasks.items()} for w, tasks in samples.items()}
