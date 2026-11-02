"""Pure E-0030 A2 BoolQ statistics and preregistered rules.

No artifacts are read here. ``counts`` maps each seed (3, 4, 5) to exact
``(correct, total)`` pairs for SEG-04..07 and SEG-05/iv_swap. ``checks`` maps
each seed to the diagnostic booleans and path-check observations documented
in :data:`IMPLEMENTATION_CHECKS` and :data:`PATH_CHECKS`. ``base`` is the
SEG-00 pair. ``intervals`` maps each seed and arm label to one binary outcome
per paired A2 BoolQ row; its row order must be identical across every entry.
The function returns exact Fraction-backed point estimates and the locked
shared-row bootstrap intervals.
"""
from fractions import Fraction
import math

import numpy as np

SEEDS = (3, 4, 5)
ARMS = ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07')
LABELS = (*ARMS, 'SEG-05/iv_swap')
IMPLEMENTATION_CHECKS = (
    'off_matches_base_dev',
    'initial_matches_base_dev',
    'no_channel_identity',
    'reload_identical',
    'initial_parameters_identical',
    'train_ids_disjoint_a2',
    'a2_ids_and_hashes_match',
)
PATH_CHECKS = ('path_dev_same_hardware', 'path_dev_mismatch_count', 'a2_iv_off_matches_base')
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20261101
MAX_DIFFERENT_HARDWARE_MISMATCHES = 10


def number(value):
    """Serialize an exact score/difference (unit accuracy) as a fraction and pp."""
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'pp': float(100 * value)}


def _fraction(pair, label):
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
            type(pair[0]) is not int or type(pair[1]) is not int or
            pair[1] <= 0 or not 0 <= pair[0] <= pair[1]):
        raise ValueError(f'invalid exact BoolQ counts for {label}')
    return Fraction(pair[0], pair[1])


def _seed_quantities(seed_counts):
    if set(seed_counts) != set(LABELS):
        raise ValueError('each seed needs SEG-04..07 and SEG-05/iv_swap')
    totals = {pair[1] for pair in seed_counts.values() if isinstance(pair, (tuple, list)) and len(pair) == 2}
    scores = {label: _fraction(pair, f'{label}') for label, pair in seed_counts.items()}
    if len(totals) != 1:
        raise ValueError('paired BoolQ row counts differ within a seed')
    gain = scores['SEG-05'] - max(scores['SEG-06'], scores['SEG-07'])
    lora = scores['SEG-05'] - scores['SEG-04']
    swap = scores['SEG-05'] - scores['SEG-05/iv_swap']
    return scores, gain, lora, swap


def _validate_intervals(intervals, totals):
    if set(intervals) != set(SEEDS):
        raise ValueError('bootstrap outcomes are required for seeds 3, 4 and 5')
    binary = {}
    row_count = None
    for seed in SEEDS:
        if set(intervals[seed]) != set(LABELS):
            raise ValueError(f'seed {seed} bootstrap outcomes need SEG-04..07 and iv_swap')
        binary[seed] = {}
        for label in LABELS:
            values = np.asarray(intervals[seed][label])
            if values.ndim != 1 or values.size != totals[seed]:
                raise ValueError('bootstrap vectors must be one-dimensional and match count totals')
            if values.dtype.kind not in 'biu' or np.any((values != 0) & (values != 1)):
                raise ValueError('bootstrap outcomes must contain only binary integer values')
            if row_count is None:
                row_count = values.size
            elif values.size != row_count:
                raise ValueError('all arms and seeds must share the same paired row count')
            binary[seed][label] = values.astype(np.int64, copy=False)
    return binary, row_count


def _bootstrap(binary, row_count):
    """Paired BoolQ bootstrap with one shared draw matrix and per-draw max control."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    g = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    l = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    # Chunking bounds memory while consuming the same Generator stream as a
    # single (repeats, rows) integer draw.
    cursor = 0
    while cursor < BOOTSTRAP_REPEATS:
        end = min(cursor + 128, BOOTSTRAP_REPEATS)
        indices = rng.integers(0, row_count, size=(end - cursor, row_count))
        seed_gain_counts = np.zeros(end - cursor, dtype=np.int64)
        seed_lora_counts = np.zeros(end - cursor, dtype=np.int64)
        for seed in SEEDS:
            correct = {arm: binary[seed][arm][indices].sum(axis=1) for arm in ARMS}
            seed_gain_counts += correct['SEG-05'] - np.maximum(correct['SEG-06'], correct['SEG-07'])
            seed_lora_counts += correct['SEG-05'] - correct['SEG-04']
        denominator = len(SEEDS) * row_count
        g[cursor:end] = seed_gain_counts / denominator
        l[cursor:end] = seed_lora_counts / denominator
        cursor = end
    return {
        'G': [float(x) for x in np.quantile(g, [0.025, 0.975])],
        'L': [float(x) for x in np.quantile(l, [0.025, 0.975])],
    }


def _check_bool(checks, seed, name):
    return checks[seed].get(name) is True


def analyze(counts, checks, base, intervals):
    """Apply E-0030 rules to exact BoolQ counts and paired row outcomes.

    Diagnostic input keys per seed are the seven :data:`IMPLEMENTATION_CHECKS`
    plus the three :data:`PATH_CHECKS`. The path mismatch count is an integer:
    same hardware requires zero mismatches; different hardware permits at most
    ten. Caller supplies the iv_off/Base identity result as a boolean.
    """
    if set(counts) != set(SEEDS) or set(checks) != set(SEEDS):
        raise ValueError('E-0030 requires counts and checks for seeds 3, 4 and 5')
    quantities = {}
    totals = {}
    for seed in SEEDS:
        scores, gain, lora, swap = _seed_quantities(counts[seed])
        quantities[seed] = {'scores': scores, 'gain': gain, 'lora': lora, 'swap': swap}
        totals[seed] = next(iter(counts[seed].values()))[1]
    base_score = _fraction(base, 'SEG-00 Base_seg')
    if any(totals[s] != base[1] for s in SEEDS):
        raise ValueError('Base_seg row count differs from arm rows')
    binary, row_count = _validate_intervals(intervals, totals)
    if any(row_count != totals[s] for s in SEEDS):
        raise ValueError('bootstrap row count differs from point-count rows')
    for seed in SEEDS:
        for label in LABELS:
            if int(binary[seed][label].sum()) != counts[seed][label][0]:
                raise ValueError(f'bootstrap outcomes disagree with exact correct count for seed {seed} {label}')

    implementation = {}
    path = {}
    for seed in SEEDS:
        c = checks[seed]
        mismatch_count = c.get('path_dev_mismatch_count')
        if type(mismatch_count) is not int or mismatch_count < 0:
            raise ValueError(f'seed {seed} path_dev_mismatch_count must be a nonnegative integer')
        if type(c.get('path_dev_same_hardware')) is not bool:
            raise ValueError(f'seed {seed} path_dev_same_hardware must be boolean')
        same_hardware = c.get('path_dev_same_hardware') is True
        dev_ok = mismatch_count == 0 if same_hardware else mismatch_count <= MAX_DIFFERENT_HARDWARE_MISMATCHES
        implementation[seed] = {name: _check_bool(checks, seed, name) for name in IMPLEMENTATION_CHECKS}
        path[seed] = {
            'dev_reproduction': bool(dev_ok),
            'path_dev_same_hardware': same_hardware,
            'path_dev_mismatch_count': mismatch_count,
            'a2_iv_off_matches_base': _check_bool(checks, seed, 'a2_iv_off_matches_base'),
        }
    implementation_ok = all(all(v.values()) for v in implementation.values())
    path_ok = all(v['dev_reproduction'] and v['a2_iv_off_matches_base'] for v in path.values())

    G = sum((quantities[s]['gain'] for s in SEEDS), Fraction()) / len(SEEDS)
    L = sum((quantities[s]['lora'] for s in SEEDS), Fraction()) / len(SEEDS)
    W = sum((quantities[s]['swap'] for s in SEEDS), Fraction()) / len(SEEDS)
    ci = _bootstrap(binary, row_count)
    # The threshold comparison is strict, as preregistered.
    g_lower_positive = ci['G'][0] > 0
    supported = G > 0 and g_lower_positive
    challenged = G <= 0
    direction = ('invalid_implementation' if not implementation_ok or not path_ok else
                 'supported' if supported else 'challenged' if challenged else 'inconclusive')
    secondary_confirmed = direction == 'supported' and L > 0 and ci['L'][0] > 0

    threshold = base_score + Fraction(1, 100)
    learned = {s: {a: quantities[s]['scores'][a] >= threshold for a in ARMS} for s in SEEDS}
    unlearned_controls = {s: not learned[s]['SEG-06'] and not learned[s]['SEG-07'] for s in SEEDS}
    per_seed = {str(s): {
        'gain': number(quantities[s]['gain']),
        'lora': number(quantities[s]['lora']),
        'swap': number(quantities[s]['swap']),
        'boolq': {label: number(score) for label, score in quantities[s]['scores'].items()},
    } for s in SEEDS}
    gains = [quantities[s]['gain'] for s in SEEDS]
    gain_sd = math.sqrt(sum(float((x - G) ** 2) for x in gains) / len(gains))
    rules = {
        '1': {'level': 'diagnostic', 'passed': implementation_ok, 'checks': {str(s): implementation[s] for s in SEEDS},
              'action': 'interpret' if implementation_ok else 'repair_no_conclusion'},
        '2': {'level': 'diagnostic', 'passed': path_ok, 'checks': {str(s): path[s] for s in SEEDS},
              'action': 'interpret' if path_ok else 'invalid_a2_no_conclusion'},
        '3': {'level': 'descriptive', 'per_seed': per_seed, 'G': number(G), 'L': number(L), 'W': number(W),
              'gain_positive_seed_count': sum(x > 0 for x in gains),
              'gain_seed_sd_population': gain_sd,
              'weights': 'equal over seeds 3,4,5'},
        '4': {'level': 'descriptive', 'bootstrap': {'method': 'paired BoolQ row bootstrap',
              'repeats': BOOTSTRAP_REPEATS, 'seed': BOOTSTRAP_SEED, 'shared_indices': True,
              'max_control_recomputed_each_draw': True, 'quantile': 'numpy linear default',
              'G_ci95': ci['G'], 'L_ci95': ci['L']}},
        '5': {'level': 'confirmatory', 'outcome': direction if direction != 'invalid_implementation' else 'not_evaluated',
              'G_positive': G > 0, 'G_ci_lower_positive': g_lower_positive,
              'challenged_G_nonpositive': G <= 0},
        '6': {'level': 'confirmatory' if direction == 'supported' else 'descriptive',
              'evaluated': direction == 'supported', 'confirmed': secondary_confirmed,
              'L_positive': L > 0, 'L_ci_lower_positive': ci['L'][0] > 0,
              'reason': 'confirmed' if secondary_confirmed else
                        'fixed_sequence_not_reached' if direction != 'supported' else 'criterion_not_met'},
        '7': {'level': 'descriptive', 'threshold': number(threshold),
              'learned': {str(s): learned[s] for s in SEEDS},
              'no_history_r128_controls_unlearned': {str(s): unlearned_controls[s] for s in SEEDS},
              'mean_mainly_one_seed': direction == 'supported' and sum(x > 0 for x in gains) < 2},
        '8': {'level': 'descriptive', 'other_metrics': 'PIQA, macro, NLL, dev contrasts and pinned E-0027 comparisons in locked analysis',
              'hypothesis_status_changed': False},
    }
    return {'rules': rules, 'direction': direction,
            'eligible_edge': direction if direction in ('supported', 'challenged') else None,
            'secondary_confirmed': secondary_confirmed, 'G': number(G), 'L': number(L), 'W': number(W),
            'ci95': {'G': ci['G'], 'L': ci['L']}, 'per_seed': per_seed}
