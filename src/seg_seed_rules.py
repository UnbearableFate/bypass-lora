"""E-0027@1 preregistered rules 1-9 on exact BoolQ fractions (seeds 0, 1, 2).

Returns rule outcomes only; never changes a hypothesis status or starts work.
"""
from fractions import Fraction

SEEDS = (0, 1, 2)
NEW_SEEDS = (1, 2)
ARMS = ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07')
LABELS = (*ARMS, 'SEG-05/normal', 'SEG-05/iv_swap')
PP = Fraction(1, 100)
# E-0027 Design rule 2: the seed-0 quantities taken from E-0025's locked analysis.
SEED0 = {'gain': Fraction(11, 250), 'lora': Fraction(17, 1000), 'swap': Fraction(65, 1000)}
IMPLEMENTATION_CHECKS = ('off_matches_base', 'initial_matches_base', 'no_channel_identity',
                         'reload_identical', 'initial_parameters_identical')


def number(x):
    return {'numerator': x.numerator, 'denominator': x.denominator, 'pp': float(100*x)}


def fraction(counts):
    correct, total = counts
    if type(correct) is not int or type(total) is not int or total <= 0 or not 0 <= correct <= total:
        raise ValueError('invalid exact BoolQ correctness counts')
    return Fraction(correct, total)


def per_seed(counts):
    """counts: {label: (correct, total)} for one seed, BoolQ only."""
    if set(counts) != set(LABELS):
        raise ValueError('each seed needs SEG-04..07 plus the SEG-05 normal/swap intervention pair')
    totals = {c[1] for c in counts.values()}
    if len(totals) != 1:
        raise ValueError('paired BoolQ row counts differ within a seed')
    s = {k: fraction(v) for k, v in counts.items()}
    control = 'SEG-06' if s['SEG-06'] >= s['SEG-07'] else 'SEG-07'
    return {'scores': s, 'better_control': control,
            'gain': s['SEG-05']-max(s['SEG-06'], s['SEG-07']),
            'lora': s['SEG-05']-s['SEG-04'],
            'swap': s['SEG-05/normal']-s['SEG-05/iv_swap']}


def analyze(counts, checks, base):
    """counts: {seed: {label: (correct, total)}} BoolQ; checks: {seed: {check: bool}} for new seeds;
    base: SEG-00 Base_seg BoolQ (correct, total)."""
    if set(counts) != set(SEEDS):
        raise ValueError('rules need seeds 0, 1 and 2')
    if set(checks) != set(NEW_SEEDS):
        raise ValueError('implementation checks are required for both new seeds')
    quantities = {s: per_seed(counts[s]) for s in SEEDS}
    for name, value in SEED0.items():
        if quantities[0][name] != value:
            raise ValueError(f'seed-0 {name} differs from the E-0025 locked analysis')
    base_score = fraction(base)
    if any(base[1] != c[1] for s in SEEDS for c in counts[s].values()):
        raise ValueError('Base_seg row count differs from the arm rows')
    implementation = {s: {k: checks[s].get(k) is True for k in IMPLEMENTATION_CHECKS} for s in NEW_SEEDS}
    implementation_ok = all(all(v.values()) for v in implementation.values())
    G = sum(quantities[s]['gain'] for s in SEEDS)/3
    L = sum(quantities[s]['lora'] for s in SEEDS)/3
    W = sum(quantities[s]['swap'] for s in SEEDS)/3
    gain1, gain2 = quantities[1]['gain'], quantities[2]['gain']
    supports = G >= 3*PP and gain1 > 0 and gain2 > 0 and W >= G/2 and L >= PP
    challenges = G < PP or (gain1 <= 0 and gain2 <= 0) or L <= 0
    assert not (supports and challenges)
    direction = ('invalid_implementation' if not implementation_ok else
                 'supports' if supports else 'challenges' if challenges else 'inconclusive')
    threshold = base_score+PP
    learned = {s: {a: quantities[s]['scores'][a] >= threshold for a in ARMS} for s in SEEDS}
    unlearned = {s: not learned[s]['SEG-06'] and not learned[s]['SEG-07'] for s in SEEDS}
    seeds = {str(s): {'gain': number(q['gain']), 'lora': number(q['lora']), 'swap': number(q['swap']),
                      'better_control': q['better_control'],
                      'boolq': {k: number(v) for k, v in q['scores'].items()}} for s, q in quantities.items()}
    rules = {
        '1': {'level': 'diagnostic', 'passed': implementation_ok, 'checks': implementation,
              'action': 'interpret' if implementation_ok else 'repair_no_conclusion'},
        '2': {'level': 'screening', 'per_seed': seeds,
              'seed0_source': 'E-0025 locked analysis ATT-0303-1 (not rerun)'},
        '3': {'level': 'screening', 'G': number(G), 'L': number(L), 'W': number(W), 'weights': 'equal over seeds 0,1,2'},
        '4': {'level': 'screening', 'supports': bool(implementation_ok and supports),
              'conditions': {'G>=3': G >= 3*PP, 'gain1>0': gain1 > 0, 'gain2>0': gain2 > 0,
                             'W>=G/2': W >= G/2, 'L>=1': L >= PP}},
        '5': {'level': 'screening', 'challenges': bool(implementation_ok and challenges),
              'conditions': {'G<1': G < PP, 'gain1<=0 and gain2<=0': gain1 <= 0 and gain2 <= 0, 'L<=0': L <= 0}},
        '6': {'level': 'screening', 'inconclusive': direction == 'inconclusive',
              'no_more_seeds_width_tap_loop': direction == 'inconclusive'},
        '7': {'level': 'screening', 'threshold': number(threshold), 'learned': {str(s): v for s, v in learned.items()},
              'no_history_r128_controls_unlearned': {str(s): v for s, v in unlearned.items()}},
        '8': {'level': 'screening', 'piqa_macro_nll': 'descriptive_only', 'hypothesis_status_changed': False},
        '9': {'level': 'screening', 'H-0019': 'remains inconclusive',
              'stage_report': ('BoolQ task-level supports report' if direction == 'supports' else
                               'stop side-state line per E-0025 rule 10; mechanistic stage report'
                               if direction == 'challenges' else None)},
    }
    return {'rules': rules, 'direction': direction,
            'eligible_edge': direction if direction in ('supports', 'challenges') else None,
            'G': number(G), 'L': number(L), 'W': number(W), 'per_seed': seeds}
