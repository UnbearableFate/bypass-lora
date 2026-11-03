"""E-0032@1 preregistered rules 2-6 on exact BoolQ dev fractions (PIQA only for rule 5).

Returns rule outcomes only; never changes a hypothesis status or starts work.
counts[seed][label][task] = (correct, total). Reference labels per seed:
SEG-00, SEG-04, SEG-05, SEG-06, SEG-07, SEG-05/iv_swap. Variant labels:
SEG-W1, SEG-W2, SEG-X5 (each with '/iv_swap') and SEG-X6, SEG-X7.
"""
from fractions import Fraction

SEEDS = (0, 1, 2)
NEW_SEEDS = (1, 2)
PP = Fraction(1, 100)
REFERENCE = ('SEG-00', 'SEG-04', 'SEG-05', 'SEG-06', 'SEG-07', 'SEG-05/iv_swap')
VARIANTS = ('SEG-W1', 'SEG-W2', 'SEG-X5')       # rule-4 tie order: W1, W2, X
CONTROLS = {'SEG-W1': ('SEG-06', 'SEG-07'), 'SEG-W2': ('SEG-06', 'SEG-07'), 'SEG-X5': ('SEG-X6', 'SEG-X7')}
HYPOTHESIS = {'SEG-W1': 'H-0032', 'SEG-W2': 'H-0032', 'SEG-X5': 'H-0033'}
G_SEG05 = Fraction(88, 3000)                     # D-0062: three-seed SEG-05 gain mean
SCREEN_DELTA = Fraction(3, 200)                  # +1.5 pp
SCREEN_GAIN_X = G_SEG05+SCREEN_DELTA             # 133/3000
REPLICATION_DG = PP                              # mean new-seed delta-gain >= +1 pp
G_MIN, L_MIN, PIQA_DROP = 3*PP, PP, PP


def number(x):
    return {'numerator': x.numerator, 'denominator': x.denominator, 'pp': float(100*x)}


def fraction(counts):
    correct, total = counts
    if type(correct) is not int or type(total) is not int or total <= 0 or not 0 <= correct <= total:
        raise ValueError('invalid exact correctness counts')
    return Fraction(correct, total)


def scores(counts, task):
    totals = {v[task][1] for v in counts.values()}
    if len(totals) != 1:
        raise ValueError(f'paired {task} row counts differ within a seed')
    return {label: fraction(v[task]) for label, v in counts.items()}


def reference_quantities(s):
    return {'gain': s['SEG-05']-max(s['SEG-06'], s['SEG-07']),
            'learned_threshold': s['SEG-00']+PP,
            'controls_learned': max(s['SEG-06'], s['SEG-07']) >= s['SEG-00']+PP}


def variant_quantities(V, s, ref):
    controls = CONTROLS[V]
    gain = s[V]-max(s[c] for c in controls)
    return {'keep': s[V], 'delta': s[V]-s['SEG-05'], 'gain': gain, 'delta_gain': gain-ref['gain'],
            'lora': s[V]-s['SEG-04'], 'swap': s[V]-s[V+'/iv_swap'],
            'controls': {c: s[c] for c in controls},
            'controls_learned': max(s[c] for c in controls) >= ref['learned_threshold']}


def analyze(counts, checks):
    """counts: {seed: {label: {'boolq': (c, t), 'piqa': (c, t)}}}; seed 0 holds every variant
    label; seeds 1 and 2 hold the reference labels plus at most the one replicated variant.
    checks: {rule-1 check name: bool}."""
    if set(counts) - set(SEEDS) or 0 not in counts or any(s not in counts for s in SEEDS):
        raise ValueError('rules need the reference records of seeds 0, 1 and 2')
    for seed in SEEDS:
        if not set(REFERENCE) <= set(counts[seed]):
            raise ValueError(f'seed {seed} lacks reference labels')
    seed0 = set(counts[0]) - set(REFERENCE)
    expected0 = {*VARIANTS, *(v+'/iv_swap' for v in VARIANTS), 'SEG-X6', 'SEG-X7'}
    if seed0 != expected0:
        raise ValueError('seed 0 needs all five E-0032 arms and the three Keep swaps')
    boolq = {s: scores(counts[s], 'boolq') for s in SEEDS}
    piqa = {s: scores(counts[s], 'piqa') for s in SEEDS}
    ref = {s: reference_quantities(boolq[s]) for s in SEEDS}
    if sum(ref[s]['gain'] for s in SEEDS)/3 != G_SEG05:
        raise ValueError('reference SEG-05 gains do not reproduce G_SEG05 = 88/3000')
    implementation_ok = bool(checks) and all(v is True for v in checks.values())

    q0 = {V: variant_quantities(V, boolq[0], ref[0]) for V in VARIANTS}
    screen = {}
    for V in VARIANTS:
        delta_ok = q0[V]['delta'] >= SCREEN_DELTA
        gain_ok = True if V != 'SEG-X5' else q0[V]['gain'] >= SCREEN_GAIN_X
        screen[V] = {'delta>=1.5': delta_ok, 'passed': delta_ok and gain_ok,
                     **({'gain_X>=133/3000': gain_ok} if V == 'SEG-X5' else {})}
    passed = [V for V in VARIANTS if screen[V]['passed']]
    best = max((q0[V]['delta'] for V in passed), default=None)
    selected = next((V for V in passed if q0[V]['delta'] == best), None)

    replicated = {s: set(counts[s]) - set(REFERENCE) for s in NEW_SEEDS}
    if selected is None and any(replicated.values()):
        raise ValueError('no variant passed screening, so no new-seed variant run is allowed')
    expected_new = (set() if selected is None else
                    {selected, selected+'/iv_swap', *(CONTROLS[selected] if selected == 'SEG-X5' else ())})
    complete = selected is not None and all(replicated[s] == expected_new for s in NEW_SEEDS)
    if any(replicated[s] and replicated[s] != expected_new for s in NEW_SEEDS):
        raise ValueError('new-seed runs differ from the one selected variant')

    rule5 = None
    if complete:
        V = selected
        q = {0: q0[V], **{s: variant_quantities(V, boolq[s], ref[s]) for s in NEW_SEEDS}}
        mean_dg = (q[1]['delta_gain']+q[2]['delta_gain'])/2
        G = sum(q[s]['gain'] for s in SEEDS)/3
        W = sum(q[s]['swap'] for s in SEEDS)/3
        L = sum(q[s]['lora'] for s in SEEDS)/3
        piqa_ok = {s: piqa[s][V] >= piqa[s]['SEG-05']-PIQA_DROP for s in SEEDS}
        conditions = {'delta1>0': q[1]['delta'] > 0, 'delta2>0': q[2]['delta'] > 0,
                      'dg1>0': q[1]['delta_gain'] > 0, 'dg2>0': q[2]['delta_gain'] > 0,
                      'mean_dg_new>=1': mean_dg >= REPLICATION_DG, 'G>=3': G >= G_MIN, 'W>=G/2': W >= G/2,
                      'L>=1': L >= L_MIN, **{f'piqa{s}_not_below_SEG05-1': piqa_ok[s] for s in SEEDS}}
        if V == 'SEG-X5':
            conditions.update({f'X_controls_learned{s}': q[s]['controls_learned'] for s in NEW_SEEDS})
        rule5 = {'variant': V, 'supports': all(conditions.values()), 'conditions': conditions,
                 'mean_dg_new': number(mean_dg), 'G': number(G), 'W': number(W), 'L': number(L),
                 'mean_delta_3seed_descriptive': number(sum(q[s]['delta'] for s in SEEDS)/3),
                 'mean_dg_3seed_descriptive': number(sum(q[s]['delta_gain'] for s in SEEDS)/3),
                 'per_seed': {str(s): describe(q[s]) for s in SEEDS}}
        # Contract rule 5: a new seed where X's own controls did not learn makes that
        # seed's gain_X descriptive only, so H-0033 is then inconclusive (no challenge).
        x_unlearned = V == 'SEG-X5' and not all(q[s]['controls_learned'] for s in NEW_SEEDS)
        rule5['x_controls_unlearned'] = x_unlearned
        rule5['challenges'] = mean_dg <= 0 and not x_unlearned

    unsupported = {V: q0[V]['delta'] <= 0 for V in VARIANTS}
    outcomes = {}
    for H, members in (('H-0032', ('SEG-W1', 'SEG-W2')), ('H-0033', ('SEG-X5',))):
        if rule5 and HYPOTHESIS[rule5['variant']] == H and rule5['supports']:
            outcome = 'supports'
        elif all(unsupported[V] for V in members) or (rule5 and HYPOTHESIS[rule5['variant']] == H and rule5['challenges']):
            outcome = 'challenges'
        elif selected in members and not complete:
            outcome = 'pending_replication'
        else:
            outcome = 'inconclusive'
        outcomes[H] = outcome if implementation_ok else 'invalid_implementation'
    return {
        'implementation_ok': implementation_ok,
        'rules': {
            '1': {'level': 'diagnostic', 'passed': implementation_ok, 'checks': dict(checks),
                  'action': 'interpret' if implementation_ok else 'repair_no_conclusion'},
            '2': {'level': 'screening', 'G_SEG05': number(G_SEG05),
                  'reference': {str(s): {'gain_SEG05': number(ref[s]['gain']),
                                         'controls_learned': ref[s]['controls_learned'],
                                         'learned_threshold': number(ref[s]['learned_threshold'])} for s in SEEDS},
                  'seed0': {V: describe(q0[V]) for V in VARIANTS}},
            '3': {'level': 'screening', 'screen': screen, 'passed': passed,
                  'X_delta_gain_seed0_descriptive_only': True},
            '4': {'level': 'screening', 'selected': selected if implementation_ok else None,
                  'rule': 'largest delta(0) among passed; ties W1, W2, X', 'replication_complete': complete},
            '5': rule5,
            '6': {'level': 'screening', 'unsupported_delta0<=0': unsupported,
                  'replication_mean_dg<=0': None if rule5 is None else rule5['mean_dg_new']['pp'] <= 0,
                  'replication_challenge': None if rule5 is None else rule5['challenges']},
        },
        'outcomes': outcomes,
        'reopen_request': bool(implementation_ok and 'supports' in outcomes.values()),
    }


def describe(q):
    return {'keep': number(q['keep']), 'delta': number(q['delta']), 'gain': number(q['gain']),
            'delta_gain': number(q['delta_gain']), 'lora': number(q['lora']), 'swap': number(q['swap']),
            'controls': {c: number(v) for c, v in q['controls'].items()},
            'controls_learned': q['controls_learned']}
