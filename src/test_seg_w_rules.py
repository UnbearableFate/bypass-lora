"""E-0032@1 rules 2-6: exact boundaries on synthetic counts (no model or GPU)."""
from fractions import Fraction

import pytest

from seg_w_rules import analyze, G_SEG05

# BoolQ correct counts of the recorded E-0025/E-0027 references (1000 rows each).
REFERENCE = {0: {'SEG-00': 609, 'SEG-04': 642, 'SEG-05': 659, 'SEG-06': 615, 'SEG-07': 612, 'SEG-05/iv_swap': 594},
             1: {'SEG-00': 609, 'SEG-04': 634, 'SEG-05': 657, 'SEG-06': 609, 'SEG-07': 631, 'SEG-05/iv_swap': 595},
             2: {'SEG-00': 609, 'SEG-04': 631, 'SEG-05': 653, 'SEG-06': 613, 'SEG-07': 635, 'SEG-05/iv_swap': 583}}
REF_GAIN = {0: 44, 1: 26, 2: 18}
CHECKS = {'initial_matches_base': True, 'off_matches_base': True, 'reload_identical': True}


def entry(boolq, piqa=690):
    return {'boolq': (boolq, 1000), 'piqa': (piqa, 1000)}


def counts(seed0, new=None, piqa=None):
    """seed0: {variant: (delta, swap_drop)} plus 'X': (reset, t0); new: {seed: {label: boolq}}."""
    piqa = piqa or {}
    result = {s: {k: entry(v) for k, v in REFERENCE[s].items()} for s in (0, 1, 2)}
    for V in ('SEG-W1', 'SEG-W2', 'SEG-X5'):
        delta, swap = seed0[V]
        result[0][V] = entry(659+delta, piqa.get((0, V), 690))
        result[0][V+'/iv_swap'] = entry(659+delta-swap)
    reset, t0 = seed0['X']
    result[0]['SEG-X6'], result[0]['SEG-X7'] = entry(reset), entry(t0)
    for s, labels in (new or {}).items():
        for label, value in labels.items():
            result[s][label] = entry(value, piqa.get((s, label), 690))
    return result


BASE0 = {'SEG-W1': (0, 60), 'SEG-W2': (0, 60), 'SEG-X5': (0, 60), 'X': (612, 612)}


def test_reference_reproduces_G_SEG05():
    assert sum(REF_GAIN.values()) == 88 and G_SEG05 == Fraction(88, 3000)
    result = analyze(counts(BASE0), CHECKS)
    assert result['rules']['2']['reference']['0']['gain_SEG05']['numerator'] == 11  # 44/1000
    assert result['rules']['2']['reference']['0']['controls_learned'] is False
    assert result['rules']['2']['reference']['1']['controls_learned'] is True


@pytest.mark.parametrize('delta, passed', [(15, True), (14, False)])
def test_W_screen_threshold_is_inclusive(delta, passed):
    result = analyze(counts({**BASE0, 'SEG-W2': (delta, 60)}), CHECKS)
    assert result['rules']['3']['screen']['SEG-W2']['passed'] is passed
    assert result['rules']['4']['selected'] == ('SEG-W2' if passed else None)
    assert result['outcomes']['H-0032'] == ('pending_replication' if passed else 'inconclusive')


@pytest.mark.parametrize('delta, control, passed', [
    (15, 629, True),    # gain 45/1000 >= 133/3000
    (15, 630, False),   # gain 44/1000 < 133/3000 although delta passes
    (14, 600, False),   # gain passes, delta does not
])
def test_X_screen_needs_delta_and_absolute_gain(delta, control, passed):
    result = analyze(counts({**BASE0, 'SEG-X5': (delta, 60), 'X': (control, 600)}), CHECKS)
    assert result['rules']['3']['screen']['SEG-X5']['passed'] is passed


def test_selection_by_largest_delta_with_W1_W2_X_tie_order():
    result = analyze(counts({'SEG-W1': (20, 60), 'SEG-W2': (20, 60), 'SEG-X5': (19, 60), 'X': (600, 600)}), CHECKS)
    assert result['rules']['3']['passed'] == ['SEG-W1', 'SEG-W2', 'SEG-X5']
    assert result['rules']['4']['selected'] == 'SEG-W1'
    result = analyze(counts({'SEG-W1': (16, 60), 'SEG-W2': (20, 60), 'SEG-X5': (20, 60), 'X': (600, 600)}), CHECKS)
    assert result['rules']['4']['selected'] == 'SEG-W2'
    assert result['outcomes'] == {'H-0032': 'pending_replication', 'H-0033': 'inconclusive'}
    # X wins on delta even though its seed-0 delta-gain is larger for W (descriptive only).
    result = analyze(counts({'SEG-W1': (16, 60), 'SEG-W2': (0, 60), 'SEG-X5': (17, 60), 'X': (629, 600)}), CHECKS)
    assert result['rules']['4']['selected'] == 'SEG-X5'


def w1_new(d1, d2, swap=60):
    return {1: {'SEG-W1': 657+d1, 'SEG-W1/iv_swap': 657+d1-swap}, 2: {'SEG-W1': 653+d2, 'SEG-W1/iv_swap': 653+d2-swap}}


@pytest.mark.parametrize('d1, d2, H32', [
    (10, 10, 'supports'),       # mean new-seed delta-gain exactly +1 pp
    (10, 9, 'inconclusive'),    # mean 0.95 pp
    (20, 0, 'inconclusive'),    # delta2 = 0
    (5, -5, 'challenges'),      # mean <= 0
    (-1, -1, 'challenges'),
])
def test_W_replication_boundaries(d1, d2, H32):
    seed0 = {**BASE0, 'SEG-W1': (20, 60)}
    result = analyze(counts(seed0, w1_new(d1, d2)), CHECKS)
    assert result['rules']['4']['replication_complete'] is True
    assert result['outcomes']['H-0032'] == H32
    assert result['reopen_request'] is (H32 == 'supports')


def test_W_three_seed_G_W_L_and_piqa_conditions():
    seed0 = {**BASE0, 'SEG-W1': (20, 60)}
    # W = mean swap must be >= G/2: G = (64+36+28)/3000; swap 21 everywhere fails.
    low_swap = analyze(counts({**seed0, 'SEG-W1': (20, 21)}, w1_new(10, 10, swap=21)), CHECKS)
    assert low_swap['rules']['5']['conditions']['W>=G/2'] is False
    assert low_swap['outcomes']['H-0032'] == 'inconclusive'
    piqa = analyze(counts(seed0, w1_new(10, 10), piqa={(1, 'SEG-W1'): 679}), CHECKS)
    assert piqa['rules']['5']['conditions']['piqa1_not_below_SEG05-1'] is False
    assert piqa['outcomes']['H-0032'] == 'inconclusive'
    edge = analyze(counts(seed0, w1_new(10, 10), piqa={(1, 'SEG-W1'): 680}), CHECKS)
    assert edge['outcomes']['H-0032'] == 'supports'


def x_new(d, controls):
    return {s: {'SEG-X5': base+d, 'SEG-X5/iv_swap': base+d-60, 'SEG-X6': 600, 'SEG-X7': controls[s]}
            for s, base in ((1, 657), (2, 653))}


def test_X_replication_requires_its_own_learned_controls():
    seed0 = {**BASE0, 'SEG-X5': (20, 60), 'X': (620, 600)}
    good = analyze(counts(seed0, x_new(15, {1: 620, 2: 619})), CHECKS)
    assert good['rules']['4']['selected'] == 'SEG-X5'
    assert good['outcomes']['H-0033'] == 'supports'
    unlearned = analyze(counts(seed0, x_new(15, {1: 620, 2: 618})), CHECKS)
    assert unlearned['rules']['5']['conditions']['X_controls_learned2'] is False
    assert unlearned['outcomes']['H-0033'] == 'inconclusive'


def test_X_unlearned_new_seed_controls_block_a_replication_challenge():
    # RV-0105 M1: mean new-seed delta-gain <= 0, but a new seed's X controls did not
    # learn, so that seed's gain_X is descriptive only and H-0033 is inconclusive.
    seed0 = {**BASE0, 'SEG-X5': (20, 60), 'X': (620, 600)}
    unlearned = analyze(counts(seed0, x_new(-15, {1: 618, 2: 620})), CHECKS)
    assert unlearned['rules']['5']['mean_dg_new']['pp'] <= 0
    assert unlearned['rules']['5']['x_controls_unlearned'] is True
    assert unlearned['rules']['5']['challenges'] is False
    assert unlearned['outcomes']['H-0033'] == 'inconclusive'
    assert unlearned['rules']['6']['replication_mean_dg<=0'] is True
    assert unlearned['rules']['6']['replication_challenge'] is False
    learned = analyze(counts(seed0, x_new(-15, {1: 620, 2: 620})), CHECKS)
    assert learned['rules']['5']['x_controls_unlearned'] is False
    assert learned['outcomes']['H-0033'] == 'challenges'
    assert learned['rules']['6']['replication_challenge'] is True


def test_seed0_challenges_and_invalid_implementation():
    result = analyze(counts({'SEG-W1': (0, 60), 'SEG-W2': (-3, 60), 'SEG-X5': (0, 60), 'X': (600, 600)}), CHECKS)
    assert result['outcomes'] == {'H-0032': 'challenges', 'H-0033': 'challenges'}
    one = analyze(counts({'SEG-W1': (1, 60), 'SEG-W2': (-3, 60), 'SEG-X5': (1, 60), 'X': (600, 600)}), CHECKS)
    assert one['outcomes'] == {'H-0032': 'inconclusive', 'H-0033': 'inconclusive'}
    broken = analyze(counts(BASE0), {**CHECKS, 'reload_identical': False})
    assert set(broken['outcomes'].values()) == {'invalid_implementation'} and not broken['reopen_request']


def test_inputs_must_match_the_selection():
    with pytest.raises(ValueError):
        analyze(counts(BASE0, w1_new(10, 10)), CHECKS)                 # nothing passed screening
    seed0 = {**BASE0, 'SEG-W1': (20, 60)}
    with pytest.raises(ValueError):
        analyze(counts(seed0, {1: {'SEG-W2': 670, 'SEG-W2/iv_swap': 600}}), CHECKS)
    data = counts(BASE0)
    del data[0]['SEG-X7']
    with pytest.raises(ValueError):
        analyze(data, CHECKS)
    data = counts(BASE0)
    data[1]['SEG-05'] = entry(658)
    with pytest.raises(ValueError, match='G_SEG05'):
        analyze(data, CHECKS)
