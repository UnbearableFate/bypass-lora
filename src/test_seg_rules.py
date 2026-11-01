from copy import deepcopy

import pytest

from seg_rules import analyze, gain_intervals

CHECKS = dict(off_matches_base=True, initial_matches_base=True, no_channel_identity=True, reload_identical=True)


def grid(gain=30, swap=15, *, controls=600, lo=650, full=700, second_gain=0):
    correctness = {'SEG-00': 500, 'SEG-01': controls, 'SEG-02': controls+gain,
                   'SEG-03': controls-10, 'SEG-04': lo, 'RB1-07': full,
                   'SEG-02/iv_off': 500, 'SEG-02/iv_reset': controls+gain-10,
                   'SEG-02/iv_swap': controls+gain-swap}
    result = {a: {t: (c, 1000) for t in ('piqa', 'boolq')} for a, c in correctness.items()}
    for a in result:
        c, n = result[a]['boolq']
        if a == 'SEG-02': c = controls+second_gain
        elif a == 'SEG-02/iv_swap': c = controls+second_gain-15
        result[a]['boolq'] = c, n
    return result


@pytest.mark.parametrize('gain,drop,outcome,release', [
    (30, 15, 'supports', False), (30, 14, 'descriptive', False),
    (29, 15, 'pending_r128', True), (1, 1, 'pending_r128', True),
    (0, 0, 'challenges', False), (-1, 0, 'challenges', False)])
def test_exact_r32_thresholds(gain, drop, outcome, release):
    report = analyze(grid(gain, drop), CHECKS)
    assert report['primary'] == 'piqa'
    assert report['direction'] == outcome
    assert report['release_r128'] is release


def add128(counts, gain, drop):
    for a, c in {'SEG-05': 600+gain, 'SEG-06': 600, 'SEG-07': 590,
                 'SEG-05/iv_swap': 600+gain-drop, 'SEG-05/iv_off': 500, 'SEG-05/iv_reset': 590}.items():
        counts[a] = {t: (c, 1000) for t in ('piqa', 'boolq')}
    return counts


@pytest.mark.parametrize('g32,g128,drop,outcome', [
    (9, 9, 0, 'challenges'), (10, 9, 0, 'descriptive'), (20, 0, 0, 'descriptive'),
    (9, 10, 0, 'descriptive'), (29, 30, 15, 'supports'), (9, 30, 14, 'descriptive')])
def test_D0044_both_ranks_below_one_and_r128_support(g32, g128, drop, outcome):
    report = analyze(add128(grid(g32, 0), g128, drop), CHECKS)
    assert report['direction'] == outcome
    assert report['release_r128'] is False
    assert report['r128_completed'] is True
    if outcome == 'supports':
        assert report['per_task']['piqa']['support_width'] == 128


def test_dynamic_range_exact_three_boolq_fallback_and_no_usable_information():
    counts = grid(30, 15, lo=670, second_gain=30)
    assert analyze(counts, CHECKS)['primary'] == 'piqa'
    counts['SEG-04']['piqa'] = 671, 1000
    assert analyze(counts, CHECKS)['primary'] == 'boolq'
    counts['SEG-04']['boolq'] = 671, 1000
    report = analyze(counts, CHECKS)
    assert report['primary'] is None and report['eligible_edge'] is None
    assert report['rules']['3']['action'] == 'stop_no_usable_information_removed'


def test_secondary_only_cannot_create_primary_edge_or_release():
    counts = grid(0, 0, second_gain=30)
    report = analyze(counts, CHECKS)
    assert report['direction'] == 'inconclusive_secondary_only'
    assert report['eligible_edge'] is None and report['release_r128'] is False
    assert report['rules']['8']['two_new_seed_replications_needed']


def test_unlearned_controls_are_diagnostic_and_failed_implementation_blocks_release():
    counts = grid(30, 15, controls=500, lo=500)
    r = analyze(counts, CHECKS)
    assert r['rules']['4']['no_history_controls_unlearned']
    assert r['rules']['5']['level'] == 'diagnostic' and r['eligible_edge'] is None
    r = analyze(grid(20), dict(CHECKS, reload_identical=False))
    assert r['direction'] == 'invalid_implementation'
    assert not r['release_r128'] and r['eligible_edge'] is None


def test_same_allocation_normal_reference_for_swap_drop():
    counts = grid(30, 14)
    counts['SEG-02/normal'] = dict(piqa=(631, 1000), boolq=(600, 1000))
    assert analyze(counts, CHECKS)['direction'] == 'supports'


def test_no_incomplete_grid_or_unauthorized_widening():
    with pytest.raises(ValueError, match='release gate'):
        analyze(add128(grid(30), 30, 15), CHECKS)
    counts = grid(); del counts['SEG-02/iv_reset']
    with pytest.raises(ValueError, match='intervention'):
        analyze(counts, CHECKS)
    with pytest.raises(ValueError, match='all three'):
        analyze(dict(grid(), **{'SEG-05': grid()['SEG-02']}), CHECKS)


def test_gain_bootstrap_max_control_and_shared_deterministic_draws():
    rows = {a: {t: [int(i < correct) for i in range(10)] for t in ('piqa', 'boolq')}
            for a, correct in {'SEG-00': 3, 'SEG-01': 4, 'SEG-02': 6, 'SEG-03': 5}.items()}
    first = gain_intervals(rows, repeats=137)
    assert first == gain_intervals(deepcopy(rows), repeats=137)
    assert first['32']['piqa']['ci95_pp'][0] >= 0
    rows['SEG-01'] = deepcopy(rows['SEG-02'])
    assert gain_intervals(rows, repeats=137)['32']['piqa']['ci95_pp'] == [0., 0.]
