"""Synthetic tests for the pure E-0030 A2 BoolQ rules."""
import numpy as np
import pytest

import seg_a2_rules as rules


def counts_for(gain=1, lora=0, swap=0, total=1000):
    control = total * 3 // 5
    keep = control + gain
    return {
        'SEG-04': (keep - lora, total),
        'SEG-05': (keep, total),
        'SEG-06': (control, total),
        'SEG-07': (max(0, control - 10), total),
        'SEG-05/iv_swap': (keep - swap, total),
    }


def passing_checks():
    return {s: {
        **{key: True for key in rules.IMPLEMENTATION_CHECKS},
        'path_dev_same_hardware': True,
        'path_dev_mismatch_count': 0,
        'a2_iv_off_matches_base': True,
    } for s in rules.SEEDS}


def row_vectors(total=1000, gain_positions=(0,), lora_positions=(), swap_positions=()):
    """Build valid paired rows where Keep beats each control at selected rows."""
    values = {}
    for seed in rules.SEEDS:
        values[seed] = {}
        for arm in rules.ARMS:
            values[seed][arm] = np.zeros(total, dtype=np.int8)
        values[seed]['SEG-05'][list(gain_positions)] = 1
        values[seed]['SEG-04'][list(lora_positions)] = 1
        values[seed]['SEG-06'][:] = 0
        values[seed]['SEG-07'][:] = 0
        values[seed]['SEG-05/iv_swap'] = np.zeros(total, dtype=np.int8)
        values[seed]['SEG-05/iv_swap'][list(swap_positions)] = 1
    return values


def vectors_from_counts(seed_counts):
    values = {}
    for seed, seed_counts_for_seed in seed_counts.items():
        values[seed] = {}
        for label in rules.LABELS:
            correct, total = seed_counts_for_seed[label]
            values[seed][label] = np.array(
                [1] * correct + [0] * (total - correct), dtype=np.int8)
    return values


def test_exact_point_quantities_and_fixed_sequence_secondary_gate(monkeypatch):
    monkeypatch.setattr(rules, 'BOOTSTRAP_REPEATS', 32)
    counts = {s: counts_for(gain=100, lora=2, swap=3) for s in rules.SEEDS}
    rows = vectors_from_counts(counts)
    result = rules.analyze(counts, passing_checks(), (590, 1000), rows)
    assert result['G']['numerator'] == 1 and result['G']['denominator'] == 10
    assert result['L']['numerator'] == 1 and result['L']['denominator'] == 500
    assert result['per_seed']['3']['swap']['numerator'] == 3
    assert result['direction'] == 'supported'
    assert result['rules']['6']['evaluated'] is True
    assert result['secondary_confirmed'] is False  # L and its lower bound do not clear zero.


def test_bootstrap_uses_shared_indices_and_recomputes_max_control_per_draw(monkeypatch):
    monkeypatch.setattr(rules, 'BOOTSTRAP_REPEATS', 17)
    n = 5
    data = row_vectors(n, gain_positions=(0, 1), lora_positions=(2,), swap_positions=(3,))
    # Distinct control profiles make the winning control vary across resamples.
    for seed in rules.SEEDS:
        data[seed]['SEG-06'] = np.array([1, 1, 0, 0, 0], dtype=np.int8)
        data[seed]['SEG-07'] = np.array([0, 0, 1, 1, 1], dtype=np.int8)
        data[seed]['SEG-05'] = np.array([1, 1, 1, 0, 0], dtype=np.int8)
        data[seed]['SEG-04'] = np.array([0, 0, 0, 1, 1], dtype=np.int8)

    # Independent locked-stream calculation; one draw matrix is shared by all
    # seeds and arms and the max is applied after each resampled accuracy.
    rng = np.random.default_rng(rules.BOOTSTRAP_SEED)
    draw = rng.integers(0, n, size=(rules.BOOTSTRAP_REPEATS, n))
    gs, ls = [], []
    for indices in draw:
        gain_count, lora_count = 0, 0
        for seed in rules.SEEDS:
            correct = {a: int(data[seed][a][indices].sum()) for a in rules.ARMS}
            gain_count += correct['SEG-05'] - max(correct['SEG-06'], correct['SEG-07'])
            lora_count += correct['SEG-05'] - correct['SEG-04']
        gs.append(gain_count / (len(rules.SEEDS) * n))
        ls.append(lora_count / (len(rules.SEEDS) * n))
    got = rules._bootstrap(data, n)
    assert got['G'] == [float(x) for x in np.quantile(gs, [.025, .975])]
    assert got['L'] == [float(x) for x in np.quantile(ls, [.025, .975])]


def test_zero_gain_bootstrap_stays_exactly_zero(monkeypatch):
    monkeypatch.setattr(rules, 'BOOTSTRAP_REPEATS', 32)
    n = 23
    data = {s: {label: np.zeros(n, dtype=np.int8) for label in rules.LABELS}
            for s in rules.SEEDS}
    for seed in rules.SEEDS:
        data[seed]['SEG-06'] = np.ones(n, dtype=np.int8)
        data[seed]['SEG-07'] = np.ones(n, dtype=np.int8)
        data[seed]['SEG-05'] = np.ones(n, dtype=np.int8)
        data[seed]['SEG-04'] = np.ones(n, dtype=np.int8)
    intervals = rules._bootstrap(data, n)
    assert intervals['G'] == [0.0, 0.0]
    assert intervals['L'] == [0.0, 0.0]


def test_strict_interval_lower_boundary_is_not_supported(monkeypatch):
    monkeypatch.setattr(rules, 'BOOTSTRAP_REPEATS', 10_000)
    n = 100
    # One positive row makes the exact mean positive; most paired draws omit it,
    # so the lower linear quantile is exactly zero.
    counts = {s: counts_for(gain=1, total=n) for s in rules.SEEDS}
    rows = vectors_from_counts(counts)
    result = rules.analyze(counts, passing_checks(), (0, n), rows)
    assert result['G']['numerator'] > 0
    assert result['ci95']['G'][0] == 0
    assert result['direction'] == 'inconclusive'
    assert result['rules']['5']['G_ci_lower_positive'] is False


def test_fixed_sequence_skips_lora_confirmation_without_primary_support(monkeypatch):
    monkeypatch.setattr(rules, 'BOOTSTRAP_REPEATS', 64)
    counts = {s: counts_for(gain=0, lora=100, total=1000) for s in rules.SEEDS}
    rows = row_vectors(1000, lora_positions=range(100))
    result = rules.analyze(counts, passing_checks(), (590, 1000), rows)
    assert result['direction'] == 'challenged'
    assert result['rules']['6']['evaluated'] is False
    assert result['rules']['6']['reason'] == 'fixed_sequence_not_reached'


@pytest.mark.parametrize('bad_check', [
    'off_matches_base_dev', 'initial_matches_base_dev', 'no_channel_identity',
    'reload_identical', 'initial_parameters_identical', 'train_ids_disjoint_a2',
    'a2_ids_and_hashes_match', 'a2_iv_off_matches_base',
])
def test_failed_implementation_or_path_check_blocks_confirmation(monkeypatch, bad_check):
    monkeypatch.setattr(rules, 'BOOTSTRAP_REPEATS', 16)
    checks = passing_checks()
    checks[4][bad_check] = False
    result = rules.analyze(
        {s: counts_for(gain=20) for s in rules.SEEDS}, checks, (590, 1000),
        vectors_from_counts({s: counts_for(gain=20) for s in rules.SEEDS}),
    )
    assert result['direction'] == 'invalid_implementation'
    assert result['eligible_edge'] is None
    assert result['rules']['5']['outcome'] == 'not_evaluated'


def test_path_gate_allows_at_most_ten_mismatches_only_on_different_hardware(monkeypatch):
    monkeypatch.setattr(rules, 'BOOTSTRAP_REPEATS', 8)
    checks = passing_checks()
    checks[3]['path_dev_same_hardware'] = False
    checks[3]['path_dev_mismatch_count'] = 10
    counts = {s: counts_for(total=10) for s in rules.SEEDS}
    rows = vectors_from_counts(counts)
    result = rules.analyze(counts, checks, (6, 10), rows)
    assert result['rules']['2']['passed'] is True
    checks[3]['path_dev_mismatch_count'] = 11
    result = rules.analyze(counts, checks, (6, 10), rows)
    assert result['direction'] == 'invalid_implementation'


def test_input_shape_and_invalid_boolean_checks_are_rejected(monkeypatch):
    monkeypatch.setattr(rules, 'BOOTSTRAP_REPEATS', 8)
    counts = {s: counts_for(total=10) for s in rules.SEEDS}
    rows = vectors_from_counts(counts)
    with pytest.raises(ValueError, match='row count'):
        rules.analyze(counts, passing_checks(), (6, 10), row_vectors(9))
    checks = passing_checks()
    checks[3]['path_dev_same_hardware'] = 1
    with pytest.raises(ValueError, match='must be boolean'):
        rules.analyze(counts, checks, (6, 10), rows)
