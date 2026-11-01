"""Synthetic tests for E-0021 exact screening rules and paired bootstrap."""
import rb1_rules
import pytest


def _counts(default=(500, 1000), **changes):
    result = {label: {task: default for task in rb1_rules.TASKS}
              for label in ('Base', *rb1_rules.ARM_IDS)}
    for label, values in changes.items():
        result[label] = {task: values.get(task, default) for task in rb1_rules.TASKS}
    return result


def _report(counts, nll=None, **kwargs):
    return rb1_rules.analyze_counts(counts, nll, **kwargs)


def test_macro_accuracy_uses_exact_task_stratified_fractions():
    rows = [
        {'task': 'piqa', 'gold': 0, 'prediction': 0},
        {'task': 'piqa', 'gold': 0, 'prediction': 1},
        {'task': 'boolq', 'gold': 1, 'prediction': 1},
    ]
    got = rb1_rules.counts_from_rows(rows)
    assert got == {'piqa': (1, 2), 'boolq': (1, 1)}


def test_exact_rule10_upper_boundary_is_excluded_and_gates_still_release():
    # The exact Base+1pp upper boundary is outside Rule 10's strict upper bound.
    counts = _counts(**{
        'RB1-07': {'piqa': (520, 1000), 'boolq': (500, 1000)},
        'RB1-02': {'piqa': (520, 1000), 'boolq': (500, 1000)},
        'RB1-08': {'piqa': (520, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['rules']['1_G0_positive_control']['passed'] is True
    assert report['rules']['10_all_arms_within_Base_plus_minus_1pp']['passed'] is False
    assert report['progression'] == 'draft_RB5_and_review_conclusion'
    assert report['rules']['7_RB5_release']['passed'] is True
    assert report['rules']['7_RB5_release']['gate_condition_met'] is True


def test_s_f_star_prefers_s2_when_difference_is_strictly_below_half_pp():
    counts = {label: {task: (500, 1000) for task in rb1_rules.TASKS}
              for label in ('Base', *rb1_rules.ARM_IDS)}
    # A 0.5 percentage point difference on one task is a 0.25 pp macro gap.
    counts['RB1-03']['piqa'] = (500, 1000)
    counts['RB1-03']['boolq'] = (500, 1000)
    counts['RB1-04']['piqa'] = (505, 1000)
    report = _report(counts)
    assert report['rules']['5_GL_loop']['S2_minus_S4'] == {
        'numerator': -1, 'denominator': 400, 'value': -0.0025,
        'percentage_points': -0.25,
    }
    assert report['selection']['S_F_star'] == 'RB1-03'


def test_exact_half_pp_depth_difference_chooses_higher_macro_s4():
    counts = {label: {task: (500, 1000) for task in rb1_rules.TASKS}
              for label in ('Base', *rb1_rules.ARM_IDS)}
    counts['RB1-03']['piqa'] = (490, 1000)
    counts['RB1-04']['piqa'] = (500, 1000)
    report = _report(counts)
    assert report['rules']['5_GL_loop']['S2_minus_S4']['percentage_points'] == -0.5
    assert report['selection']['S_F_star'] == 'RB1-04'


def test_s4_positive_loop_contrast_does_not_support_gl_without_fixed_query_gate():
    counts = _counts(**{
        'RB1-03': {'piqa': (500, 1000), 'boolq': (500, 1000)},
        'RB1-04': {'piqa': (540, 1000), 'boolq': (500, 1000)},
        'RB1-05': {'piqa': (530, 1000), 'boolq': (500, 1000)},
        'RB1-02': {'piqa': (500, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['rules']['5_GL_loop']['selected'] == 'RB1-04'
    assert report['rules']['5_GL_loop']['selected_minus_RB1-02']['percentage_points'] == 2
    assert report['rules']['5_GL_loop']['S4_minus_fixed_query']['percentage_points'] == 0.5
    assert report['rules']['5_GL_loop']['passed'] is False
    assert report['screening_directions']['loop'] == 'descriptive'


def test_c_star_strict_near_best_boundary_excludes_exactly_one_pp_behind():
    counts = _counts(**{
        'RB1-08': {'piqa': (540, 1000), 'boolq': (500, 1000)},
        'RB1-09': {'piqa': (500, 1000), 'boolq': (500, 1000)},
    })
    # C_up is exactly one macro pp behind C_dual and must be excluded.
    counts['RB1-09']['piqa'] = (520, 1000)
    report = _report(counts)
    assert report['selection']['effective_concat_arms'] == ['RB1-08', 'RB1-09']
    assert report['selection']['near_best_effective_concat_arms'] == ['RB1-08']
    assert report['selection']['C_star_RB'] == 'RB1-08'


def test_c_star_prefers_lower_parameter_c_up_when_both_are_near_best():
    counts = _counts(**{
        'RB1-08': {'piqa': (530, 1000), 'boolq': (500, 1000)},
        'RB1-09': {'piqa': (520, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['selection']['near_best_effective_concat_arms'] == ['RB1-08', 'RB1-09']
    assert report['selection']['C_star_RB'] == 'RB1-09'


def test_raw_gc_without_effective_concat_does_not_release_or_trigger_pilot():
    counts = _counts(**{
        'RB1-07': {'piqa': (520, 1000), 'boolq': (500, 1000)},
        'RB1-02': {'piqa': (460, 1000), 'boolq': (500, 1000)},
        'RB1-08': {'piqa': (480, 1000), 'boolq': (500, 1000)},
        'RB1-09': {'piqa': (480, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['rules']['6_GC_concat']['raw_gate_met'] is True
    assert report['rules']['6_GC_concat']['resolved_gate_met'] is False
    assert report['selection']['C_star_RB'] is None
    assert report['rules']['7_RB5_release']['passed'] is False
    assert report['rules']['8_C_S1_S4_pilot']['triggered'] is False
    assert report['rules']['9_no_residual_or_concat_signal']['triggered'] is True
    assert report['raw_gates']['GC'] is True
    assert report['resolved_gates']['GC'] is False
    assert report['screening_directions']['concat'] == 'supports'


def test_rule10_includes_base_minus_one_but_excludes_base_plus_one():
    counts = _counts(**{
        'RB1-01': {'piqa': (480, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['rules']['10_all_arms_within_Base_plus_minus_1pp']['passed'] is True

    counts['RB1-02'] = {'piqa': (520, 1000), 'boolq': (500, 1000)}
    report = _report(counts)
    assert report['rules']['10_all_arms_within_Base_plus_minus_1pp']['passed'] is False


def test_missing_intervention_data_is_pending_and_does_not_challenge_history():
    counts = _counts(**{
        'RB1-02': {'piqa': (510, 1000), 'boolq': (500, 1000)},
        'RB1-01': {'piqa': (500, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['rules']['4_GH_side_history']['passed'] is None
    assert report['rules']['4_GH_side_history']['assessed'] is False
    assert report['screening_directions']['history'] == 'descriptive'


def test_history_gate_includes_exact_one_point_boundaries():
    counts = _counts(**{
        'RB1-02': {'piqa': (520, 1000), 'boolq': (500, 1000)},
        'RB1-01': {'piqa': (500, 1000), 'boolq': (500, 1000)},
        'RB1-02/iv_swap': {'piqa': (500, 1000), 'boolq': (500, 1000)},
        'RB1-02/iv_permute': {'piqa': (500, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['rules']['4_GH_side_history']['passed'] is True
    assert report['screening_directions']['history'] == 'supports'


def test_history_drops_use_same_arm_normal_when_it_differs_from_training_run():
    counts = _counts(**{
        'RB1-02': {'piqa': (540, 1000), 'boolq': (500, 1000)},
        'RB1-01': {'piqa': (500, 1000), 'boolq': (500, 1000)},
        'RB1-02/normal': {'piqa': (530, 1000), 'boolq': (500, 1000)},
        'RB1-02/iv_swap': {'piqa': (520, 1000), 'boolq': (500, 1000)},
        'RB1-02/iv_permute': {'piqa': (520, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    gh = report['rules']['4_GH_side_history']
    assert gh['normal_reference'] == 'RB1-02/normal'
    assert gh['RB1-02_vs_RB1-01']['percentage_points'] == 2
    assert gh['drop_vs_normal']['iv_swap']['percentage_points'] == 0.5
    assert gh['passed'] is False


def test_main_support_requires_each_task_difference_not_only_macro():
    counts = _counts(**{
        'RB1-02': {'piqa': (650, 1000), 'boolq': (370, 1000)},
        'RB1-07': {'piqa': (500, 1000), 'boolq': (500, 1000)},
        'Base': {'piqa': (500, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['rules']['11_screening_directions']['main_utility'] == 'descriptive'
    assert report['rules']['11_screening_directions']['main_utility'] == report['screening_directions']['main_utility']


def test_gr_tolerates_minus_one_but_main_hypothesis_needs_plus_one():
    counts = _counts(**{
        'RB1-07': {'piqa': (540, 1000), 'boolq': (500, 1000)},
        'RB1-02': {'piqa': (520, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    assert report['rules']['3_GR_residual_single_read']['RB1-02_vs_RB1-07']['percentage_points'] == -1
    assert report['rules']['3_GR_residual_single_read']['passed'] is True
    assert report['screening_directions']['main_utility'] == 'challenges'

    counts['RB1-02'] = {'piqa': (560, 1000), 'boolq': (500, 1000)}
    report = _report(counts)
    assert report['rules']['3_GR_residual_single_read']['RB1-02_vs_RB1-07']['percentage_points'] == 1
    assert report['screening_directions']['main_utility'] == 'supports'


def test_secondary_loop_contrasts_are_descriptive_and_t0_depth_is_not_claimed():
    report = _report(_counts())
    secondary = report['rules']['5_GL_loop']['secondary_descriptive_contrasts']
    assert secondary['RB1-06_minus_RB1-02']['value'] == 0
    assert secondary['RB1-04_minus_RB1-06']['value'] == 0
    assert any('same-depth T0' in note for note in report['notes'])


def test_bootstrap_uses_shared_task_indices_and_reports_paired_intervals():
    counts = _counts(default=(2, 4))
    rows = {
        label: {'piqa': [0, 1, 1, 0], 'boolq': [1, 1, 0, 0]}
        for label in ('Base', *rb1_rules.ARM_IDS)
    }
    report = _report(counts, bootstrap_rows=rows, bootstrap_repeats=30,
                     bootstrap_seed=20261101, chunk_size=7)
    paired = rb1_rules.paired_interval(report['bootstrap']['comparisons'], 'RB1-01', 'Base')
    assert paired['difference'] == 0
    assert paired['ci95'] == [0, 0]


def test_count_data_includes_supplied_loop_intervention_for_bootstrap():
    counts = _counts(default=(2, 4), **{
        'RB1-03/iv_window1': {'piqa': (2, 4), 'boolq': (2, 4)},
    })
    rows = {
        label: {'piqa': [0, 1, 1, 0], 'boolq': [1, 1, 0, 0]}
        for label in counts
    }
    report = _report(counts, bootstrap_rows=rows, bootstrap_repeats=4)
    assert 'RB1-03/iv_window1' in report['metrics']
    paired = rb1_rules.paired_interval(
        report['bootstrap']['comparisons'], 'RB1-03/iv_window1', 'RB1-04')
    assert paired['difference'] == 0


def test_rule9_triggers_when_arm_is_at_exact_rule10_upper_boundary():
    counts = _counts(**{
        'RB1-07': {'piqa': (520, 1000), 'boolq': (500, 1000)},
    })
    report = _report(counts)
    rule9 = report['rules']['9_no_residual_or_concat_signal']
    assert rule9['gate_condition_met'] is True
    assert report['rules']['10_all_arms_within_Base_plus_minus_1pp']['passed'] is False
    assert rule9['triggered'] is True
    assert report['progression'] == 'check_T0_and_gradient_then_at_most_one_probe'


def test_negative_counts_and_bootstrap_count_row_mismatch_raise():
    bad = _counts()
    bad['RB1-02']['piqa'] = (-1, 1000)
    with pytest.raises(rb1_rules.RulesError, match='invalid'):
        _report(bad)

    counts = _counts(default=(2, 4))
    rows = {
        label: {'piqa': [0, 1, 1, 0], 'boolq': [1, 1, 0, 0]}
        for label in counts
    }
    rows['RB1-02']['piqa'] = [1, 1, 1, 0]
    with pytest.raises(rb1_rules.RulesError, match='differs from supplied counts'):
        _report(counts, bootstrap_rows=rows, bootstrap_repeats=2)

