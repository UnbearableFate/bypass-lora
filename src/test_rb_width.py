"""E-0026 r128 width probe: arm registry, source binding and locked rule boundaries."""
import json

import pytest

from rb1_artifacts import validate_formal_run
from rb_train import FAMILIES, LORA_RANKS, PARAMETERS, adapter_config, contract
from rb_width_analysis import decide, evidence_line, implementation_checks
from rb_width_check import nearest_rank
from test_rb1_latency import _formal_run


def test_width_arms_change_only_rank_and_keep_rb1_families():
    for width, narrow in (('W-01', 'RB1-01'), ('W-02', 'RB1-02')):
        wide, base = adapter_config(width).__dict__, adapter_config(narrow).__dict__
        assert wide['rank'] == 128 and base['rank'] == 32
        assert {k: v for k, v in wide.items() if k != 'rank'} == {k: v for k, v in base.items() if k != 'rank'}
        assert FAMILIES[width] == FAMILIES[narrow]
    assert FAMILIES['W-03'] == FAMILIES['RB1-07'] and LORA_RANKS == {'RB1-07': 7, 'W-03': 31}
    with pytest.raises(ValueError):
        adapter_config('W-03')
    # Contract estimate: 4 layers x (5*1024*r + 2r^2) and 90,112 LoRA parameters per rank.
    assert PARAMETERS['W-01'] == PARAMETERS['W-02'] == 4*(5*1024*128 + 2*128**2)
    assert PARAMETERS['W-03'] == 31*90112 and PARAMETERS['RB1-07'] == 7*90112
    assert {contract(a, s) for a in ('W-01', 'W-02', 'W-03') for s in ('smoke', 'train')} == {'E-0026@1'}
    assert (contract('RB1-02', 'smoke'), contract('RB1-02', 'train')) == ('E-0019@1', 'E-0021@1')


def test_nearest_rank_uses_implementation_counts_and_breaks_ties_low():
    counts = {r: r*90112 for r in range(28, 35)}
    assert nearest_rank(PARAMETERS['W-02'], counts) == 31
    assert nearest_rank(150, {1: 100, 2: 200}) == 1


def _width_run(tmp_path, arm, source):
    run = _formal_run(tmp_path, arm)
    path = run/'manifest.json'
    value = json.loads(path.read_text())
    value['metrics']['source.commit'] = source
    path.write_text(json.dumps(value))
    return run


@pytest.mark.parametrize('arm', ['W-01', 'W-02', 'W-03'])
def test_width_runs_bind_to_the_named_reviewed_source(tmp_path, arm):
    source = 'a'*40
    run = _width_run(tmp_path, arm, source)
    manifest, _, adapter = validate_formal_run(run, arm, source)
    assert manifest['contract'] == 'E-0026@1' and adapter == run/'adapter'
    with pytest.raises(ValueError, match='reviewed training source'):
        validate_formal_run(run, arm)
    with pytest.raises(ValueError, match='source commit differs'):
        validate_formal_run(run, arm, 'b'*40)


def test_rb1_arms_stay_pinned_to_the_e0021_source(tmp_path):
    run = _formal_run(tmp_path, 'RB1-02')
    validate_formal_run(run, 'RB1-02')
    with pytest.raises(ValueError, match='pinned'):
        validate_formal_run(run, 'RB1-02', 'a'*40)


def _rows(piqa, boolq):
    """1000 rows per task with the given numbers correct (gold 1)."""
    return ([{'task': 'piqa', 'id': i, 'gold': 1, 'prediction': int(i < piqa)} for i in range(1000)] +
            [{'task': 'boolq', 'id': i, 'gold': 1, 'prediction': int(i < boolq)} for i in range(1000)])


def _labels(**macro_counts):
    """Each value is (piqa_correct, boolq_correct)."""
    defaults = {'Base': (700, 700), 'W-01': (750, 750), 'W-02': (750, 750), 'W-03': (750, 750),
                'RB1-01': (740, 740), 'RB1-02': (740, 740), 'RB1-07': (760, 760),
                'W-02/normal': (750, 750), 'W-02/iv_off': (700, 700), 'W-02/iv_swap': (740, 740),
                'W-02/iv_permute': (600, 600)}
    defaults.update(macro_counts)
    return {label: _rows(*value) for label, value in defaults.items()}


def test_rule3_boundaries_are_inclusive_on_exact_fractions():
    # W-02 exactly Base+1 pp and exactly 1 pp below W-03: supports.
    rows = _labels(Base=(700, 700), **{'W-02': (710, 710), 'W-03': (720, 720)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule3_H0018']['outcome'] == 'supports' and rules['rule8_no_further_widening'] is False
    # One row lower on one task: macro is Base+0.95 pp, challenges.
    rows = _labels(Base=(700, 700), **{'W-02': (709, 710), 'W-03': (719, 720)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule3_H0018']['outcome'] == 'challenges' and rules['rule8_no_further_widening'] is True
    # d_L one row below -1 pp with W-02 effective: challenges.
    rows = _labels(**{'W-02': (750, 750), 'W-03': (760, 761)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule3_H0018']['outcome'] == 'challenges'


def test_rule4_needs_both_tasks_within_minus_one():
    rows = _labels(**{'W-02': (780, 740), 'W-03': (740, 755)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule4_practical_advantage']['d_L']['percentage_points'] == pytest.approx(1.25)
    assert rules['rule4_practical_advantage']['recorded'] is False
    rows = _labels(**{'W-02': (770, 750), 'W-03': (750, 750)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule4_practical_advantage']['recorded'] is True


def test_rule5_reopen_uses_intervention_normal_and_both_thresholds():
    rows = _labels(**{'W-02': (760, 760), 'W-01': (750, 750), 'W-02/normal': (760, 760), 'W-02/iv_swap': (750, 750)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule5_history']['reopen_H0014'] is True
    rows = _labels(**{'W-02': (760, 760), 'W-01': (750, 750), 'W-02/normal': (760, 760), 'W-02/iv_swap': (751, 750)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule5_history']['reopen_H0014'] is False and rules['rule5_history']['descriptive_only'] is True
    rows = _labels(**{'W-02': (750, 750), 'W-01': (750, 750)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule5_history']['no_measurable_history_contribution'] is True


def test_failed_implementation_withholds_all_rule_outcomes():
    rows = _labels(**{'W-02': (800, 800)})
    rules, _ = decide(rows, rows['W-02/normal'], False)
    assert rules['rule3_H0018']['outcome'] is None
    assert rules['rule4_practical_advantage']['recorded'] is None
    assert rules['rule5_history']['reopen_H0014'] is None and rules['rule8_no_further_widening'] is None
    assert rules['rule3_H0018']['d_L']['percentage_points'] == pytest.approx(5.0)


def test_lora_learnability_sets_main_comparison_evidence_level_at_inclusive_boundary():
    # W-03 one row below Base+1 pp: diagnostic, but the comparison is still computed.
    rows = _labels(Base=(700, 700), **{'W-03': (709, 710), 'W-02': (750, 750)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule2_learnability']['lora_control_not_learned'] is True
    assert rules['rule3_H0018']['outcome'] == 'supports'
    assert {rules[k]['evidence_level'] for k in ('rule3_H0018', 'rule4_practical_advantage', 'rule5_history')} == {'diagnostic'}
    assert rules['evidence_level'] == 'diagnostic' and 'diagnostic' in evidence_line(rules)
    # W-03 exactly Base+1 pp: screening.
    rows = _labels(Base=(700, 700), **{'W-03': (710, 710), 'W-02': (750, 750)})
    rules, _ = decide(rows, rows['W-02/normal'], True)
    assert rules['rule2_learnability']['lora_control_not_learned'] is False
    assert {rules[k]['evidence_level'] for k in ('evidence_level',)} | {
        rules[k]['evidence_level'] for k in ('rule3_H0018', 'rule4_practical_advantage', 'rule5_history')} == {'screening'}
    assert evidence_line(rules) == 'Evidence level for rules 3-5: screening.'


def _implementation_inputs(shared_off_change=False):
    base = [dict(r, scores=[0., 0.]) for r in _rows(700, 700)]
    off = [dict(r) for r in base]
    same_allocation = [dict(r) for r in base]
    if shared_off_change:
        for rows in (off, same_allocation):
            rows[0]['prediction'] = 1 - rows[0]['prediction']
    registration = {'passed': True, 'lora_rank_nearest_W-02': 31,
                    'arms': {arm: {'trainable': PARAMETERS[arm]} for arm in ('W-01', 'W-02', 'W-03')}}
    training = {arm: {'metrics': {'reload.identical': True, 'params.stored': PARAMETERS[arm],
                                  'params.trainable': PARAMETERS[arm], 'params.with_grad': PARAMETERS[arm]}}
                for arm in ('W-01', 'W-02', 'W-03')}
    init = {arm: [dict(r) for r in base] for arm in ('W-01', 'W-02', 'W-03')}
    conditions = {'iv_off': {'predictions': off}, 'Base_same_allocation': {'predictions': same_allocation}}
    return {'Base': base}, init, conditions, registration, training


def test_rule1a_requires_iv_off_to_match_pinned_Base_not_only_same_allocation():
    result = implementation_checks(*_implementation_inputs())
    assert result['passed'] is True and result['checks']['a_iv_off_equals_Base'] is True
    result = implementation_checks(*_implementation_inputs(shared_off_change=True))
    assert result['iv_off_rows_identical_to_same_allocation_Base'] is True
    assert result['checks']['a_iv_off_equals_Base'] is False and result['passed'] is False
    rows = _labels(**{'W-02': (800, 800)})
    rules, _ = decide(rows, rows['W-02/normal'], result['passed'])
    assert rules['evidence_level'] is None and 'withheld' in evidence_line(rules)
    assert (rules['rule3_H0018']['outcome'], rules['rule4_practical_advantage']['recorded'],
            rules['rule5_history']['reopen_H0014'], rules['rule8_no_further_widening']) == (None,)*4
