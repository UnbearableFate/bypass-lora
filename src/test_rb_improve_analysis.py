from fractions import Fraction

import pytest

from rb_improve_analysis import rule_outcomes
from rb_improve_lr import endpoint
from rb_train import E0029_ARMS


def scores():
    return {arm: Fraction(7, 10) for arm in (*E0029_ARMS, 'RB1-09', 'RB1-07', 'Base', 'X-01')}


def test_implementation_failure_or_pending_x01_withholds_all_rule_results():
    with pytest.raises(ValueError, match='implementation'):
        rule_outcomes(scores(), False)
    incomplete = scores()
    del incomplete['X-01']
    with pytest.raises(ValueError, match='pending'):
        rule_outcomes(incomplete)


@pytest.mark.parametrize('arm', ['L-56', 'L-48'])
def test_H0023_includes_exact_one_pp_boundary_and_activates_H0027(arm):
    values = scores()
    values[arm] += Fraction(1, 100)
    result = rule_outcomes(values)['3_H0023']
    assert result['outcome'] == 'challenges' and result['activate_H0027']
    values[arm] -= Fraction(1, 10000)
    assert rule_outcomes(values)['3_H0023']['outcome'] == 'supports'


@pytest.mark.parametrize('arm,rule', [('G-T', '4_H0024'), ('G-QN', '4_H0024'),
    ('G-N', '4_H0024'), ('P-T', '5_H0025'), ('P-N', '5_H0025'), ('Q-N', '6_H0026')])
def test_gain_rules_include_one_pp_and_zero_and_preserve_inconclusive(arm, rule):
    values = scores()
    assert rule_outcomes(values)[rule]['outcome'] == 'challenges'
    values[arm] += Fraction(1, 200)
    assert rule_outcomes(values)[rule]['outcome'] == 'inconclusive'
    values[arm] += Fraction(1, 200)
    assert rule_outcomes(values)[rule]['outcome'] == 'supports'


def test_IG_win_over_all7_lora_requires_GA_control_in_replication():
    values = scores()
    values['G-N'] += Fraction(1, 100)
    result = rule_outcomes(values)
    assert result['7_LoRA']['new_two_seed_replication'] == ['G-N']
    assert result['7_LoRA']['LoRA_GA_required_in_replication'] == ['G-N']
    assert result['8_evidence']['hypothesis_status_changes'] is False


def test_up_lora_lr_endpoint_extension_is_once_in_the_correct_direction():
    assert endpoint({1e-4: 1., 3e-4: 1.1, 1e-3: 1.2}) == 3e-5
    assert endpoint({1e-4: 1.2, 3e-4: 1.1, 1e-3: 1.}) == 3e-3
    assert endpoint({1e-4: 1.1, 3e-4: 1., 1e-3: 1.2}) is None
    with pytest.raises(ValueError):
        endpoint({1e-4: 1.})
