from fractions import Fraction as F
import pytest

from rb_tap_analysis import rule_outcomes


def scores(**updates):
    values={'Base':F(7,10),'RB1-07':F(78,100),'RB1-09':F(775,1000),
            'X-01':F(785,1000),'X-02':F(775,1000),'X-03':F(765,1000),
            'ETA-X-1':F(795,1000),'ETA-QK-1':F(785,1000)}
    values.update(updates);return values


def test_exact_inclusive_one_pp_tap_eta_and_history_boundaries():
    result=rule_outcomes(scores(),{'X':'ETA-X-1','QK':'ETA-QK-1'},{'X':F(1,100),'QK':F(195,10000)})
    assert result['3_H0021']['outcome']=='supports'
    assert result['6_H0022']['outcome']=='supports'
    assert all(v['outcome']=='supports' for v in result['6_H0022']['per_tap'].values())
    assert all(v['ask_coordinator_about_H0014'] for v in result['4_history'].values())


def test_one_row_below_support_threshold_is_descriptive():
    result=rule_outcomes(scores(**{'X-01':F(1569,2000)}),{'X':'X-01','QK':'RB1-09'},
                         {'X':F(1,100),'QK':F(1,100)})
    assert result['3_H0021']['outcome']=='descriptive'
    assert result['4_history']['X']['ask_coordinator_about_H0014'] is False
    assert result['6_H0022']['outcome']=='challenges'
    assert all(v['d_eta']['numerator']==0 for v in result['6_H0022']['per_tap'].values())


def test_eta_at_least_one_support_rule_and_inconclusive_middle():
    selected={'X':'ETA-X-1','QK':'ETA-QK-1'}
    drops={'X':F(0),'QK':F(0)}
    result=rule_outcomes(scores(**{'ETA-QK-1':F(0)}),selected,drops)
    assert result['6_H0022']['outcome']=='supports'
    result=rule_outcomes(scores(**{'ETA-X-1':F(79,100),'ETA-QK-1':F(0)}),selected,drops)
    assert result['6_H0022']['outcome']=='inconclusive'


def test_stop_rule_keeps_strict_lora_boundary_and_tap_conjunction():
    selected={'X':'X-01','QK':'RB1-09'};drops={'X':F(0),'QK':F(0)}
    # X remains above QK, and exactly LoRA-1pp is not strictly below it.
    result=rule_outcomes(scores(**{'X-01':F(77,100),'RB1-09':F(76,100)}),selected,drops)
    assert result['10_no_more_ordinary_tap_variants']['triggered'] is False
    result=rule_outcomes(scores(**{'X-01':F(1539,2000),'RB1-09':F(76,100)}),selected,drops)
    assert result['10_no_more_ordinary_tap_variants']['triggered'] is True
    result=rule_outcomes(scores(**{'X-01':F(775,1000)}),selected,drops)
    assert result['10_no_more_ordinary_tap_variants']['triggered'] is True


def test_invalid_implementation_cannot_produce_scientific_outcome():
    with pytest.raises(ValueError,match='withheld'):
        rule_outcomes(scores(),{'X':'X-01','QK':'RB1-09'},{'X':F(0),'QK':F(0)},False)
