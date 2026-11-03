"""Exact boundary and split-stratified pairing checks for E-0031 rules."""
from fractions import Fraction as F

import pytest

from rb_qvu_analysis import paired_intervals, rules


def scores(h=F(81,100), t=F(79,100), x=F(80,100), l=F(81,100)):
    merged = {'H-QVU':h, 'T0-QVU':t, 'X-01':x, 'RB1-07':l}
    return merged, dict(merged, Base=F(70,100))


def test_inclusive_exact_practical_and_history_gates():
    merged, dev = scores(t=F(80,100))
    value = rules(merged,dev,F(1,100))
    assert value['3_H0030']['outcome'] == 'supports'
    assert value['4_H0031']['outcome'] == 'supports'
    assert value['6_conditional_controls']['triggered']
    assert value['7_replication']['propose_two_new_seeds']


@pytest.mark.parametrize('h,t,swap,h30,h31', [
    (F(80,100),F(80,100),F(2,100),'challenges','challenges'),
    (F(805,1000),F(79,100),F(2,100),'inconclusive','supports'),
    (F(81,100),F(805,1000),F(2,100),'supports','inconclusive'),
    (F(81,100),F(79,100),F(999,100000),'supports','inconclusive')])
def test_unrounded_boundaries_and_swap_cannot_replace_history(h,t,swap,h30,h31):
    merged,dev = scores(h=h,t=t)
    value = rules(merged,dev,swap)
    assert value['3_H0030']['outcome'] == h30
    assert value['4_H0031']['outcome'] == h31


def test_lora_floor_and_tokenwise_practical_trigger_are_independent():
    merged,dev = scores(h=F(81,100),t=F(82,100),l=F(82,100))
    value = rules(merged,dev,F(2,100))
    assert value['3_H0030']['outcome'] == 'inconclusive'
    assert value['4_H0031']['outcome'] == 'challenges'
    assert value['5_T0_practical']['passes']
    assert value['6_conditional_controls']['triggered']
    assert not value['10_route']['coordinator_stop_review_required']


def test_no_practical_signal_triggers_only_coordinator_stop_review():
    merged,dev = scores(h=F(80,100),t=F(80,100))
    value = rules(merged,dev,F(0))
    assert not value['6_conditional_controls']['triggered']
    assert value['10_route']['coordinator_stop_review_required']


def rows(offset=0):
    return [dict(task=t,split=s,id=offset+100*i+j,gold=0,prediction=0,gold_loss_tokens=1)
        for i,s in enumerate(('dev','calibration')) for t in ('piqa','boolq') for j in range(3+i)]


def test_common_stratified_indices_preserve_exact_paired_differences():
    left = rows()
    right = [dict(r,prediction=1) for r in left]
    result = paired_intervals({'left':left,'right':right},repeats=217)
    assert result['left_minus_right']['ci95_pp'] == pytest.approx([100,100])
    assert result['left_minus_right']['estimate_pp'] == 100
    assert result == paired_intervals({'right':right,'left':left},repeats=217)


@pytest.mark.parametrize('mutation', ['gold','id','split','gold_loss_tokens','order'])
def test_pairing_rejects_consequential_membership_or_label_changes(mutation):
    left = rows(); right = [dict(r) for r in left]
    if mutation == 'order':
        right.reverse()
    else:
        right[0][mutation] = 'calibration' if mutation == 'split' else 2
    with pytest.raises(ValueError,match='paired rows'):
        paired_intervals({'left':left,'right':right},repeats=2)
