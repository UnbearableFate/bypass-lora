import json

import pytest

from v2r16_analysis import (AnalysisError, d1_gate, h3_gate, summarize_curve,
                            validate_manifest, descriptive_bootstrap)


def _point(correct, count=1000, nll=2.0):
    return {'count': count, 'correct_count': correct, 'em': correct/count, 'nll': nll}


def test_d1_exact_thresholds_and_plateau_condition():
    passed = d1_gate({'tf_normal': _point(700), 'tf_shuffled': _point(200), 'zero': _point(200)}, False)
    partial = d1_gate({'tf_normal': _point(400), 'tf_shuffled': _point(200), 'zero': _point(200)}, True)
    failed = d1_gate({'tf_normal': _point(399), 'tf_shuffled': _point(200), 'zero': _point(200)}, True)
    inconclusive = d1_gate({'tf_normal': _point(399), 'tf_shuffled': _point(200), 'zero': _point(200)}, False)
    assert passed['status'] == 'pass' and passed['margin_fraction'] == '1/2'
    assert partial['status'] == 'partial' and partial['margin_fraction'] == '1/5'
    assert failed['status'] == 'fail'
    assert inconclusive['status'] == 'inconclusive'


def test_h3_exact_margin_nll_and_all_vs_last_weight_choice():
    controls = {'normal': _point(300, nll=1.0),
                'zero': _point(200, nll=2.0),
                'shuffled': _point(100, nll=3.0)}
    standard = h3_gate(controls, all_question_target_weight=0.125,
                       last_position_target_weight=0.0)
    hard = h3_gate(controls, all_question_target_weight=0.0,
                   last_position_target_weight=0.125, weight_scope='last_position')
    assert standard['passed'] and standard['em_margin_fraction'] == '1/10'
    assert standard['target_weight_scope'] == 'all_question_positions'
    assert hard['passed'] and hard['target_weight_scope'] == 'last_position'
    controls['normal'] = _point(300, nll=2.0)
    assert not h3_gate(controls, 0.125)['passed']  # NLL comparison is strict.


def _curve(slope):
    rows = []
    for step in range(1, 6001):
        if step <= 300:
            lr = 0.003 * step/300
        elif step <= 5000:
            lr = 0.003
        else:
            lr = 0.003 * 0.08 * (6000-step)/999
        rows.append({'step': step, 'loss': 5.0 + slope*(step-5001), 'lr': lr})
    return rows


def test_curve_ols_plateau_uses_fitted_decline_and_preserves_negative_declines():
    flatish = summarize_curve(_curve(-0.00005))
    declining = summarize_curve(_curve(-0.0001))
    rising = summarize_curve(_curve(0.0001))
    assert flatish['plateau'] and flatish['fitted_decline_999_steps'] < 0.05
    assert not declining['plateau'] and declining['fitted_decline_999_steps'] > 0.05
    assert rising['plateau'] and rising['fitted_decline_999_steps'] < 0.0
    assert [bucket['steps'] for bucket in flatish['per_500_step_mean_answer_ce']][-1] == [5501, 6000]
    assert flatish['final_1000_lr_max'] < flatish['peak_lr']
    assert flatish['final_1000_max_lr_fraction_of_peak'] == pytest.approx(0.08)


def test_manifest_rejects_smoke_and_incomplete_runs():
    manifest = {'status': 'complete', 'validator': {'passed': True},
                'experiment': 'E-0016', 'contract': 'E-0016@1', 'arm': 'D1',
                'eligible_as_evidence': True, 'smoke_steps': 0, 'smoke_eval_limit': 0,
                'metrics': {'train': {'steps': 6000},
                            'checkpoint_reload': {'full_records_identical': True,
                                                  'summaries_identical': True}}}
    validate_manifest(manifest, 'D1')
    smoke = dict(manifest, smoke_steps=10)
    with pytest.raises(AnalysisError, match='smoke'):
        validate_manifest(smoke, 'D1')
    incomplete = dict(manifest, status='failed')
    with pytest.raises(AnalysisError, match='complete'):
        validate_manifest(incomplete, 'D1')
    missing_contract = dict(manifest)
    missing_contract.pop('contract')
    with pytest.raises(AnalysisError, match='contract'):
        validate_manifest(missing_contract, 'D1')


def test_descriptive_bootstrap_reuses_episode_indices_across_controls():
    records = [{'correct': i % 2 == 0, 'nll': float(i+1)} for i in range(8)]
    result = descriptive_bootstrap({'normal': records, 'zero': list(records), 'shuffled': list(records)})
    assert result['normal-zero']['em']['ci95'] == [0., 0.]
    assert result['normal-shuffled']['nll']['ci95'] == [0., 0.]
