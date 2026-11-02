from fractions import Fraction

import pytest

from rb_improve_analysis import bind_reference, rule_outcomes, runtime_metrics
from rb1_artifacts import reference
from rb_improve_lr import endpoint
from rb_train import E0029_ARMS


def scores():
    return {arm: Fraction(7, 10) for arm in (*E0029_ARMS, 'RB1-09', 'RB1-07', 'Base', 'X-01')}


def test_implementation_failure_withholds_all_rule_results():
    with pytest.raises(ValueError, match='implementation'):
        rule_outcomes(scores(), False)


def test_pending_x01_preserves_independent_rules_and_defers_only_dependent_comparisons():
    incomplete = scores()
    del incomplete['X-01']
    incomplete['Q-N'] += Fraction(1, 100)
    result = rule_outcomes(incomplete)
    assert result['6_H0026']['outcome'] == 'supports'
    assert result['6_H0026']['d_V_status'] == 'pending_X01'
    assert result['6_H0026']['d_V_descriptive'] is None
    assert set(result['2_learnability']) == set(E0029_ARMS)
    assert set(result['7_LoRA']['differences']) == set(E0029_ARMS[:6])
    for rule, delta in [('3_H0023', 'd_up'), ('4_H0024', 'd_IG'), ('5_H0025', 'd_pair')]:
        assert result[rule]['outcome'] == 'pending_X01'
        assert result[rule]['pending_comparisons'] and result[rule][delta]
    assert result['3_H0023']['activate_H0027'] is False
    assert result['4_H0024']['activate_H0028'] is False


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


def test_analysis_rejects_retried_x01_manifest_and_stale_reference_checkpoint(tmp_path):
    manifest = tmp_path/'manifest.json'
    weights = tmp_path/'adapter.pt'
    manifest.write_text('{"arm":"X-01"}')
    weights.write_bytes(b'original checkpoint')
    inventory = {'manifest': reference(manifest), 'adapter_weights': reference(weights)}
    validated = {'sources': {'X-01': {'manifest': inventory['manifest']}}}
    diagnostic = {'training_inputs': inventory}
    bind_reference('X-01', inventory, xcheck=validated, diagnostic=diagnostic)
    manifest.write_text('{"arm":"X-01","retry":true}')
    retry = dict(inventory, manifest=reference(manifest))
    with pytest.raises(ValueError, match='run that passed'):
        bind_reference('X-01', retry, xcheck=validated)
    weights.write_bytes(b'replacement checkpoint')
    changed = dict(inventory, adapter_weights=reference(weights))
    with pytest.raises(ValueError, match='different checkpoint'):
        bind_reference('X-01', changed, xcheck=validated, diagnostic=diagnostic)


def test_runtime_report_preserves_contract_metrics_for_all_eight_arms():
    expected = {'train.steady_step_seconds': .23, 'train.peak_cuda_bytes': 2700000000,
        'params.stored': 794624, 'params.trainable': 794624, 'params.with_grad': 794624,
        'runtime.total_forward_input_tokens': 2000000,
        'runtime.accounting': {'states': {'6': {'M.shape': [1, 32, 32], 'dtype': 'torch.float32'}}}}
    manifests = {arm: {'metrics': dict(expected, **{'dev.piqa.accuracy': .7})} for arm in E0029_ARMS}
    report = runtime_metrics(manifests)
    assert set(report) == set(E0029_ARMS)
    assert all(value == expected for value in report.values())
