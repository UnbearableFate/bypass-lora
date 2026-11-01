"""PBS-only regressions for E-0010 rules, pairing, bootstrap and formal inputs."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

import v2_factorial_analysis as a


def decision_arms():
    return {arm: dict(controls={m: dict(count=1000, correct_count=200, em=.2, nll=3.)
                                      for m in a.CONTROLS},
                     target_weight=.0625, n_eff=16., normalized_read_error=1.,
                     trainable_parameters=100) for arm in a.ARM_SPECS}


def set_normal(arms, arm, count, *, nll=2., weight=.125, n_eff=8., params=100):
    arms[arm]['controls']['normal'].update(correct_count=count, em=count/1000, nll=nll)
    arms[arm].update(target_weight=weight, n_eff=n_eff, trainable_parameters=params)


def test_h3_inclusive_em_and_weight_boundaries_with_strict_nll():
    arms = decision_arms()
    set_normal(arms, 'V2-01', 300)
    d = a.locked_decisions(arms)
    assert d['H-0003']['status'] == 'supported'
    assert d['H-0003']['inputs']['V2-01']['em_margin_fraction'] == '1/10'
    assert d['H-0003']['eligible'] == ['V2-01']
    arms['V2-01']['controls']['normal']['nll'] = 3.
    assert a.locked_decisions(arms)['H-0003']['status'] == 'challenged'


def test_h3_requires_both_negative_controls_and_distinguishes_degenerate_reads():
    arms = decision_arms()
    set_normal(arms, 'V2-01', 300, weight=.124999999)
    assert a.locked_decisions(arms)['H-0003']['status'] == 'inconclusive'
    arms['V2-01']['controls']['shuffled'].update(correct_count=201, em=.201)
    assert a.locked_decisions(arms)['H-0003']['status'] == 'challenged'
    assert a.locked_decisions(arms)['V3']['selected_arm'] is None


def test_dual_probe_can_satisfy_h3_but_never_enters_h4_cells():
    arms = decision_arms()
    set_normal(arms, 'V2-10', 1000, weight=1., n_eff=1.)
    d = a.locked_decisions(arms)
    assert d['H-0003']['eligible'] == ['V2-10']
    assert d['V3']['selected_arm'] == 'V2-10'
    assert d['H-0004']['status'] == 'inconclusive'
    assert len(d['H-0004']['cells']) == 4


def test_h4_soft_requires_all_three_conditions_and_three_cells():
    arms = decision_arms()
    for arm in ('V2-03', 'V2-04', 'V2-07'):
        set_normal(arms, arm, 201)
    d = a.locked_decisions(arms)['H-0004']
    assert (d['status'], d['soft_wins']) == ('supported', 3)
    arms['V2-07']['n_eff'] = 16.
    assert a.locked_decisions(arms)['H-0004']['status'] == 'inconclusive'
    arms['V2-07']['n_eff'] = 8.
    arms['V2-07']['target_weight'] = .0625
    assert a.locked_decisions(arms)['H-0004']['soft_wins'] == 2


def test_h4_kernel_requires_only_higher_em_and_ties_never_count():
    arms = decision_arms()
    for arm in ('V2-01', 'V2-02', 'V2-05'):
        set_normal(arms, arm, 201, weight=.01, n_eff=16.)
    d = a.locked_decisions(arms)['H-0004']
    assert (d['status'], d['kernel_wins']) == ('challenged', 3)
    assert d['cells'][-1]['classification'] == 'inconclusive'


def test_v3_strict_one_pp_window_then_parameters_then_dev_nll():
    arms = decision_arms()
    set_normal(arms, 'V2-01', 400, params=1000)
    set_normal(arms, 'V2-02', 391, params=900, nll=2.1)
    set_normal(arms, 'V2-03', 390, params=1, nll=1.)  # exactly 1pp: excluded
    set_normal(arms, 'V2-04', 392, params=900, nll=2.)
    d = a.locked_decisions(arms)['V3']
    assert d['near_best'] == ['V2-01', 'V2-02', 'V2-04']
    assert d['selected_arm'] == 'V2-04'


def test_v3_does_not_invent_a_fourth_tiebreaker():
    arms = decision_arms()
    for arm in ('V2-01', 'V2-02'):
        set_normal(arms, arm, 400)
    d = a.locked_decisions(arms)['V3']
    assert d['selected_arm'] is None and d['unresolved_exact_tie']
    assert d['finalists'] == ['V2-01', 'V2-02']


def test_decisions_refuse_incomplete_dev():
    arms = decision_arms()
    arms['V2-01']['controls']['normal']['count'] = 999
    with pytest.raises(a.AnalysisError, match='full dev'):
        a.locked_decisions(arms)


def test_bootstrap_matches_explicit_episode_draws_and_pairs_every_column():
    values = np.array([[0., 0., 4.], [1., 1., 2.], [0., 0., 1.], [1., 1., 3.]])
    indices = np.random.default_rng(17).integers(0, 4, size=(57, 4))
    expected = values[indices].mean(1)
    actual = a.paired_bootstrap(values, repeats=57, seed=17, chunk_size=7)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
    np.testing.assert_allclose(a.paired_bootstrap(values, repeats=57, seed=17, chunk_size=57), actual,
                               rtol=0, atol=1e-15)
    assert np.var(actual[:, 0]) > 0


def test_factorial_effect_normalization_and_probe_exclusion():
    weights = a.effect_weights()
    responses = {}
    for arm, (tap, reader, fusion) in a.ARM_SPECS.items():
        x, y, z = (1 if tap == 'qknorm_pre_rope' else -1,
                   1 if reader == 'soft_slot' else -1, 1 if fusion == 'C_up' else -1)
        responses[arm] = .5 + .05*x + .02*y + .01*x*y + .03*x*y*z
    got = {name: sum(w*responses[arm] for arm, w in spec.items()) for name, spec in weights.items()}
    assert got['feature'] == pytest.approx(.1)
    assert got['reader'] == pytest.approx(.04)
    assert got['feature x reader'] == pytest.approx(.02)
    assert got['feature x reader x fusion'] == pytest.approx(.06)
    assert got['fusion'] == pytest.approx(0.)
    assert len(weights) == 7
    assert all('V2-09' not in spec and 'V2-10' not in spec for spec in weights.values())


def _rows(split, n):
    return [dict(episode_id=hashlib.sha256(f'{split}-{i}'.encode()).hexdigest(),
                 target=1, prediction=1 if i % 4 == 0 else 2, correct=i % 4 == 0, nll=2.,
                 answer_ne_subject=i % 7 != 0, relation='emails' if i % 2 else 'rescues',
                 target_fact_position=i % 16, support_order='natural' if i % 2 else 'shuffled')
            for i in range(n)]


def _summary(rows, mode):
    summary = {k: v for k, v in a.point(rows).items() if k != 'correct_count'}
    summary['answer_ne_subject'] = a.point([r for r in rows if r['answer_ne_subject']])
    summary['strata'] = {axis: {name: a.point([r for r in rows if str(r[axis]) == name])
                                for name in {str(r[axis]) for r in rows}}
                         for axis in ('relation', 'target_fact_position', 'support_order')}
    summary['diagnostics'] = {'address.target_available': mode != 'shuffled', 'address.n_eff': 16.,
                              'read.normalized_error': 1.}
    if mode != 'shuffled':
        summary['diagnostics']['address.target_weight'] = .0625
    summary['prior_em'] = .02
    return summary


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


@pytest.fixture
def formal(tmp_path):
    """Synthetic full-size formal artifact layout; never opens a data split."""
    path = tmp_path/'ATT-test'
    arm_name = 'direct_hidden.kernel_matrix.F_out'
    directory = path/arm_name
    manifest = dict(status='complete', validator={'passed': True}, model='Qwen/Qwen3-0.6B-Base',
        revision=a.REVISION, seed=0, batch_size=8, evaluation_batch_size=16, token_budget=4000000,
        smoke_steps=0, smoke_eval_limit=0, microbatch_size=8, data_sha256=a.DATA_HASHES,
        torch='2.10.0+cu128', cuda='12.8')
    cfg = dict(layers=[14], rank=128, branch='memory', fusion='F_out', tap='direct_hidden',
        backend='kernel_matrix', steps=1, mode='inclusive', update='accumulate', readout='increment',
        eta=.25, read_gain=1., eps_norm=1e-6, eps_read=1e-6, write_scope='visible_fact_delimiter',
        slot_capacity=16, model_revision=a.REVISION)
    train = dict(steps=6000, tokens=4000470, stored_parameters=100, trainable_parameters=100,
                 parameters_with_nonzero_gradient=100, training_seconds=1200., stable_step_seconds=.2,
                 tokens_per_second=3300., peak_allocated_bytes=1500000000, gradient_norm_mean=1.,
                 gradient_norm_max=2., clip_fraction=.2, learning_curves={})
    result = dict(config=cfg, lr=.01, initialization_max_abs=0., schedule=a.SCHEDULE,
                  train=train, evaluation={}, checkpoint_reload={}, wall_seconds=1500.)
    for split, count in a.COUNTS.items():
        rows = _rows(split, count)
        result['evaluation'][split] = {m: _summary(rows, m) for m in a.CONTROLS}
        for mode in a.CONTROLS:
            _jsonl(directory/f'{split}.{mode}.predictions.jsonl', rows)
        _json(directory/f'{split}.metrics.json', result['evaluation'][split])
        if split == 'dev':
            _jsonl(directory/'reload/dev.normal.predictions.jsonl', rows)
            result['checkpoint_reload'] = dict(predictions_identical=True, dev_normal=_summary(rows, 'normal'))
            for milestone in ('25', '50', '75'):
                curves = {m: _summary(rows, m) for m in ('normal', 'zero', 'shuffled')}
                train['learning_curves'][milestone] = curves
                _json(directory/f'curve-{milestone}/dev.metrics.json', curves)
                for mode in curves:
                    _jsonl(directory/f'curve-{milestone}/dev.{mode}.predictions.jsonl', rows)
    _json(path/'manifest.json', manifest)
    _json(path/'schedule.json', a.SCHEDULE)
    _jsonl(directory/'train.jsonl', [dict(step=i, tokens=round(i*4000470/6000), loss=2.,
                                        lr=.01 if i < 6000 else 0., gradient_norm=1., seconds=.2)
                                   for i in range(1, 6001)])
    _save_result(path, result)
    return path, directory, result


def _save_result(path, result):
    name = 'direct_hidden.kernel_matrix.F_out'
    _json(path/'metrics.json', {name: result})
    _json(path/name/'metrics.json', result)


def test_formal_loader_accepts_full_outputs_and_preserves_provenance(formal):
    path, _, _ = formal
    loaded = a.load_run(path, 'V2-01')
    assert len(loaded['records']['dev']['normal']) == 1000
    assert len(loaded['records']['calibration']['normal']) == 2048
    assert len(loaded['source']['sha256']) >= 30


@pytest.mark.parametrize('field,value,match', [
    ('lr', .003, 'released LR'),
    ('schedule', dict(a.SCHEDULE, tokens=200170), 'wrong schedule'),
    ('initialization_max_abs', .01, 'initialization'),
])
def test_formal_loader_rejects_wrong_release_settings(formal, field, value, match):
    path, _, result = formal
    result[field] = value
    _save_result(path, result)
    with pytest.raises(a.AnalysisError, match=match):
        a.load_run(path, 'V2-01')


def test_formal_loader_rejects_nonfinite_episode_nll(formal):
    path, directory, _ = formal
    rows = _rows('dev', 1000)
    rows[0]['nll'] = float('nan')
    _jsonl(directory/'dev.normal.predictions.jsonl', rows)
    with pytest.raises(a.AnalysisError, match='NLL'):
        a.load_run(path, 'V2-01')


def test_formal_loader_rejects_reordered_control_even_with_same_summary(formal):
    path, directory, _ = formal
    _jsonl(directory/'dev.zero.predictions.jsonl', list(reversed(_rows('dev', 1000))))
    with pytest.raises(a.AnalysisError, match='pairing mismatch'):
        a.load_run(path, 'V2-01')


def test_formal_loader_rejects_reported_reload_success_with_changed_predictions(formal):
    path, directory, _ = formal
    rows = _rows('dev', 1000)
    rows[1]['prediction'] = 3  # remains incorrect, so all summaries still match
    _jsonl(directory/'reload/dev.normal.predictions.jsonl', rows)
    with pytest.raises(a.AnalysisError, match='reload predictions mismatch'):
        a.load_run(path, 'V2-01')


def test_formal_loader_rejects_forged_summary(formal):
    path, directory, result = formal
    result['evaluation']['dev']['normal']['em'] = .9
    _save_result(path, result)
    _json(directory/'dev.metrics.json', result['evaluation']['dev'])
    with pytest.raises(a.AnalysisError, match='summary em mismatch'):
        a.load_run(path, 'V2-01')


def test_cross_arm_pairing_checks_metadata_and_split_disjointness():
    loaded = {arm: {'records': {s: {'normal': _rows(s, 5)} for s in a.COUNTS}}
              for arm in a.ARM_SPECS}
    a.check_alignment(loaded)
    bad = deepcopy(loaded)
    bad['V2-02']['records']['dev']['normal'][0]['target'] = 9
    with pytest.raises(a.AnalysisError, match='cross-arm'):
        a.check_alignment(bad)
    for run in loaded.values():
        run['records']['calibration']['normal'] = run['records']['dev']['normal']
    with pytest.raises(a.AnalysisError, match='overlap'):
        a.check_alignment(loaded)


def test_formal_analysis_requires_exact_arm_set_and_distinct_paths(tmp_path):
    with pytest.raises(a.AnalysisError, match='exactly'):
        a.analyze({'V2-01': tmp_path})
    with pytest.raises(a.AnalysisError, match='duplicate run'):
        a.analyze({arm: tmp_path for arm in a.ARM_SPECS})


def test_descriptive_paired_intervals_and_dual_comparisons():
    loaded = {arm: {'records': {'dev': {mode: _rows('dev', 32) for mode in a.CONTROLS}}}
              for arm in a.ARM_SPECS}
    report = a.descriptive(loaded, 'dev', repeats=40)
    assert report['normal_minus_control']['V2-01']['zero']['em'] == {'estimate': 0., 'ci95': [0., 0.]}
    assert report['C_dual_minus_comparator']['V2-09 - V2-06']['nll']['ci95'] == [0., 0.]
    assert report['factorial']['feature x reader']['normal']['em']['estimate'] == pytest.approx(0.)
    assert report['strata']['V2-01']['normal']['answer_ne_subject']['count'] == 27
