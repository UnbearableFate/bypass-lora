"""CPU tests for E-0003 run pairing and bootstrap analysis."""
import json

import pytest

import v1_analysis


def _write_run(path, *, boolq_id='b0', reverse=False, correct=True,
               reload_identical=True, diverged=False):
    path.mkdir()
    rows = [
        {'task': 'piqa', 'split': 'dev', 'id': 'p0', 'gold': 0,
         'prediction': 0 if correct else 1, 'scores': [-1.0, -2.0],
         'gold_nll_sum': 2.0, 'gold_loss_tokens': 2},
        {'task': 'piqa', 'split': 'dev', 'id': 'p1', 'gold': 1,
         'prediction': 1, 'scores': [-2.0, -1.0],
         'gold_nll_sum': 3.0, 'gold_loss_tokens': 3},
        {'task': 'boolq', 'split': 'dev', 'id': boolq_id, 'gold': 1,
         'prediction': 1, 'scores': [-2.0, -1.0],
         'gold_nll_sum': 4.0, 'gold_loss_tokens': 1},
    ]
    if reverse:
        rows.reverse()
    (path / 'predictions.json').write_text(json.dumps(rows))
    (path / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'validator': {'passed': True},
        'metrics': {'reload.identical': reload_identical, 'train.diverged': diverged,
                    'dev.piqa.accuracy': .5},
    }))


def _input_runs(tmp_path, **kwargs):
    run_paths = {}
    for run_id in v1_analysis.ARM_IDS:
        path = tmp_path / run_id
        _write_run(path, **kwargs)
        run_paths[run_id] = path
    base = tmp_path / 'base'
    _write_run(base, **kwargs)
    return run_paths, base


def test_analysis_uses_identical_task_stratified_row_draws_for_every_arm(tmp_path):
    run_paths, base = _input_runs(tmp_path)
    report = v1_analysis.analyze(run_paths, base, repeats=40, seed=13, chunk_size=7)

    assert report['bootstrap']['seed'] == 13
    assert report['arms']['V1-01']['metrics']['macro_accuracy'] == 1.0
    # Identical row outcomes imply every paired replicate difference is exactly zero.
    diff = report['comparisons']['V1-01 - Base']['macro_accuracy']
    assert diff['difference'] == 0.0
    assert diff['ci95'] == [0.0, 0.0]


@pytest.mark.parametrize('kwargs', [
    {'boolq_id': 'different'},
    {'reverse': True},
])
def test_analysis_rejects_dev_id_gold_or_order_mismatch(tmp_path, kwargs):
    run_paths, base = _input_runs(tmp_path)
    # Keep the first run as reference and alter one comparator.
    changed = tmp_path / 'changed'
    _write_run(changed, **kwargs)
    run_paths['V1-02'] = changed

    with pytest.raises(v1_analysis.AnalysisError, match='rows or order differ'):
        v1_analysis.analyze(run_paths, base, repeats=5)


def test_analysis_rejects_nonfinite_dev_candidate_score(tmp_path):
    run_paths, base = _input_runs(tmp_path)
    broken = tmp_path / 'broken'
    _write_run(broken)
    rows = json.loads((broken / 'predictions.json').read_text())
    rows[0]['scores'][0] = float('nan')
    (broken / 'predictions.json').write_text(json.dumps(rows))
    run_paths['V1-01'] = broken

    with pytest.raises(v1_analysis.AnalysisError, match='finite'):
        v1_analysis.analyze(run_paths, base, repeats=5)


def test_c_star_is_none_when_no_concat_arm_is_effective(tmp_path):
    run_paths, base = _input_runs(tmp_path)
    report = v1_analysis.analyze(run_paths, base, repeats=5)

    assert report['selection']['C_star'] is None
    assert report['selection']['C_effective'] == []
    assert report['selection']['concat_wins'] is False


def _many_runs(tmp_path, counts, nll=None):
    """Write matched 100-row/task fixtures; counts maps IDs to (PIQA, BoolQ)."""
    nll = nll or {}
    all_counts = {arm: counts.get(arm, (50, 50)) for arm in (*v1_analysis.ARM_IDS, 'Base')}
    paths = {}
    for arm, task_counts in all_counts.items():
        path = tmp_path / f'many-{arm}'
        path.mkdir(parents=True)
        rows = []
        for task, correct_count in zip(v1_analysis.TASKS, task_counts):
            loss = nll.get(arm, {}).get(task, 1.0)
            for index in range(100):
                gold = index % 2
                prediction = gold if index < correct_count else 1 - gold
                rows.append({'task': task, 'split': 'dev', 'id': f'{task}-{index}',
                             'gold': gold, 'prediction': prediction,
                             'scores': [-1.0, -2.0], 'gold_nll_sum': loss,
                             'gold_loss_tokens': 1})
        (path / 'predictions.json').write_text(json.dumps(rows))
        (path / 'manifest.json').write_text(json.dumps({
            'status': 'complete', 'validator': {'passed': True},
            'metrics': {'reload.identical': True, 'dev.piqa.accuracy': .5},
        }))
        paths[arm] = path
    return {arm: paths[arm] for arm in v1_analysis.ARM_IDS}, paths['Base']


def test_effectiveness_threshold_is_inclusive_at_exactly_one_percentage_point(tmp_path):
    counts = {'V1-03': (52, 50)}  # macro 0.51 vs Base 0.50
    run_paths, base = _many_runs(tmp_path, counts)
    selection = v1_analysis.analyze(run_paths, base, repeats=2)['selection']

    assert selection['C_effective'] == ['V1-03']
    assert selection['C_star'] == 'V1-03'


def test_one_task_half_point_gain_is_below_effectiveness_threshold(tmp_path):
    counts = {'V1-03': (51, 50)}  # macro +0.005
    run_paths, base = _many_runs(tmp_path, counts)
    selection = v1_analysis.analyze(run_paths, base, repeats=2)['selection']

    assert selection['C_effective'] == []
    assert selection['C_star'] is None


def test_c_star_prefers_fewer_parameters_among_strictly_near_best_effective_arms(tmp_path):
    counts = {'V1-03': (53, 50), 'V1-04': (52, 50), 'V1-05': (54, 50)}
    run_paths, base = _many_runs(tmp_path, counts)
    selection = v1_analysis.analyze(run_paths, base, repeats=2)['selection']

    # Gate is 0.005 below the dual best and has fewer parameters; up is exactly
    # 0.01 below best and is excluded by the strict near-best boundary.
    assert selection['C_effective_near_best_within_1pp'] == ['V1-03', 'V1-05']
    assert selection['C_simplest_effective_near_best'] == ['V1-03']
    assert selection['C_star'] == 'V1-03'


def test_gate_up_ties_use_macro_accuracy_then_nll_then_gate(tmp_path):
    higher = {'V1-03': (52, 50), 'V1-04': (53, 50)}
    run_paths, base = _many_runs(tmp_path / 'higher-macro', higher)
    assert v1_analysis.analyze(run_paths, base, repeats=2)['selection']['C_star'] == 'V1-04'

    counts = {'V1-03': (53, 50), 'V1-04': (53, 50)}
    nll = {'V1-03': {'piqa': 1.2}, 'V1-04': {'piqa': 1.0}}
    run_paths, base = _many_runs(tmp_path / 'nll', counts, nll)
    assert v1_analysis.analyze(run_paths, base, repeats=2)['selection']['C_star'] == 'V1-04'

    tied = tmp_path / 'equal-nll'
    run_paths, base = _many_runs(tied, counts)
    assert v1_analysis.analyze(run_paths, base, repeats=2)['selection']['C_star'] == 'V1-03'


def test_concat_win_and_dual_decisions_use_inclusive_exact_one_point_boundaries(tmp_path):
    counts = {'V1-01': (50, 50), 'V1-02': (50, 50),
              'V1-03': (50, 50), 'V1-04': (50, 50), 'V1-05': (52, 50)}
    run_paths, base = _many_runs(tmp_path, counts)
    selection = v1_analysis.analyze(run_paths, base, repeats=2)['selection']
    assert selection['C_star'] == 'V1-05'
    assert selection['concat_wins'] is True
    assert selection['C_dual_better_than_C_gate'] is True


def test_concat_cannot_win_without_c_star_even_when_raw_concat_score_is_high(tmp_path):
    counts = {'V1-01': (50, 50), 'V1-02': (50, 50),
              'V1-03': (50, 50), 'V1-04': (50, 50), 'V1-05': (52, 50)}
    # Raising Base above all concat arms removes C* while the best raw concat is
    # still above F*. For this fixture Base can be raised only by changing both
    # task counts, so construct it explicitly while retaining the aligned rows.
    all_counts = dict(counts, Base=(55, 55))
    run_paths, base = _many_runs(tmp_path / 'high-base', all_counts)
    selection = v1_analysis.analyze(run_paths, base, repeats=2)['selection']
    assert selection['C_star'] is None
    assert selection['concat_wins'] is False


def test_f_star_uses_exact_one_point_boundary(tmp_path):
    # F_pre exactly 1 pp above F_out wins; a strict sub-1pp advantage is treated
    # as near-equal and selects F_out.
    counts = {'V1-01': (50, 50), 'V1-02': (52, 50)}
    run_paths, base = _many_runs(tmp_path, counts)
    selection = v1_analysis.analyze(run_paths, base, repeats=2)['selection']
    assert selection['F_star']['selected'] == 'V1-02'

    close = tmp_path / 'close'
    counts = {'V1-01': (50, 50), 'V1-02': (51, 50)}
    run_paths, base = _many_runs(close, counts)
    selection = v1_analysis.analyze(run_paths, base, repeats=2)['selection']
    assert selection['F_star']['selected'] == 'V1-01'


@pytest.mark.parametrize('reload_identical,diverged', [(False, False), (True, True)])
def test_analysis_rejects_unverified_or_diverged_run(tmp_path, reload_identical, diverged):
    run_paths, base = _input_runs(tmp_path)
    bad = tmp_path / 'bad-run'
    _write_run(bad, reload_identical=reload_identical, diverged=diverged)
    run_paths['V1-01'] = bad
    with pytest.raises(v1_analysis.AnalysisError):
        v1_analysis.analyze(run_paths, base, repeats=2)
