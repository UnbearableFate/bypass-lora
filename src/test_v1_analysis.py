"""CPU tests for E-0003 run pairing and bootstrap analysis."""
import json

import pytest

import v1_analysis


def _write_run(path, *, boolq_id='b0', reverse=False, correct=True):
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
        'metrics': {'reload.identical': True, 'dev.piqa.accuracy': .5},
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


def test_c_star_stays_unresolved_without_an_effectiveness_rule(tmp_path):
    run_paths, base = _input_runs(tmp_path)
    report = v1_analysis.analyze(run_paths, base, repeats=5)

    assert report['selection']['C_star'] is None
    assert report['selection']['C_star_status'] == 'coordinator_definition_needed'
    assert report['selection']['C_simplest_near_best'] == ['V1-03', 'V1-04']
