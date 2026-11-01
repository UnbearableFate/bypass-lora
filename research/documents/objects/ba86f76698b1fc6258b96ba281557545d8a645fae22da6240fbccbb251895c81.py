"""CPU tests for E-0003 run pairing and bootstrap analysis."""
import json
from pathlib import Path
import shutil

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


def _analyze_numeric(run_paths, base, **kwargs):
    """Exercise selection/bootstrap math without treating tiny rows as formal input."""
    paths = {arm: Path(path) for arm, path in run_paths.items()}
    paths['Base'] = Path(base)
    loaded = {}
    for arm, path in paths.items():
        loaded[arm] = (json.loads((path / 'predictions.json').read_text()),
                       json.loads((path / 'manifest.json').read_text()))
    return v1_analysis._analyze_loaded(loaded, paths, **kwargs)


LOCKED_ARTIFACT = Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0022-1/F_out')


def _formal_runs(tmp_path):
    """Build full-size formal evidence from the saved locked split artifacts."""
    split_map = json.loads((LOCKED_ARTIFACT / 'split-map.json').read_text())
    run_paths = {}
    all_ids = (*v1_analysis.ARM_IDS, 'Base')
    for run_id in all_ids:
        path = tmp_path / run_id
        path.mkdir(parents=True)
        shutil.copyfile(LOCKED_ARTIFACT / 'split-map.json', path / 'split-map.json')
        shutil.copyfile(LOCKED_ARTIFACT / 'data-metadata.json', path / 'data-metadata.json')
        actual_arm = 'base' if run_id == 'Base' else v1_analysis.CONFIG_ARMS[run_id]
        (path / 'config.json').write_text(json.dumps({
            'arm': actual_arm, 'lr': .0003, 'tokens': 2000000, 'batch': 2,
            'eval_limit': None, 'smoke_steps': None, 'e3': True,
            'data_dir': '/locked/test/data', 'output': str(path),
        }))
        metrics = {
            'plan.steps': v1_analysis.FULL_STEPS,
            'plan.input_tokens': v1_analysis.FULL_INPUT_TOKENS,
            'plan.order_sha256': v1_analysis.TRAIN_ORDER_SHA256,
            'reload.identical': True, 'train.diverged': False,
        }
        if run_id == 'Base':
            metrics.update({'params.trainable': 0, 'params.adapter_stored': 0,
                            'params.with_grad': 0})
        else:
            expected = v1_analysis.TRAINABLE_PARAMS[run_id]
            metrics.update({'train.steps': v1_analysis.FULL_STEPS,
                            'train.input_tokens': v1_analysis.FULL_INPUT_TOKENS,
                            'train.examples': v1_analysis.FULL_EXAMPLES,
                            'params.trainable': expected, 'params.adapter_stored': expected,
                            'params.with_grad': expected})
        (path / 'manifest.json').write_text(json.dumps({
            'status': 'complete', 'validator': {'passed': True}, 'metrics': metrics,
        }))
        predictions = []
        for task in v1_analysis.TASKS:
            for index, row_id in enumerate(split_map[task]['dev']):
                gold = index % 2
                predictions.append({'task': task, 'split': 'dev', 'id': row_id,
                                    'gold': gold, 'prediction': gold,
                                    'scores': [-1.0, -2.0], 'gold_nll_sum': 1.0,
                                    'gold_loss_tokens': 1})
        (path / 'predictions.json').write_text(json.dumps(predictions))
        if run_id == 'Base':
            base_path = path
        else:
            run_paths[run_id] = path
    return run_paths, base_path


def test_analysis_uses_identical_task_stratified_row_draws_for_every_arm(tmp_path):
    run_paths, base = _input_runs(tmp_path)
    report = _analyze_numeric(run_paths, base, repeats=40, seed=13, chunk_size=7)

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
        _analyze_numeric(run_paths, base, repeats=5)


def test_analysis_rejects_nonfinite_dev_candidate_score(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    broken = run_paths['V1-01']
    rows = json.loads((broken / 'predictions.json').read_text())
    rows[0]['scores'][0] = float('nan')
    (broken / 'predictions.json').write_text(json.dumps(rows))

    with pytest.raises(v1_analysis.AnalysisError, match='finite'):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_c_star_is_none_when_no_concat_arm_is_effective(tmp_path):
    run_paths, base = _input_runs(tmp_path)
    report = _analyze_numeric(run_paths, base, repeats=5)

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
    selection = _analyze_numeric(run_paths, base, repeats=2)['selection']

    assert selection['C_effective'] == ['V1-03']
    assert selection['C_star'] == 'V1-03'


def test_one_task_half_point_gain_is_below_effectiveness_threshold(tmp_path):
    counts = {'V1-03': (51, 50)}  # macro +0.005
    run_paths, base = _many_runs(tmp_path, counts)
    selection = _analyze_numeric(run_paths, base, repeats=2)['selection']

    assert selection['C_effective'] == []
    assert selection['C_star'] is None


def test_c_star_prefers_fewer_parameters_among_strictly_near_best_effective_arms(tmp_path):
    counts = {'V1-03': (53, 50), 'V1-04': (52, 50), 'V1-05': (54, 50)}
    run_paths, base = _many_runs(tmp_path, counts)
    selection = _analyze_numeric(run_paths, base, repeats=2)['selection']

    # Gate is 0.005 below the dual best and has fewer parameters; up is exactly
    # 0.01 below best and is excluded by the strict near-best boundary.
    assert selection['C_effective_near_best_within_1pp'] == ['V1-03', 'V1-05']
    assert selection['C_simplest_effective_near_best'] == ['V1-03']
    assert selection['C_star'] == 'V1-03'


def test_gate_up_ties_use_macro_accuracy_then_nll_then_gate(tmp_path):
    higher = {'V1-03': (52, 50), 'V1-04': (53, 50)}
    run_paths, base = _many_runs(tmp_path / 'higher-macro', higher)
    assert _analyze_numeric(run_paths, base, repeats=2)['selection']['C_star'] == 'V1-04'

    counts = {'V1-03': (53, 50), 'V1-04': (53, 50)}
    nll = {'V1-03': {'piqa': 1.2}, 'V1-04': {'piqa': 1.0}}
    run_paths, base = _many_runs(tmp_path / 'nll', counts, nll)
    assert _analyze_numeric(run_paths, base, repeats=2)['selection']['C_star'] == 'V1-04'

    tied = tmp_path / 'equal-nll'
    run_paths, base = _many_runs(tied, counts)
    assert _analyze_numeric(run_paths, base, repeats=2)['selection']['C_star'] == 'V1-03'


def test_concat_win_and_dual_decisions_use_inclusive_exact_one_point_boundaries(tmp_path):
    counts = {'V1-01': (50, 50), 'V1-02': (50, 50),
              'V1-03': (50, 50), 'V1-04': (50, 50), 'V1-05': (52, 50)}
    run_paths, base = _many_runs(tmp_path, counts)
    selection = _analyze_numeric(run_paths, base, repeats=2)['selection']
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
    selection = _analyze_numeric(run_paths, base, repeats=2)['selection']
    assert selection['C_star'] is None
    assert selection['concat_wins'] is False


def test_f_star_uses_exact_one_point_boundary(tmp_path):
    # F_pre exactly 1 pp above F_out wins; a strict sub-1pp advantage is treated
    # as near-equal and selects F_out.
    counts = {'V1-01': (50, 50), 'V1-02': (52, 50)}
    run_paths, base = _many_runs(tmp_path, counts)
    selection = _analyze_numeric(run_paths, base, repeats=2)['selection']
    assert selection['F_star']['selected'] == 'V1-02'

    close = tmp_path / 'close'
    counts = {'V1-01': (50, 50), 'V1-02': (51, 50)}
    run_paths, base = _many_runs(close, counts)
    selection = _analyze_numeric(run_paths, base, repeats=2)['selection']
    assert selection['F_star']['selected'] == 'V1-01'


@pytest.mark.parametrize('reload_identical,diverged', [(False, False), (True, True)])
def test_analysis_rejects_unverified_or_diverged_run(tmp_path, reload_identical, diverged):
    run_paths, base = _formal_runs(tmp_path)
    manifest_path = run_paths['V1-01'] / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['metrics']['reload.identical'] = reload_identical
    manifest['metrics']['train.diverged'] = diverged
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(v1_analysis.AnalysisError):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_formal_analysis_accepts_correctly_mapped_full_fixture(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    report = v1_analysis.analyze(run_paths, base, repeats=1, chunk_size=1)

    assert report['arms']['V1-01']['dev_examples'] == {'piqa': 1000, 'boolq': 1000}
    assert report['arms']['V1-07']['manifest_metrics']['train.steps'] == v1_analysis.FULL_STEPS


def test_formal_analysis_rejects_aliased_resolved_run_paths(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    alias = tmp_path / 'alias'
    alias.symlink_to(run_paths['V1-01'], target_is_directory=True)
    run_paths['V1-02'] = alias

    with pytest.raises(v1_analysis.AnalysisError, match='unique resolved'):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_review_reproducer_same_f_out_path_for_all_labels_and_base_is_rejected(tmp_path):
    smoke = LOCKED_ARTIFACT
    run_paths = {run_id: smoke for run_id in v1_analysis.ARM_IDS}

    with pytest.raises(v1_analysis.AnalysisError, match='unique resolved'):
        v1_analysis.analyze(run_paths, smoke, repeats=1)


def test_distinct_att0022_smoke_paths_are_rejected_by_actual_smoke_config(tmp_path):
    root = LOCKED_ARTIFACT.parent
    run_paths = {run_id: root / actual_arm
                 for run_id, actual_arm in v1_analysis.CONFIG_ARMS.items()}
    _, base = _formal_runs(tmp_path)

    with pytest.raises(v1_analysis.AnalysisError, match='smoke_steps and eval_limit'):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_formal_analysis_rejects_wrong_actual_arm_and_missing_config(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    config_path = run_paths['V1-01'] / 'config.json'
    config = json.loads(config_path.read_text())
    config['arm'] = 'F_pre'
    config_path.write_text(json.dumps(config))
    with pytest.raises(v1_analysis.AnalysisError, match='config arm'):
        v1_analysis.analyze(run_paths, base, repeats=1)

    run_paths, base = _formal_runs(tmp_path / 'missing')
    (run_paths['V1-01'] / 'config.json').unlink()
    with pytest.raises(v1_analysis.AnalysisError, match='config.json'):
        v1_analysis.analyze(run_paths, base, repeats=1)


@pytest.mark.parametrize(('key', 'value'), [('smoke_steps', 20), ('eval_limit', 32)])
def test_formal_analysis_rejects_smoke_or_limited_evaluation(tmp_path, key, value):
    run_paths, base = _formal_runs(tmp_path)
    config_path = run_paths['V1-01'] / 'config.json'
    config = json.loads(config_path.read_text())
    config[key] = value
    config_path.write_text(json.dumps(config))

    with pytest.raises(v1_analysis.AnalysisError, match='smoke_steps and eval_limit'):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_formal_analysis_rejects_short_actual_training_with_full_plan(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    manifest_path = run_paths['V1-01'] / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['metrics']['train.steps'] = 20
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(v1_analysis.AnalysisError, match='train.steps'):
        v1_analysis.analyze(run_paths, base, repeats=1)


@pytest.mark.parametrize('artifact', ['order', 'split_hash', 'source_hash'])
def test_formal_analysis_rejects_locked_order_or_data_mismatch(tmp_path, artifact):
    run_paths, base = _formal_runs(tmp_path)
    path = run_paths['V1-01']
    if artifact == 'order':
        file = path / 'manifest.json'
        content = json.loads(file.read_text())
        content['metrics']['plan.order_sha256'] = 'wrong-order'
    elif artifact == 'split_hash':
        file = path / 'data-metadata.json'
        content = json.loads(file.read_text())
        content['split_mapping_sha256'] = 'wrong-split'
    else:
        file = path / 'data-metadata.json'
        content = json.loads(file.read_text())
        content['source_files']['piqa']['sha256'] = 'wrong-source'
    file.write_text(json.dumps(content))

    with pytest.raises(v1_analysis.AnalysisError):
        v1_analysis.analyze(run_paths, base, repeats=1)


@pytest.mark.parametrize('metric', ['params.adapter_stored', 'params.trainable', 'params.with_grad'])
def test_formal_analysis_rejects_unexpected_parameter_counts(tmp_path, metric):
    run_paths, base = _formal_runs(tmp_path)
    manifest_path = run_paths['V1-08'] / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['metrics'][metric] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(v1_analysis.AnalysisError, match=metric):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_formal_analysis_rejects_1000_dev_rows_with_wrong_membership(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    predictions_path = run_paths['V1-04'] / 'predictions.json'
    predictions = json.loads(predictions_path.read_text())
    predictions[0]['id'] = -1
    predictions_path.write_text(json.dumps(predictions))

    with pytest.raises(v1_analysis.AnalysisError, match='IDs/membership/order'):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_formal_analysis_rejects_fewer_than_1000_dev_rows_per_task(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    predictions_path = run_paths['V1-04'] / 'predictions.json'
    predictions = json.loads(predictions_path.read_text())
    predictions.pop(0)
    predictions_path.write_text(json.dumps(predictions))

    with pytest.raises(v1_analysis.AnalysisError, match='IDs/membership/order'):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_formal_analysis_rejects_base_training_consumption(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    manifest_path = base / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['metrics']['train.steps'] = 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(v1_analysis.AnalysisError, match='Base: unexpected training'):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_formal_analysis_requires_base_e3_configuration(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    config_path = base / 'config.json'
    config = json.loads(config_path.read_text())
    config['e3'] = False
    config_path.write_text(json.dumps(config))

    with pytest.raises(v1_analysis.AnalysisError, match='Base: config must have e3=true'):
        v1_analysis.analyze(run_paths, base, repeats=1)


def test_formal_analysis_rejects_smoke_base_configuration(tmp_path):
    run_paths, base = _formal_runs(tmp_path)
    config_path = base / 'config.json'
    config = json.loads(config_path.read_text())
    config['smoke_steps'] = 20
    config_path.write_text(json.dumps(config))

    with pytest.raises(v1_analysis.AnalysisError, match='Base: smoke_steps and eval_limit'):
        v1_analysis.analyze(run_paths, base, repeats=1)
