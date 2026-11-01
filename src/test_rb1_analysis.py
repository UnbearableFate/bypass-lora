"""Synthetic full-grid analysis and pairing/provenance failure paths."""
import copy
import json
from pathlib import Path

import pytest

import rb1_analysis as analysis
from rb1_artifacts import read, reference
from rb1_rules import ARM_IDS, TASKS
from test_rb1_latency import _formal_run


def predictions(correct):
    return [{'task': task, 'split': 'dev', 'id': i, 'gold': 0,
             'prediction': int(i >= correct), 'gold_nll_sum': 2., 'gold_loss_tokens': 1,
             'scores': [-2., -3. if i < correct else -1.]} for task in TASKS for i in range(1000)]


def metrics(rows):
    result = {}
    for task, value in analysis.summarize_rows(rows).items():
        if isinstance(value, dict):
            result.update({f'dev.{task}.{k}': v for k, v in value.items()})
    result['dev.nll'] = 2.
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def full_grid(tmp_path, monkeypatch):
    base = tmp_path/'Base'
    base.mkdir()
    base_rows = predictions(500)
    for name in ('manifest.json', 'config.json', 'predictions.json', 'split-map.json', 'data-metadata.json'):
        write(base/name, {'fixture': name})
    (base/'manifest.json').write_bytes(Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0024-1/manifest.json').read_bytes())
    monkeypatch.setattr(analysis, '_read_run', lambda *args: (base_rows, {'metrics': metrics(base_rows)}))
    successes = dict(zip(ARM_IDS, (580, 600, 605, 620, 602, 610, 610, 640, 635)))
    roots, manifests = {}, {}
    for arm in ARM_IDS:
        root = _formal_run(tmp_path/'training', arm)
        roots[arm] = str(root)
        manifest = read(root/'manifest.json')
        m = manifest['metrics']
        m['plan.order_sha256'] = 'a'*64
        layers = {str(i): {'writes': 0 if arm == 'RB1-01' else m['train.input_tokens'],
                           'reads': 0 if arm == 'RB1-01' else m['train.input_tokens']*(2 if arm == 'RB1-03' else 4 if arm in ('RB1-04', 'RB1-05', 'RB1-06') else 1),
                           'valid_tokens': m['train.input_tokens'], 'attention': m['train.examples'],
                           'ffn': m['train.examples']} for i in (6, 13, 20, 27)}
        m['runtime.accounting'] = {'layers': {} if arm == 'RB1-07' else layers,
                                    'states': {}, 'state_M_c_bytes': 0 if arm in ('RB1-01', 'RB1-07') else 16896}
        rows = predictions(successes[arm])
        m.update(metrics(rows))
        write(root/'predictions.json', rows)
        for step in (489, 977, 1466):
            write(root/f'dev-step-{step}.json', {'step': step, 'metrics': metrics(rows), 'predictions': rows})
        write(root/'manifest.json', manifest)
        manifests[arm] = reference(root/'manifest.json')
    interventions = {}
    for arm in ('RB1-01', 'RB1-02', 'RB1-04', 'RB1-08', 'RB1-09'):
        root = tmp_path/('IV-'+arm)
        interventions[arm] = str(root)
        normal = predictions(successes[arm])
        def condition(rows, mode='normal'):
            steps = (1, 4) if arm == 'RB1-04' else (1,)
            records = [] if arm == 'RB1-01' or mode in ('iv_off', 'Base_same_allocation') else [
                {'task': task, 'layer': layer, 'step': step, 'scope': scope,
                 'count': 1000, 'empty_reads': 0,
                 'means': {'current_weight': 0. if mode == 'iv_strict' else 1.,
                           'n_eff': 1., 'entropy': 0., 'relative_n_eff': 1., 'readable_positions': 1.}}
                for task in TASKS for layer in (6, 13, 20, 27) for step in steps
                for scope in ('all_valid_input', 'answer_prediction')]
            for record in records:
                record['nonempty_means'] = dict(record['means'])
            return {'predictions': rows, 'metrics': metrics(rows), 'read_diagnostics': records}
        conditions = {'normal': condition(normal), 'Base_same_allocation': condition(base_rows)}
        for mode in (('iv_swap',) if arm == 'RB1-01' else analysis.MODES):
            conditions[mode] = condition(normal if arm == 'RB1-01' else base_rows if mode == 'iv_off' else predictions(successes[arm]-20), mode)
        write(root/'conditions.json', conditions)
        write(root/'provenance.json', {'training_manifest': {k: manifests[arm][k] for k in ('path', 'sha256')}})
        write(root/'manifest.json', {'status': 'complete', 'validator': {'passed': True}, 'contract': 'E-0021@1',
                                     'arm': arm, 'stage': 'intervention', 'seed': 0})
    latency_root = tmp_path/'Latency'
    samples = [1.] * 128
    timed = ('RB1-01', 'RB1-02', 'RB1-04', 'RB1-08', 'RB1-07', 'RB1-07-merged')
    measurements = {arm: {'batch': 1, 'prefill_length': 512, 'prefill_passes': 5, 'decode_tokens': 128,
                           'prefill_seconds': [1.]*5, 'prefill_tokens_per_second': 512.,
                           'decode_samples_ms': samples, 'decode_p50_ms': 1., 'decode_p95_ms': 1.,
                           'peak_cuda_bytes': 10, 'adapter_persistent_state': {'M_c_bytes': 16896 if arm in ('RB1-02', 'RB1-04', 'RB1-08') else 0}}
                    for arm in timed}
    write(latency_root/'manifest.json', {'status': 'complete', 'validator': {'passed': True}, 'contract': 'E-0021@1'})
    write(latency_root/'mig-profile.json', {'passed': True, 'profile': '2g.24gb'})
    write(latency_root/'latency.json', {'measurements': measurements, 'loaded_refs': {
        arm: {'artifact_inventory': {'manifest': manifests[arm]}} for arm in timed if arm in ARM_IDS}})
    costs = tmp_path/'costs.json'
    attempts = [root.split('/')[-1] for root in [*roots.values(), *interventions.values(), str(latency_root)]]
    write(costs, [{'attempt_id': attempt, 'execution_state': 'ended', 'usage': {
        'basis': 'measured', 'accounting_ref': f'fixture:{attempt}', 'quantity': .01,
        'unit': 'mig_instance_hours', 'source_ref': 'synthetic scheduler fixture'}} for attempt in attempts])
    return {'expected': {'source_commit': analysis.ACCEPTED_TRAINING_SOURCE_COMMIT,
                         'lr_lock_sha256': analysis.ACCEPTED_LR_LOCK_SHA256,
                         'base_manifest_sha256': analysis.ACCEPTED_BASE_MANIFEST_SHA256},
            'training_runs': roots, 'base_run': str(base), 'intervention_runs': interventions,
            'latency_run': str(latency_root), 'observations': [str(costs)]}


@pytest.mark.parametrize('flipped', [False, True])
def test_full_synthetic_grid_has_exact_rules_interventions_and_locked_bootstrap(full_grid, flipped):
    if flipped:
        path = Path(full_grid['intervention_runs']['RB1-02'])/'conditions.json'
        conditions = read(path)
        row = conditions['normal']['predictions'][0]
        row.update(prediction=1, scores=[-2., -1.])
        conditions['normal']['metrics'] = metrics(conditions['normal']['predictions'])
        write(path, conditions)
    report = analysis.analyze(full_grid)
    assert report['selection']['S_F_star'] == 'RB1-04'
    assert report['selection']['C_star_RB'] == 'RB1-09'
    assert report['bootstrap']['repeats'] == 10000
    assert report['bootstrap']['seed'] == 20261101
    assert report['rules']['4_GH_side_history']['passed'] is True
    assert report['rules']['4_GH_side_history']['normal_reference'] == 'RB1-02/normal'
    assert report['interventions']['RB1-02']['iv_swap']['macro_accuracy']['difference'] == pytest.approx(-.0195 if flipped else -.02)
    agreement = report['interventions_normal_vs_training']['RB1-02']
    assert agreement['prediction_mismatches'] == int(flipped)
    assert agreement['max_abs_score_difference'] == (2 if flipped else 0)
    assert agreement['macro_accuracy_difference'] == pytest.approx(-.0005 if flipped else 0)
    assert len(report['training_runtime']) == 9
    assert report['cost']['node_h'] == pytest.approx(15*.01*.25)
    assert 'Single-seed dev screening' in analysis.markdown(report)


@pytest.mark.parametrize('pin', ['source_commit', 'lr_lock_sha256', 'base_manifest_sha256'])
def test_expected_release_rejects_nonadmitted_binding_before_reading_results(full_grid, pin):
    full_grid['expected'][pin] = '0'*(40 if pin == 'source_commit' else 64)
    with pytest.raises(ValueError, match='pre-result RB1 release pins'):
        analysis.analyze(full_grid, stage='select')


def test_source_binding_rejects_consistent_but_unreviewed_training(full_grid):
    for root in full_grid['training_runs'].values():
        path = Path(root)/'manifest.json'
        value = read(path)
        value['metrics']['source.commit'] = 'c'*40
        write(path, value)
    with pytest.raises(ValueError, match='covered RB1 training release'):
        analysis.analyze(full_grid, stage='select')


@pytest.mark.parametrize('failure,match', [('duplicate', 'duplicate'), ('order', 'paired'),
                                        ('gold', 'gold score'), ('score', 'tie rule'), ('partial', '1000-row')])
def test_prediction_boundary_rejects_invalid_pairing(failure, match):
    paired = predictions(600)
    rows = copy.deepcopy(paired)
    if failure == 'duplicate':
        rows[1] = rows[0]
    elif failure == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    elif failure == 'gold':
        rows[0]['gold'] = 1
    elif failure == 'score':
        rows[0]['prediction'] = 1
    else:
        rows.pop()
    with pytest.raises(ValueError, match=match):
        analysis.validate_rows(rows, 'fixture', paired)


def test_final_rejects_wrong_checkpoint_provenance_and_incomplete_interventions(full_grid):
    del full_grid['intervention_runs']['RB1-04']
    with pytest.raises(ValueError, match='four intervention targets'):
        analysis.analyze(full_grid)


def test_cost_is_deduplicated_by_scheduler_accounting_ref(full_grid):
    paths = full_grid['observations']
    once = analysis.measured_cost(paths)
    twice = analysis.measured_cost(paths+paths)
    assert once['node_h'] == twice['node_h']
    assert len(once['rows']) == len(twice['rows'])
