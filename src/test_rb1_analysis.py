"""Synthetic full-grid analysis and pairing/provenance failure paths."""
import copy
import json

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
    monkeypatch.setattr(analysis, '_read_run', lambda *args: (base_rows, {'metrics': metrics(base_rows)}))
    lock = tmp_path/'lock.json'
    split = tmp_path/'split.json'
    write(split, {'fixture': 'split'})
    write(lock, {'experiment': 'E-0020@1', 'locked_lrs': {f: .001 for f in ('RBc-R', 'RBc-T', 'RBc-L', 'RBc-C')},
                 'split_manifest_sha256': reference(split)['sha256']})
    successes = dict(zip(ARM_IDS, (580, 600, 605, 620, 602, 610, 610, 640, 635)))
    roots, manifests = {}, {}
    for arm in ARM_IDS:
        root = _formal_run(tmp_path/'training', arm)
        roots[arm] = str(root)
        config = read(root/'config.json')
        config.update(lr_lock=str(lock), split_manifest=str(split))
        write(root/'config.json', config)
        manifest = read(root/'manifest.json')
        m = manifest['metrics']
        m.update({'lr_lock.sha256': reference(lock)['sha256'],
                  'split_manifest.sha256': reference(split)['sha256'], 'plan.order_sha256': 'a'*64})
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
    return {'training_runs': roots, 'base_run': str(base), 'intervention_runs': interventions,
            'latency_run': str(latency_root), 'observations': [str(costs)]}


def test_full_synthetic_grid_has_exact_rules_interventions_and_locked_bootstrap(full_grid):
    report = analysis.analyze(full_grid)
    assert report['selection']['S_F_star'] == 'RB1-04'
    assert report['selection']['C_star_RB'] == 'RB1-09'
    assert report['bootstrap']['repeats'] == 10000
    assert report['bootstrap']['seed'] == 20261101
    assert report['rules']['4_GH_side_history']['passed'] is True
    assert report['rules']['4_GH_side_history']['normal_reference'] == 'RB1-02/normal'
    assert report['interventions']['RB1-02']['iv_swap']['macro_accuracy']['difference'] == pytest.approx(-.02)
    assert len(report['training_runtime']) == 9
    assert report['cost']['node_h'] == pytest.approx(15*.01*.25)
    assert 'Single-seed dev screening' in analysis.markdown(report)


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
