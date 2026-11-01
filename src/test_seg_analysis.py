from copy import deepcopy

import pytest

import seg_analysis as analysis
from rb1_artifacts import reference
from rb_history import donor_indices
from seg_artifacts import validate
from test_seg_artifacts import formal, predictions, write, SOURCE


@pytest.fixture
def grid(tmp_path, monkeypatch):
    roots, manifests, refs = {}, {}, {}
    for i in range(5):
        arm = f'SEG-{i:02}'; root = tmp_path/arm
        formal(root, arm, monkeypatch)
        manifests[arm], refs[arm], _ = validate(root, arm, SOURCE)
        roots[arm] = str(root)
    check = tmp_path/'checks.json'
    write(check, {'status': 'complete', 'metrics': {'no_channel_identity': True}, 'validator': {'passed': True}})
    import rb1_artifacts
    monkeypatch.setattr(rb1_artifacts, 'validate_formal_run', lambda *args: ({'metrics': {}}, {}, None))
    monkeypatch.setattr(analysis, 'validate', lambda root, arm, source: (manifests[arm], refs[arm], None))
    full = tmp_path/'RB1-07'; write(full/'predictions.json', predictions(700))
    iv = tmp_path/'iv'
    rows = predictions()
    mapping = donor_indices(rows, 'R1')
    donors = [{'task': r['task'], 'recipient': r['id'], 'donor': rows[mapping[i]]['id']} for i, r in enumerate(rows)]
    conditions = {kind: {'predictions': rows, 'audit': {'donors': donors if kind == 'iv_swap' else []}}
                  for kind in ('Base_seg', 'normal', 'iv_off', 'iv_swap', 'iv_reset')}
    write(iv/'conditions.json', conditions)
    write(iv/'provenance.json', {'training': refs['SEG-02'], 'checkpoint_files': analysis.read(tmp_path/'SEG-02'/'checkpoint.json'),
                                'training_source': SOURCE, 'evaluation_source': SOURCE})
    write(iv/'manifest.json', {'status': 'complete', 'contract': 'E-0025@1', 'stage': 'intervention', 'arm': 'SEG-02',
                              'metrics': {'off.identical_to_Base_seg': True}, 'validator': {'passed': True}})
    return {'runs': roots, 'training_source': SOURCE, 'evaluation_source': SOURCE,
        'full_context_lora_run': str(full), 'deterministic_checks': {'manifest_ref': reference(check)},
        'interventions': {'SEG-02': str(iv)}, 'observations': []}


def test_full_synthetic_analysis_runs_locked_bootstrap_and_rules(grid):
    report = analysis.analyze(grid)
    assert report['bootstrap']['repeats'] == 10000 and report['bootstrap']['seed'] == 20261101
    assert report['rules']['1']['passed'] is True and report['primary'] == 'piqa'
    assert report['direction'] == 'challenges'
    assert 'Single-seed dev screening' in analysis.markdown(report)


@pytest.mark.parametrize('failure', ['missing', 'donor', 'checkpoint', 'initial', 'off'])
def test_analysis_rejects_incomplete_grid_and_wrong_donor_checkpoint_identity(grid, failure):
    root = grid['interventions']['SEG-02']
    if failure == 'missing': grid['interventions'] = {}
    elif failure == 'checkpoint':
        p = analysis.read(root+'/provenance.json'); p['checkpoint_files']['adapter.pt']['sha256'] = 'f'*64
        write(__import__('pathlib').Path(root)/'provenance.json', p)
    elif failure == 'initial':
        ref = grid['deterministic_checks']['manifest_ref']; ref['sha256'] = 'a'*64
    else:
        p = analysis.read(root+'/conditions.json')
        if failure == 'donor': p['iv_swap']['audit']['donors'][0]['donor'] = p['iv_swap']['audit']['donors'][0]['recipient']
        else:
            p['iv_off']['predictions'][0]['prediction'] = 1
            p['iv_off']['predictions'][0]['scores'] = [-2., -1.]
        write(__import__('pathlib').Path(root)/'conditions.json', p)
    with pytest.raises(ValueError): analysis.analyze(grid)
