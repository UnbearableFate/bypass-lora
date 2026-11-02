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
        arm = f'SEG-{i:02}'; root = tmp_path/f'ATT-{600+i:04}-1'
        formal(root, arm, monkeypatch)
        manifests[arm], refs[arm], _ = validate(root, arm, SOURCE)
        roots[arm] = str(root)
    check = tmp_path/'ATT-0606-1'/'manifest.json'
    write(check, {'status': 'complete', 'source_commit': SOURCE,
                  'metrics': {'no_channel_identity': True}, 'validator': {'passed': True}})
    import rb1_artifacts
    monkeypatch.setattr(rb1_artifacts, 'validate_formal_run', lambda *args: ({'metrics': {}}, {}, None))
    monkeypatch.setattr(analysis, 'validate', lambda root, arm, source: (manifests[arm], refs[arm], None))
    full = tmp_path/'RB1-07'; write(full/'predictions.json', predictions(700))
    iv = tmp_path/'ATT-0605-1'
    rows = predictions()
    mapping = donor_indices(rows, 'R1')
    donors = [{'task': r['task'], 'recipient': r['id'], 'donor': rows[mapping[i]]['id']} for i, r in enumerate(rows)]
    conditions = {kind: {'predictions': rows, 'audit': {'donors': donors if kind == 'iv_swap' else []}}
                  for kind in ('Base_seg', 'normal', 'iv_off', 'iv_swap', 'iv_reset')}
    write(iv/'conditions.json', conditions)
    write(iv/'provenance.json', {'training': refs['SEG-02'], 'checkpoint_files': analysis.read(__import__('pathlib').Path(roots['SEG-02'])/'checkpoint.json'),
                                'training_source': SOURCE, 'evaluation_source': SOURCE})
    write(iv/'manifest.json', {'status': 'complete', 'contract': 'E-0025@1', 'stage': 'intervention', 'arm': 'SEG-02',
                              'metrics': {'off.identical_to_Base_seg': True}, 'validator': {'passed': True}})
    attempts = [f'ATT-{600+i:04}-1' for i in range(8)]+['ATT-0271-1']
    observations = tmp_path/'terminal-costs.json'
    write(observations, [{'attempt_id': attempt, 'execution_state': 'ended',
        'outcome': 'failed' if attempt == 'ATT-0607-1' else 'success',
        'usage': {'accounting_ref': 'synthetic-pbs:'+attempt, 'basis': 'measured',
                  'unit': 'mig_instance_hours', 'quantity': .01,
                  'source_ref': 'synthetic terminal scheduler fixture'}} for attempt in attempts])
    return {'runs': roots, 'training_source': SOURCE, 'evaluation_source': SOURCE,
        'full_context_lora_run': str(full), 'deterministic_checks': {'manifest_ref': reference(check)},
        'interventions': {'SEG-02': str(iv)}, 'observations': [str(observations)], 'accounted_attempts': attempts}


def test_full_synthetic_analysis_runs_locked_bootstrap_and_rules(grid):
    report = analysis.analyze(grid)
    assert report['bootstrap']['repeats'] == 10000 and report['bootstrap']['seed'] == 20261101
    assert report['rules']['1']['passed'] is True and report['primary'] == 'piqa'
    assert report['direction'] == 'challenges'
    assert report['cost']['terminal_coverage_complete']
    assert report['cost']['node_h'] == pytest.approx(9*.01*.25)
    assert report['cost']['within_cap'] and report['cost']['cap_node_h'] == 3
    assert 'ATT-0607-1' in report['cost']['required_attempts']  # failed jobs cost too
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


@pytest.mark.parametrize('wrong', ['check_source', 'evaluation_source', 'intervention_source'])
def test_analysis_rejects_unpinned_nontraining_sources(grid, wrong):
    from pathlib import Path
    if wrong == 'check_source':
        path = Path(grid['deterministic_checks']['manifest_ref']['path'])
        check = analysis.read(path); check['source_commit'] = 'c'*40; write(path, check)
        grid['deterministic_checks']['manifest_ref'] = reference(path)
    elif wrong == 'evaluation_source':
        grid['evaluation_source'] = 'c'*40
    else:
        path = Path(grid['interventions']['SEG-02'])/'provenance.json'
        prov = analysis.read(path); prov['evaluation_source'] = 'c'*40; write(path, prov)
    with pytest.raises(ValueError, match='source|provenance'): analysis.analyze(grid)


@pytest.mark.parametrize('missing', ['base', 'training', 'intervention', 'check', 'smoke', 'failed'])
def test_analysis_rejects_missing_terminal_attempt_usage(grid, missing):
    from pathlib import Path
    target = {'base': 'ATT-0600-1', 'training': 'ATT-0602-1', 'intervention': 'ATT-0605-1',
              'check': 'ATT-0606-1', 'smoke': 'ATT-0271-1', 'failed': 'ATT-0607-1'}[missing]
    path = Path(grid['observations'][0])
    write(path, [r for r in analysis.read(path) if r['attempt_id'] != target])
    with pytest.raises(ValueError, match='cost coverage'): analysis.analyze(grid)


def test_declared_accounting_cannot_omit_required_arm_or_smoke(grid):
    grid['accounted_attempts'].remove('ATT-0271-1')
    with pytest.raises(ValueError, match='omits a required'): analysis.analyze(grid)


def test_cost_deduplicates_exports_and_reports_a_cap_excess(grid):
    from pathlib import Path
    path = Path(grid['observations'][0]); rows = analysis.read(path)
    rows[-1]['usage']['quantity'] = 13.
    write(path, rows)
    cost = analysis.complete_cost(dict(grid, observations=grid['observations']*2), grid['runs'],
        grid['interventions'], grid['deterministic_checks']['manifest_ref']['path'])
    assert len(cost['rows']) == len(grid['accounted_attempts'])
    assert cost['within_cap'] is False and cost['node_h'] > 3
