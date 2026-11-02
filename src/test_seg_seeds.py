"""E-0027 new-seed path: initialization, no-channel identity, artifacts, rules, analysis."""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from bypass_model import BranchModel
from rb1_artifacts import reference
from rb_history import donor_indices
import seg_artifacts
import seg_seed_analysis
from seg_model import make_model, forward, parameter_digest
from seg_seed_rules import analyze, LABELS
from test_seg_artifacts import formal, seal, predictions, write, SOURCE

SEED0 = {'SEG-04': 642, 'SEG-05': 659, 'SEG-06': 615, 'SEG-07': 612, 'SEG-05/normal': 659, 'SEG-05/iv_swap': 594}
CHECKS = {k: True for k in ('off_matches_base', 'initial_matches_base', 'no_channel_identity',
                            'reload_identical', 'initial_parameters_identical')}


def seed_counts(gain, lora, swap):
    keep = 600+gain
    return {'SEG-06': (600, 1000), 'SEG-07': (590, 1000), 'SEG-05': (keep, 1000), 'SEG-04': (keep-lora, 1000),
            'SEG-05/normal': (keep, 1000), 'SEG-05/iv_swap': (keep-swap, 1000)}


def run_rules(one, two, checks=None):
    counts = {0: {k: (v, 1000) for k, v in SEED0.items()}, 1: seed_counts(*one), 2: seed_counts(*two)}
    return analyze(counts, checks or {1: CHECKS, 2: CHECKS}, (609, 1000))


@pytest.mark.parametrize('one, two, direction', [
    ((23, 0, -10), (23, 13, -10), 'supports'),        # G=3, L=1, W=G/2 exactly
    ((23, 0, -11), (23, 13, -10), 'inconclusive'),    # W just below G/2
    ((23, 0, -10), (22, 13, -10), 'inconclusive'),    # G just below 3
    ((23, 0, -10), (23, 12, -10), 'inconclusive'),    # L just below 1
    ((46, 13, 0), (0, 0, 0), 'inconclusive'),         # gain2 = 0 blocks supports only
    ((0, 13, 0), (0, 13, 0), 'challenges'),           # both new gains <= 0
    ((-20, 13, 0), (3, 13, 0), 'challenges'),         # G = 0.9 < 1
    ((-14, 13, 0), (0, 13, 0), 'challenges'),         # G = 1 is not < 1, but both gains <= 0
    ((-13, 13, 0), (1, 13, 0), 'inconclusive'),       # G = 32/3000 >= 1, gain2 > 0
    ((23, -17, 0), (23, 0, 0), 'challenges'),         # L = 0
])
def test_exact_three_seed_boundaries(one, two, direction):
    result = run_rules(one, two)
    assert result['direction'] == direction
    assert result['eligible_edge'] == (direction if direction != 'inconclusive' else None)


def test_rule_bookkeeping_learnability_and_failed_implementation():
    result = run_rules((23, 0, -10), (23, 13, -10))
    assert result['G']['numerator'] * 100 == 3 * result['G']['denominator']
    assert result['per_seed']['0']['gain']['pp'] == pytest.approx(4.4)
    # Base 60.9 -> learnability threshold 61.9; seed 1 controls 60.0/59.0 are unlearned.
    assert result['rules']['7']['no_history_r128_controls_unlearned'] == {'0': True, '1': True, '2': True}
    assert result['rules']['7']['learned']['0'] == {'SEG-04': True, 'SEG-05': True, 'SEG-06': False, 'SEG-07': False}
    broken = run_rules((23, 0, -10), (23, 13, -10), {1: CHECKS, 2: {**CHECKS, 'initial_parameters_identical': False}})
    assert broken['direction'] == 'invalid_implementation' and broken['eligible_edge'] is None


def test_seed0_must_equal_locked_E0025_values_and_inputs_are_complete():
    counts = {0: {k: (v, 1000) for k, v in SEED0.items()}, 1: seed_counts(1, 1, 1), 2: seed_counts(1, 1, 1)}
    counts[0]['SEG-04'] = (643, 1000)
    with pytest.raises(ValueError, match='seed-0 lora'):
        analyze(counts, {1: CHECKS, 2: CHECKS}, (609, 1000))
    counts[0]['SEG-04'] = (642, 1000)
    del counts[2]['SEG-05/iv_swap']
    with pytest.raises(ValueError):
        analyze(counts, {1: CHECKS, 2: CHECKS}, (609, 1000))


# ---- GPU: initialization and no-channel identity at the new seeds ----

@pytest.fixture(scope='module')
def base():
    from calibrate import load_base
    return load_base()


@contextmanager
def seeded(base, arm, seed):
    with patch('seg_model.load_base', return_value=base):
        model = make_model(arm, seed=seed)
    try:
        yield model
    finally:
        if isinstance(model, BranchModel):
            for i, layer in model.adapters().items():
                for h in layer.handles: h.remove()
                base.model.layers[i] = layer.original
        elif arm == 'SEG-04':
            from v1_lora import _get_parent
            for (i, path), target in model._targets.items():
                parent, name = _get_parent(base.model.layers[i], path)
                setattr(parent, name, target.base)
        base.zero_grad(set_to_none=True)


def test_r128_initial_parameters_identical_within_seed_and_distinct_across_seeds(base):
    digests = {}
    for seed in (0, 1, 2):
        for arm in ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07'):
            with seeded(base, arm, seed) as model:
                digests[seed, arm] = parameter_digest(model)
        assert digests[seed, 'SEG-05'] == digests[seed, 'SEG-06'] == digests[seed, 'SEG-07']
        assert digests[seed, 'SEG-04'] != digests[seed, 'SEG-05']
    assert len({digests[s, 'SEG-05'] for s in (0, 1, 2)}) == 3
    assert len({digests[s, 'SEG-04'] for s in (0, 1, 2)}) == 3


IDS = ([[11, 29, 73, 181, 8]], [[71, 9, 18, 41]])


@pytest.mark.parametrize('seed', [1, 2], ids=['seed1', 'seed2'])
@pytest.mark.parametrize('arm', ['SEG-04', 'SEG-06', 'SEG-07'])
def test_no_channel_identity_at_new_seed(base, arm, seed):
    a, b = (torch.tensor(x, device='cuda') for x in IDS)
    with torch.no_grad():
        untouched = base(b, use_cache=False).logits
    with seeded(base, arm, seed) as model, torch.no_grad():
        model.eval()
        if isinstance(model, BranchModel):
            for layer in model.adapters().values():
                layer.side.fusion.P.normal_(std=.03)
        else:
            for target in model._targets.values():
                target.B.normal_(std=.03)
        direct, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b))
        discarded, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), force_first=True)
        assert torch.equal(direct.logits, discarded.logits)
        assert not torch.equal(direct.logits, untouched)


@pytest.mark.parametrize('seed', [1, 2], ids=['seed1', 'seed2'])
def test_zero_initialized_keep_matches_base_at_new_seed(base, seed):
    a, b = (torch.tensor(x, device='cuda') for x in IDS)
    with torch.no_grad():
        expected = base(b, use_cache=False).logits
    with seeded(base, 'SEG-05', seed) as model, torch.no_grad():
        model.eval()
        initial, _ = forward(model, 'SEG-05', a, torch.ones_like(a), b, torch.ones_like(b))
        assert torch.equal(initial.logits, expected)


# ---- artifacts and full synthetic analysis ----

ORDERS = {0: seg_artifacts.ORDER_SHA256_BY_SEED[0], 1: 'c'*64, 2: 'e'*64}


def as_seed(root, arm, seed, digest='d'*64):
    manifest = json.loads((root/'manifest.json').read_text())
    config = json.loads((root/'config.json').read_text())
    write(root/'config.json', {**config, 'seed': seed})
    manifest.update({'contract': 'E-0027@1', 'seed': seed})
    manifest['metrics'].update({'seed': seed, 'plan.order_sha256': ORDERS[seed], 'initial.adapter_sha256': digest})
    seal(root, manifest)


@pytest.fixture
def orders(monkeypatch):
    monkeypatch.setattr(seg_artifacts, 'ORDER_SHA256_BY_SEED', ORDERS)


def test_new_seed_artifact_binding(tmp_path, monkeypatch, orders):
    root = tmp_path/'ATT-0001-1'
    formal(root, 'SEG-06', monkeypatch); as_seed(root, 'SEG-06', 1)
    seg_artifacts.validate(root, 'SEG-06', SOURCE, 1)
    for seed in (0, 2):
        with pytest.raises(ValueError):
            seg_artifacts.validate(root, 'SEG-06', SOURCE, seed)
    other = tmp_path/'ATT-0002-1'
    formal(other, 'SEG-01', monkeypatch)
    with pytest.raises(ValueError, match='not registered'):
        seg_artifacts.validate(other, 'SEG-01', SOURCE, 1)
    manifest = json.loads((root/'manifest.json').read_text()); del manifest['metrics']['initial.adapter_sha256']
    seal(root, manifest)
    with pytest.raises(ValueError, match='digest'):
        seg_artifacts.validate(root, 'SEG-06', SOURCE, 1)
    monkeypatch.setattr(seg_artifacts, 'ORDER_SHA256_BY_SEED', {**ORDERS, 1: None})
    with pytest.raises(ValueError, match='not registered'):
        seg_artifacts.validate(root, 'SEG-06', SOURCE, 1)


@pytest.fixture
def grid(tmp_path, monkeypatch, orders):
    base = tmp_path/'ATT-0100-1'; formal(base, 'SEG-00', monkeypatch)
    labels = {**SEED0, 'SEG-00': 600}
    report = {'contract': 'E-0025@1', 'stage': 'final', 'rules': {'1': {'passed': True}},
              'counts': {k: {'piqa': [600, 1000], 'boolq': [v, 1000]} for k, v in labels.items()},
              'summary': {k: {'synthetic': True} for k in (*SEED0, *(f'SEG-05/{x}' for x in seg_seed_analysis.KINDS))},
              'bootstrap': {'max_control_gain': {'128': {}}, 'pairs': {}}}
    write(tmp_path/'seed0.json', report)
    monkeypatch.setattr(seg_seed_analysis, 'SEED0_ANALYSIS_SHA256', reference(tmp_path/'seed0.json')['sha256'])
    monkeypatch.setattr(seg_seed_analysis, 'BASE_SOURCE', SOURCE)
    rows = predictions()
    mapping = donor_indices(rows, 'R1')
    donors = [{'task': r['task'], 'recipient': r['id'], 'donor': rows[mapping[i]]['id']} for i, r in enumerate(rows)]
    runs, interventions, attempts = {}, {}, []
    n = 101
    for seed in (1, 2):
        runs[str(seed)] = {}
        for arm in ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07'):
            root = tmp_path/f'ATT-{n:04}-1'; n += 1
            formal(root, arm, monkeypatch); as_seed(root, arm, seed, 'f'*64 if arm == 'SEG-04' else 'd'*64)
            runs[str(seed)][arm] = str(root); attempts.append(root.name)
        _, refs, _ = seg_artifacts.validate(runs[str(seed)]['SEG-05'], 'SEG-05', SOURCE, seed)
        iv = tmp_path/f'ATT-{n:04}-1'; n += 1
        write(iv/'conditions.json', {k: {'predictions': rows, 'audit': {'donors': donors if k == 'iv_swap' else []}}
                                     for k in seg_seed_analysis.KINDS})
        write(iv/'provenance.json', {'training': refs, 'checkpoint_files': json.loads((Path(runs[str(seed)]['SEG-05'])/'checkpoint.json').read_text()),
                                    'training_source': SOURCE, 'evaluation_source': SOURCE})
        write(iv/'manifest.json', {'status': 'complete', 'contract': 'E-0027@1', 'seed': seed, 'stage': 'intervention',
                                  'arm': 'SEG-05', 'metrics': {'off.identical_to_Base_seg': True}, 'validator': {'passed': True}})
        interventions[str(seed)] = str(iv); attempts.append(iv.name)
    check = tmp_path/'ATT-0200-1'/'manifest.json'
    write(check, {'status': 'complete', 'source_commit': SOURCE, 'validator': {'passed': True},
                  'metrics': {'no_channel_identity': True, 'no_channel_seeds': [1, 2]}})
    attempts += ['ATT-0200-1', 'ATT-0300-1']
    observations = tmp_path/'costs.json'
    write(observations, [{'attempt_id': a, 'execution_state': 'ended', 'usage': {'accounting_ref': 'pbs:'+a,
        'basis': 'measured', 'unit': 'mig_instance_hours', 'quantity': 1.1, 'source_ref': 'fixture'}} for a in attempts])
    return {'seed0_analysis': reference(tmp_path/'seed0.json'), 'base_run': str(base), 'training_source': SOURCE,
            'evaluation_source': SOURCE, 'runs': runs, 'interventions': interventions,
            'deterministic_checks': {'manifest_ref': reference(check)},
            'observations': [str(observations)], 'accounted_attempts': attempts}


def test_full_synthetic_three_seed_analysis(grid):
    report = seg_seed_analysis.analyze(grid)
    # New seeds: every arm 60.0% -> gain 0 at both seeds -> challenges.
    assert report['direction'] == 'challenges'
    assert report['rules']['1']['passed'] is True
    assert report['per_seed']['1']['gain']['pp'] == 0
    assert report['G']['numerator'] * 3000 == 44 * report['G']['denominator']
    assert report['cost']['node_h'] == pytest.approx(12*1.1*.25) and report['cost']['within_cap'] is False
    assert set(report['seeds']['1']['bootstrap']['max_control_gain']) == {'piqa', 'boolq', 'macro'}
    assert report['seeds']['1']['normal_training_prediction_flips'] == 0


@pytest.mark.parametrize('failure', ['digest', 'seed0', 'cost', 'donor', 'checks'])
def test_three_seed_analysis_rejects_or_flags_failures(grid, failure, monkeypatch):
    if failure == 'digest':
        root = Path(grid['runs']['2']['SEG-07'])
        as_seed(root, 'SEG-07', 2, 'a'*64)
        assert seg_seed_analysis.analyze(grid)['direction'] == 'invalid_implementation'
        return
    if failure == 'seed0':
        monkeypatch.setattr(seg_seed_analysis, 'SEED0_ANALYSIS_SHA256', '0'*64)
    elif failure == 'cost':
        grid['accounted_attempts'] = grid['accounted_attempts'][:-1]
    elif failure == 'donor':
        path = Path(grid['interventions']['1'])/'conditions.json'
        conditions = json.loads(path.read_text()); conditions['iv_swap']['audit']['donors'][0]['donor'] = 999
        path.write_text(json.dumps(conditions))
    elif failure == 'checks':
        path = Path(grid['deterministic_checks']['manifest_ref']['path'])
        write(path, {'status': 'complete', 'source_commit': SOURCE, 'validator': {'passed': True},
                     'metrics': {'no_channel_identity': True, 'no_channel_seeds': [1]}})
        grid['deterministic_checks'] = {'manifest_ref': reference(path)}
    with pytest.raises(ValueError):
        seg_seed_analysis.analyze(grid)
