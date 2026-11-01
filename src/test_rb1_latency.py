"""Synthetic artifact-boundary tests for the E-0021 latency driver."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

import rb1_latency as latency
from rb_train import FAMILIES, PARAMETERS, adapter_config
from v1_lora import LoRAConfig
from v1_run import lr_factor
from calibrate import REVISION


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_run(root, arm):
    run = root / arm
    adapter = run / 'adapter'
    adapter.mkdir(parents=True)
    split = run / 'split.json'
    split.write_text('{"split":"fixture"}')
    lock = run / 'lock.json'
    split_hash = _digest(split)
    lock.write_text(json.dumps({'experiment': 'E-0020@1',
        'locked_lrs': {FAMILIES[arm]: .001}, 'split_manifest_sha256': split_hash}))
    training_config = {'arm': arm, 'phase': 'R1', 'seed': 0, 'stage': 'train',
                       'lr': .001, 'lr_lock': str(lock), 'split_manifest': str(split),
                       'eval_limit': None, 'smoke_steps': None, 'three_epochs': False}
    (run / 'config.json').write_text(json.dumps(training_config))
    model_config = (LoRAConfig(layers=(6, 13, 20, 27), rank=7, alpha=7)
                    if arm == 'RB1-07' else adapter_config(arm))
    (adapter / 'config.json').write_text(json.dumps(asdict(model_config)))
    (adapter / 'adapter.pt').write_bytes((arm + '-weights').encode())
    checkpoint = {}
    for name in ('config.json', 'adapter.pt'):
        path = adapter / name
        checkpoint[name] = {'path': str(path), 'sha256': _digest(path), 'bytes': path.stat().st_size}
    (run / 'checkpoint.json').write_text(json.dumps(checkpoint))
    steps = [{'step': index, 'example_mean_nll': 2.4, 'preclip_norm': .4,
              'lr': .001 * lr_factor(index - 1, 1954), 'seconds': 1.,
              'input_tokens': 1024 if index < 1954 else 128}
             for index in range(1, 1955)]
    (run / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in steps))
    params = PARAMETERS[arm]
    metrics = {'failed': False, 'reload.identical': True, 'params.stored': params,
        'model_revision': REVISION, 'tokenizer_revision': REVISION,
        'environment.lock_sha256': _digest(Path(__file__).resolve().parents[1]/'uv.lock'),
        'source.commit': 'c'*40, 'plan.order_sha256': 'a'*64,
        'params.trainable': params, 'params.with_grad': params,
        'learning_rate': .001, 'lr_lock.sha256': _digest(lock),
        'split_manifest.sha256': split_hash, 'plan.steps': 1954, 'train.steps': 1954,
        'plan.input_tokens': 2_000_000, 'train.input_tokens': 2_000_000,
        'plan.examples': 3908, 'train.examples': 3908, 'train.last_epoch': 0}
    (run / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'contract': latency.CONTRACT, 'arm': arm, 'family': FAMILIES[arm],
        'phase': 'R1', 'seed': 0, 'stage': 'train',
        'metrics': metrics,
        'validator': {'passed': True}}))
    return run


def test_formal_checkpoint_binding_and_seed_swap_rejection(tmp_path):
    run = _formal_run(tmp_path, 'RB1-01')
    assert latency.validate_formal_checkpoint(run, 'RB1-01') == run / 'adapter'
    with pytest.raises(ValueError, match='manifest arm'):
        latency.validate_formal_checkpoint(run, 'RB1-02')


@pytest.mark.parametrize('change,match', [
    ({'stage': 'smoke'}, 'manifest arm/phase/seed/stage'),
    ({'seed': 1}, 'manifest arm/phase/seed/stage'),
    ({'phase': 'RB5'}, 'manifest arm/phase/seed/stage'),
])
def test_rejects_smoke_or_wrong_provenance(tmp_path, change, match):
    run = _formal_run(tmp_path, 'RB1-01')
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(change)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        latency.validate_formal_checkpoint(run, 'RB1-01')


def test_rejects_swapped_or_modified_checkpoint_bytes(tmp_path):
    run_a = _formal_run(tmp_path, 'RB1-01')
    run_b = _formal_run(tmp_path, 'RB1-02')
    cp_a = json.loads((run_a / 'checkpoint.json').read_text())
    cp_b = json.loads((run_b / 'checkpoint.json').read_text())
    cp_a['adapter.pt'] = cp_b['adapter.pt']
    (run_a / 'checkpoint.json').write_text(json.dumps(cp_a))
    with pytest.raises(ValueError, match='checkpoint path escapes'):
        latency.validate_formal_checkpoint(run_a, 'RB1-01')
    (run_a / 'checkpoint.json').write_text(json.dumps({
        **cp_a, 'adapter.pt': {**cp_a['adapter.pt'], 'path': str(run_a/'adapter'/'adapter.pt')}}))
    (run_a / 'adapter' / 'adapter.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash/size mismatch'):
        latency.validate_formal_checkpoint(run_a, 'RB1-01')


def test_checkpoint_map_requires_all_timing_arms():
    with pytest.raises(ValueError, match='must contain'):
        latency.validate_checkpoint_map({'RB1-01': '/unused'})


def test_cli_guards_before_model_load_and_writes_complete_metrics(tmp_path, monkeypatch):
    runs = {arm: _formal_run(tmp_path, arm) for arm in latency.TIMED_ARMS}
    mapping = tmp_path / 'checkpoints.json'
    mapping.write_text(json.dumps({arm: str(run) for arm, run in runs.items()}))
    output = tmp_path / 'timing'
    events = []

    class FakeLora:
        def __init__(self):
            self._targets = {}

    monkeypatch.setattr(latency, 'LoRAModel', FakeLora)
    monkeypatch.setattr(latency, 'profile_guard', lambda argv: events.append('guard') or 0)
    monkeypatch.setattr(latency, 'make_model', lambda arm, **kwargs:
                        events.append('load:' + arm) or FakeLora())
    monkeypatch.setattr(latency, 'measure', lambda model: {
        'prefill_seconds': [1.] * 5, 'prefill_tokens_per_second': 512.,
        'decode_p50_ms': 64.5, 'decode_p95_ms': 121., 'decode_samples_ms': list(map(float, range(1, 129))),
        'batch': 1, 'prefill_length': 512, 'prefill_passes': 5, 'decode_tokens': 128,
        'peak_cuda_bytes': 100, 'adapter_persistent_state': {'M_c_bytes': 0}})
    monkeypatch.setattr(latency, 'merge_lora', lambda model: events.append('merge') or object())
    monkeypatch.setattr(latency.torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(latency.torch.cuda, 'get_device_name', lambda: 'mock-MIG')
    monkeypatch.setattr(latency.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setenv('SR_SPEC_DIGEST', 'fixture-spec')
    monkeypatch.setenv('SR_CODE_COMMIT', 'fixture-commit')
    latency.main(['--checkpoints', str(mapping), '--include-base', '--output', str(output)])
    assert events[0] == 'guard'
    assert (output / 'manifest.json').is_file()
    manifest = json.loads((output / 'manifest.json').read_text())
    result = json.loads((output / 'latency.json').read_text())
    assert manifest['status'] == 'complete' and manifest['validator']['passed']
    assert set(result['measurements']) == {*latency.TIMED_ARMS, 'RB1-07-merged', 'Base'}
    assert result['measurements']['RB1-07']['decode_samples_ms'] == list(map(float, range(1, 129)))
    assert result['loaded_refs']['RB1-01']['artifact_inventory']['adapter_weights']['sha256']


def test_nonfinite_measurement_never_marks_manifest_complete(tmp_path, monkeypatch):
    runs = {arm: _formal_run(tmp_path, arm) for arm in latency.TIMED_ARMS}
    mapping = tmp_path / 'checkpoints.json'
    mapping.write_text(json.dumps({arm: str(run) for arm, run in runs.items()}))
    monkeypatch.setattr(latency, 'profile_guard', lambda argv: 0)
    monkeypatch.setattr(latency, 'make_model', lambda arm, **kwargs: object())
    monkeypatch.setattr(latency, 'measure', lambda model: {
        'batch': 1, 'prefill_length': 512, 'prefill_passes': 5, 'decode_tokens': 128,
        'prefill_seconds': [1.] * 5, 'prefill_tokens_per_second': 512.,
        'decode_samples_ms': [float('nan')] + list(map(float, range(1, 128))),
        'decode_p50_ms': 64., 'decode_p95_ms': 120., 'peak_cuda_bytes': 100,
        'adapter_persistent_state': {'M_c_bytes': 0}})
    monkeypatch.setattr(latency.torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setenv('SR_CODE_COMMIT', 'fixture-commit')
    monkeypatch.setenv('SR_SPEC_DIGEST', 'fixture-spec')
    output = tmp_path / 'invalid-timing'
    with pytest.raises(ValueError, match='must be finite'):
        latency.main(['--checkpoints', str(mapping), '--output', str(output)])
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'partial' and not manifest['validator']['passed']


def test_mig_guard_fails_before_any_model_load(tmp_path, monkeypatch):
    runs = {arm: _formal_run(tmp_path, arm) for arm in latency.TIMED_ARMS}
    mapping = tmp_path / 'checkpoints.json'
    mapping.write_text(json.dumps({arm: str(run) for arm, run in runs.items()}))
    events = []
    monkeypatch.setattr(latency, 'profile_guard', lambda argv: events.append('guard') or 3)
    monkeypatch.setattr(latency, 'make_model', lambda *a, **k: pytest.fail('model loaded despite failed guard'))
    monkeypatch.setenv('SR_CODE_COMMIT', 'fixture-commit')
    monkeypatch.setenv('SR_SPEC_DIGEST', 'fixture-spec')
    with pytest.raises(RuntimeError, match='MIG profile guard'):
        latency.main(['--checkpoints', str(mapping), '--output', str(tmp_path/'guard-fail')])
    assert events == ['guard']
