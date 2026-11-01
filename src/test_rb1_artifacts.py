"""Formal-budget and provenance checks for RB1 producer artifacts."""
import json

import pytest

from rb1_artifacts import validate_formal_run, verified_latency
from test_rb1_latency import _formal_run
from v1_run import lr_factor


def _rewrite(run, name, change):
    path = run / name
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    return value


def test_formal_run_returns_manifest_reference_inventory_and_adapter(tmp_path):
    run = _formal_run(tmp_path, 'RB1-01')
    manifest, refs, adapter = validate_formal_run(run, 'RB1-01')
    assert manifest['metrics']['train.input_tokens'] == 2_000_000
    assert refs['manifest']['sha256'] and refs['lr_lock']['sha256']
    assert refs['adapter_weights']['bytes'] > 0
    assert adapter == run / 'adapter'


@pytest.mark.parametrize('key,value', [
    ('params.stored', 1), ('params.trainable', 1), ('params.with_grad', 1),
])
def test_rejects_wrong_parameter_accounting(tmp_path, key, value):
    run = _formal_run(tmp_path, 'RB1-01')
    _rewrite(run, 'manifest.json', lambda data: data['metrics'].__setitem__(key, value))
    with pytest.raises(ValueError, match=key):
        validate_formal_run(run, 'RB1-01')


def test_rejects_lock_family_split_or_learning_rate_mismatch(tmp_path):
    run = _formal_run(tmp_path, 'RB1-01')
    lock = run / 'lock.json'
    value = json.loads(lock.read_text())
    value['locked_lrs']['RBc-T'] = .003
    lock.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='LR lock family/value'):
        validate_formal_run(run, 'RB1-01')


def test_rejects_actual_split_file_change(tmp_path):
    run = _formal_run(tmp_path, 'RB1-01')
    (run / 'split.json').write_text('{"split":"changed"}')
    with pytest.raises(ValueError, match='LR lock split hash'):
        validate_formal_run(run, 'RB1-01')


def test_rejects_crossing_two_million_before_final_step(tmp_path):
    run = _formal_run(tmp_path, 'RB1-01')
    steps_path = run / 'steps.jsonl'
    rows = [json.loads(line) for line in steps_path.read_text().splitlines()]
    rows.append({'step': 1955, 'example_mean_nll': 2.4, 'preclip_norm': .4,
                 'lr': .001 * lr_factor(1954, 1955), 'seconds': 1., 'input_tokens': 1})
    for index, row in enumerate(rows, 1):
        row['lr'] = .001 * lr_factor(index - 1, 1955)
    steps_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    _rewrite(run, 'manifest.json', lambda data: data['metrics'].update(
        **{'plan.steps': 1955, 'train.steps': 1955, 'plan.examples': 3910,
           'train.examples': 3910, 'plan.input_tokens': 2_000_001,
           'train.input_tokens': 2_000_001}))
    with pytest.raises(ValueError, match='continued after reaching'):
        validate_formal_run(run, 'RB1-01')


def test_accepts_natural_three_epoch_stop_below_two_million(tmp_path):
    run = _formal_run(tmp_path, 'RB1-01')
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text())
    config['three_epochs'] = False
    config_path.write_text(json.dumps(config))
    rows = []
    for step in range(1, 4):
        rows.append({'step': step, 'example_mean_nll': 2.4, 'preclip_norm': .2,
                     'lr': .001 * lr_factor(step - 1, 3), 'seconds': 1., 'input_tokens': 100})
    (run / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    _rewrite(run, 'manifest.json', lambda data: data['metrics'].update(
        **{'plan.steps': 3, 'train.steps': 3, 'plan.input_tokens': 300,
           'train.input_tokens': 300, 'plan.examples': 6, 'train.examples': 6,
           'train.last_epoch': 2, 'plan.epoch_end_steps': {'3': 3}}))
    manifest, _, _ = validate_formal_run(run, 'RB1-01')
    assert manifest['metrics']['train.input_tokens'] == 300


@pytest.mark.parametrize('bad_value,match', [(1025, 'token bound'), (0.5, 'step LR')])
def test_rejects_step_token_overflow_or_wrong_locked_lr(tmp_path, bad_value, match):
    run = _formal_run(tmp_path, 'RB1-01')
    path = run / 'steps.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if match == 'token bound':
        rows[0]['input_tokens'] = bad_value
    else:
        rows[0]['lr'] = bad_value
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=match):
        validate_formal_run(run, 'RB1-01')


def test_rejects_three_epoch_override_even_with_formal_stage(tmp_path):
    run = _formal_run(tmp_path, 'RB1-01')
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text())
    config['three_epochs'] = True
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='three-epoch override'):
        validate_formal_run(run, 'RB1-01')


def test_latency_requires_full_raw_samples_and_matching_summary():
    import statistics
    samples = list(map(float, range(1, 129)))
    prefill = [1.] * 5
    row = {'batch': 1, 'prefill_length': 512, 'prefill_passes': 5, 'decode_tokens': 128,
           'decode_samples_ms': samples, 'prefill_seconds': prefill,
           'decode_p50_ms': statistics.median(samples), 'decode_p95_ms': samples[120],
           'prefill_tokens_per_second': 512., 'peak_cuda_bytes': 100,
           'adapter_persistent_state': {'M_c_bytes': 16896}}
    assert verified_latency(row) is row
    with pytest.raises(ValueError, match='128 decode samples'):
        verified_latency({**row, 'decode_samples_ms': samples[:2]})
