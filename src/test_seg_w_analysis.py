"""E-0032 analysis wiring on the pinned real references with synthetic variant runs (PBS)."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import seg_w_analysis as analysis
from rb1_artifacts import read
from seg_model import PARAMETERS, SEG05_INITIAL_SHA256, WRITE_RULE, KEEP

SOURCE = 'a'*40
LAYERS = ('6', '13', '20', '27')


def reference_rows(seed, label):
    relative, _ = analysis.REFERENCE_FILES[seed][label]
    data = read(analysis.RUNS+relative)
    return data['iv_swap']['predictions'] if label.endswith('iv_swap') else data


def flip(rows, task, n):
    """Make n more rows of task correct (n > 0) or wrong (n < 0)."""
    rows = copy.deepcopy(rows)
    target = (lambda r: r['prediction'] != r['gold']) if n > 0 else (lambda r: r['prediction'] == r['gold'])
    changed = 0
    for r in rows:
        if changed == abs(n):
            break
        if r['task'] == task and target(r):
            r['prediction'] = 1-r['prediction']
            r['scores'] = [0., -1.] if r['prediction'] == 0 else [-1., 0.]
            changed += 1
    assert changed == abs(n)
    return rows


def write_run(root, arm, seed, rows, swap=None, initial_digest=None):
    root.mkdir(parents=True)
    base = reference_rows(0, 'SEG-00')
    files = {'predictions.json': rows, 'initial-predictions.json': base, 'initial-base-predictions.json': base,
             'iv-off-dev.json': {'predictions': base}, 'checkpoint.json': {}}
    if swap is not None:
        files['iv-swap-dev.json'] = {'predictions': swap}
    for name, data in files.items():
        (root/name).write_text(json.dumps(data))
    seg1 = 1000 if arm in KEEP else 0
    writes = 26230 if arm == 'SEG-W1' else 40000
    metrics = {'source.commit': SOURCE, 'initial.identical_to_Base_seg': True, 'off.identical_to_Base_seg': True,
               'reload.identical': True, 'params.stored': PARAMETERS[arm], 'params.trainable': PARAMETERS[arm],
               'params.with_grad': PARAMETERS[arm], 'train.segment2_tokens': 500, 'train.segment1_tokens': seg1,
               'initial.adapter_sha256': initial_digest or 'f'*64,
               'runtime.accounting': {'segment2': {i: {'writes': 500, 'reads': 500} for i in LAYERS}}}
    if arm in WRITE_RULE:
        metrics['runtime.accounting']['segment1'] = {i: {'writes': writes, 'reads': seg1} for i in LAYERS}
        one = arm == 'SEG-W1'
        metrics['train.segment1_writes'] = {'total': writes, 'min': 1 if one else 1, 'max': 1 if one else 4}
        metrics['dev.segment1_writes'] = {'min': 1, 'max': 1 if one else 5}
    from v5_train import file_sha256
    artifacts = [{'path': str(p), 'role': p.name, 'sha256': file_sha256(p)} for p in root.iterdir()]
    (root/'manifest.json').write_text(json.dumps({'contract': 'E-0032@1', 'status': 'complete',
        'validator': {'passed': True}, 'arm': arm, 'seed': seed, 'stage': 'train', 'metrics': metrics,
        'artifacts': artifacts}))
    return str(root)


@pytest.fixture
def spec(tmp_path):
    seg05 = reference_rows(0, 'SEG-05')
    seg06 = reference_rows(0, 'SEG-06')
    runs = {
        'SEG-W1': write_run(tmp_path/'w1', 'SEG-W1', 0, seg05, flip(seg05, 'boolq', -60), SEG05_INITIAL_SHA256[0]),
        'SEG-W2': write_run(tmp_path/'w2', 'SEG-W2', 0, flip(seg05, 'boolq', 20), flip(seg05, 'boolq', -40),
                            SEG05_INITIAL_SHA256[0]),
        'SEG-X5': write_run(tmp_path/'x5', 'SEG-X5', 0, flip(seg05, 'boolq', 10), flip(seg05, 'boolq', -50)),
        'SEG-X6': write_run(tmp_path/'x6', 'SEG-X6', 0, seg06),
        'SEG-X7': write_run(tmp_path/'x7', 'SEG-X7', 0, seg06)}
    for name, status in (('smoke', {'checks.failed': 0, 'checks.cases': 10}), ('replay', {'replay.rows_identical': True})):
        (tmp_path/name).mkdir()
        (tmp_path/name/'manifest.json').write_text(json.dumps({'status': 'complete', 'validator': {'passed': True},
                                                               'metrics': status}))
    data = {'contract': 'E-0032@1', 'training_source': SOURCE, 'runs': {'0': runs},
            'smoke_manifest': str(tmp_path/'smoke'/'manifest.json'),
            'replay_manifest': str(tmp_path/'replay'/'manifest.json')}
    path = tmp_path/'input.json'
    path.write_text(json.dumps(data))
    return path


def test_seed0_screening_end_to_end(spec, tmp_path):
    out = tmp_path/'out'; out.mkdir()
    values = analysis.run(type('Args', (), {'input': spec, 'output': out})())
    report = read(out/'analysis.json')
    assert values['implementation_ok'] is True
    seed0 = report['rules']['2']['seed0']
    assert seed0['SEG-W1']['delta']['pp'] == 0 and seed0['SEG-W2']['delta']['pp'] == pytest.approx(2.0)
    assert seed0['SEG-X5']['delta']['pp'] == pytest.approx(1.0)
    assert seed0['SEG-W2']['swap']['pp'] == pytest.approx(6.0)
    # W2 passes (delta 2.0), X fails (delta 1.0): W2 is selected and awaits replication.
    assert report['rules']['4']['selected'] == 'SEG-W2'
    assert report['outcomes'] == {'H-0032': 'pending_replication', 'H-0033': 'inconclusive'}
    boot = report['bootstrap']['pp']['0']
    assert boot['SEG-W1']['delta']['boolq'] == [0., 0.]
    low, high = boot['SEG-W2']['delta']['boolq']
    assert 0 < low <= 2.0 <= high
    assert report['accuracy']['0']['SEG-05']['boolq']['pp'] == pytest.approx(65.9)


def test_rule1_failures_block_interpretation(spec, tmp_path):
    data = read(spec)
    manifest_path = Path(data['runs']['0']['SEG-X6'])/'manifest.json'
    manifest = read(manifest_path)
    manifest['metrics']['reload.identical'] = False
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path/'out'; out.mkdir()
    values = analysis.run(type('Args', (), {'input': spec, 'output': out})())
    assert values['implementation_ok'] is False
    assert set(values['outcomes'].values()) == {'invalid_implementation'}


def test_tampered_run_or_reference_is_rejected(spec, tmp_path):
    data = read(spec)
    root = Path(data['runs']['0']['SEG-W1'])
    rows = read(root/'predictions.json'); rows[0]['prediction'] = 1-rows[0]['prediction']
    (root/'predictions.json').write_text(json.dumps(rows))
    out = tmp_path/'out'; out.mkdir()
    with pytest.raises(ValueError):
        analysis.run(type('Args', (), {'input': spec, 'output': out})())


def test_intervals_share_draws_and_max_control():
    rng = np.random.default_rng(0)
    rows = {}
    for label in ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07', 'SEG-W1', 'SEG-W1/iv_swap'):
        rows[label] = [{'task': t, 'gold': 0, 'prediction': int(rng.random() < .4)} for t in ('piqa', 'boolq')
                       for _ in range(50)]
    rows['SEG-W1'] = rows['SEG-05']
    result = analysis.intervals(rows, ['SEG-W1'], repeats=200)
    assert result['SEG-W1']['delta']['boolq'] == [0., 0.]
    assert result['SEG-W1']['delta_gain']['piqa'] == [0., 0.]
