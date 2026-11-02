import builtins
import hashlib
import json
from pathlib import Path

import pytest

import seg_a2_data as a2
from rb_data import confirmation_manifest
from v1_data import derive_seed


def synthetic_splits(count=1100):
    return {
        'metadata': {'synthetic': True},
        'splits': {
            task: {
                'train': [{'id': f'{task}-{i}', 'task': task, 'label': bool(i % 2),
                           'text': f'synthetic {task} {i}'} for i in range(count)],
                'dev': [],
            }
            for task in ('piqa', 'boolq')
        },
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return path


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def make_lock(tmp_path, monkeypatch):
    original = synthetic_splits()
    manifest = confirmation_manifest(original)
    split_path = write_json(tmp_path / 'rb-confirmation.json', manifest)
    monkeypatch.setattr(a2, 'SPLIT_MANIFEST_SHA256', ref(split_path)['sha256'])
    runs = {}
    for seed in a2.SEEDS:
        runs[str(seed)] = {}
        for arm in a2.ARMS:
            root = tmp_path / 'runs' / str(seed) / arm
            manifest_path = write_json(root / 'manifest.json', {
                'contract': a2.CONTRACT, 'status': 'complete', 'arm': arm,
                'seed': seed, 'phase': 'SEG', 'stage': 'train',
                'validator': {'passed': True},
                'metrics': {'source.commit': 'b' * 40,
                            'model_revision': a2.MODEL_REVISION,
                            'tokenizer_revision': a2.MODEL_REVISION},
            })
            predictions_path = write_json(root / 'predictions.json', [])
            config_path = write_json(root / 'adapter' / 'config.json', {'arm': arm})
            adapter_path = root / 'adapter' / 'adapter.pt'
            adapter_path.write_bytes(b'synthetic-checkpoint')
            runs[str(seed)][arm] = {
                'root': str(root.resolve()),
                'manifest': ref(manifest_path),
                'predictions': ref(predictions_path),
                'checkpoint': {'config.json': ref(config_path), 'adapter.pt': ref(adapter_path)},
            }
    lock = {
        'contract': a2.CONTRACT,
        'evaluation_source': 'a' * 40,
        'training_source': 'b' * 40,
        'model_revision': a2.MODEL_REVISION,
        'tokenizer_revision': a2.MODEL_REVISION,
        'split_manifest': ref(split_path),
        'ids_sha256': manifest['ids_sha256'],
        'runs': runs,
        'evaluation': a2.EVALUATION,
        'bootstrap': a2.BOOTSTRAP,
        'derangement': {
            'seed': derive_seed(a2.BASE_SEED, a2.STUDY, 'A2', 'iv_swap'),
            'method': 'within_task_fixed_derangement',
        },
        'preopening_checks': {'passed': True, 'checks': ['synthetic-fixture']},
    }
    lock_path = write_json(tmp_path / 'a2-lock.json', lock)
    monkeypatch.setenv('SR_CODE_COMMIT', 'a' * 40)
    return original, manifest, lock, lock_path


def test_load_a2_selects_manifest_order_and_excludes_other_train_rows():
    original = synthetic_splits()
    manifest = confirmation_manifest(original)
    selected = a2.load_a2(original, manifest)
    for task in ('piqa', 'boolq'):
        train_ids = {row['id'] for row in selected['splits'][task]['train']}
        a2_rows = selected['splits'][task]['a2']
        a2_ids = [row['id'] for row in a2_rows]
        assert a2_ids == manifest['ids'][task]
        assert len(a2_ids) == 1000
        assert not train_ids.intersection(a2_ids)
        assert train_ids | set(a2_ids) == {row['id'] for row in original['splits'][task]['train']}


def test_load_a2_rejects_altered_manifest_or_order():
    original = synthetic_splits()
    manifest = confirmation_manifest(original)
    changed = json.loads(json.dumps(manifest))
    changed['ids']['boolq'][0], changed['ids']['boolq'][1] = (
        changed['ids']['boolq'][1], changed['ids']['boolq'][0])
    with pytest.raises(ValueError, match='regenerated'):
        a2.load_a2(original, changed)


def test_verify_lock_authenticates_pin_and_artifact_files(tmp_path, monkeypatch):
    _, _, _, lock_path = make_lock(tmp_path, monkeypatch)
    digest = ref(lock_path)['sha256']
    verified = a2.verify_lock(lock_path, digest)
    assert verified['contract'] == a2.CONTRACT
    adapter = Path(verified['runs']['3']['SEG-05']['checkpoint']['adapter.pt']['path'])
    adapter.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash/size'):
        a2.verify_lock(lock_path, digest)


def test_verify_lock_rejects_changed_eval_settings_and_checkpoint_ref(tmp_path, monkeypatch):
    _, _, lock, lock_path = make_lock(tmp_path, monkeypatch)
    lock['evaluation']['max_length'] = 513
    write_json(lock_path, lock)
    with pytest.raises(ValueError, match='caller pin'):
        a2.verify_lock(lock_path, '0' * 64)
    lock['evaluation']['max_length'] = 512
    lock['runs']['4']['SEG-06']['checkpoint']['adapter.pt']['sha256'] = '0' * 64
    write_json(lock_path, lock)
    with pytest.raises(ValueError, match='hash/size'):
        a2.verify_lock(lock_path, ref(lock_path)['sha256'])


def test_failed_lock_stops_before_importing_or_reading_source(tmp_path, monkeypatch):
    _, _, _, lock_path = make_lock(tmp_path, monkeypatch)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == 'v1_run':
            raise AssertionError('source preparation imported before lock validation')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    with pytest.raises(ValueError, match='caller pin'):
        a2.read_a2(lock_path, '0' * 64, '/must/not/read', tmp_path / 'out',
                   3, 'SEG-05', 'synthetic hardware', unseal=True,
                   confirmation_decision_id='D-9999')
    assert not (tmp_path / 'out').exists()


def test_read_a2_requires_explicit_unseal_before_any_lock_or_data_access(tmp_path):
    with pytest.raises(ValueError, match='explicit unseal'):
        a2.read_a2(tmp_path / 'missing-lock', '0' * 64, '/must/not/read',
                   tmp_path / 'out', 3, 'SEG-05', 'synthetic hardware')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('decision', [None, '', 'D-1', 'D-99999', 'arbitrary'])
def test_read_a2_requires_recordable_confirmation_decision(tmp_path, decision):
    with pytest.raises(ValueError, match='confirmation decision'):
        a2.read_a2(tmp_path/'missing', '0'*64, '/must/not/read', tmp_path/'out',
            3, 'SEG-05', 'synthetic hardware', unseal=True, confirmation_decision_id=decision)
    assert not (tmp_path/'out').exists()
