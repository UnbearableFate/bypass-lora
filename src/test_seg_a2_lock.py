import hashlib
import json
from pathlib import Path

import pytest

import seg_a2_lock as lock_module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return path


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def lock_inputs(tmp_path, monkeypatch):
    evaluation_source, training_source = 'b' * 40, 'a' * 40
    monkeypatch.setenv('SR_CODE_COMMIT', evaluation_source)

    audit_root = tmp_path / 'e0013' / 'ATT-0129-1'
    audit_manifest = write_json(audit_root / 'manifest.json', {
        'status': 'complete', 'contract': 'E-0013@3', 'phase': 'train',
        'metrics': {'data.rows_per_epoch': 19443, 'train.examples': 58329},
        'validator': {'name': 'v5_runtime', 'passed': True},
    })
    audit_excluded = write_json(audit_root / 'excluded-train.json', [{
        'task': 'piqa', 'id': 5528, 'candidate_lengths': [537, 537],
        'reason': 'a complete candidate exceeds max_length',
    }])
    monkeypatch.setattr(lock_module, 'E0013_AUDIT_ROOT', audit_root)
    monkeypatch.setattr(lock_module, 'E0013_MANIFEST_SHA256', ref(audit_manifest)['sha256'])
    monkeypatch.setattr(lock_module, 'E0013_EXCLUDED_SHA256', ref(audit_excluded)['sha256'])
    monkeypatch.setattr(lock_module, 'E0013_MANIFEST_BYTES', audit_manifest.stat().st_size)
    monkeypatch.setattr(lock_module, 'E0013_EXCLUDED_BYTES', audit_excluded.stat().st_size)

    deterministic = write_json(tmp_path / 'deterministic.json', {
        'status': 'complete', 'source_commit': evaluation_source,
        'metrics': {'no_channel_identity': True, 'no_channel_seeds': [3, 4, 5]},
        'validator': {'passed': True},
    })
    baseline_excluded = write_json(tmp_path / 'baseline-excluded.json', [
        {'task': 'piqa', 'id': 5528, 'candidate_lengths': [537, 537],
         'reason': 'a complete candidate exceeds max_length'},
    ])
    smoke = write_json(tmp_path / 'smoke.json', {
        'status': 'complete', 'validator': {'passed': True},
        'metrics': {'preopening.loader_equivalence_seeds': [0, 1, 2],
                    'preopening.excluded_audit_reference': ref(baseline_excluded)},
    })

    split = write_json(tmp_path / 'rb-confirmation.json', {
        'ids': {'piqa': [1, 2], 'boolq': ['b1', 'b2']},
        'ids_sha256': 'c' * 64,
    })
    monkeypatch.setattr(lock_module, 'SPLIT_MANIFEST_SHA256', ref(split)['sha256'])
    runs = {}
    calls = []
    for seed in lock_module.SEEDS:
        runs[str(seed)] = {}
        for arm in lock_module.ARMS:
            root = tmp_path / 'runs' / str(seed) / arm
            write_json(root / 'seams-train.json', [])
            write_json(root / 'excluded-train.json', json.loads(baseline_excluded.read_text()))
            write_json(root / 'manifest.json', {'seed': seed, 'arm': arm})
            write_json(root / 'predictions.json', [])
            write_json(root / 'checkpoint.json', {})
            runs[str(seed)][arm] = str(root)

    def validate(root, arm, source, seed):
        calls.append((arm, source, seed))
        initial = 'd' * 64 if arm == 'SEG-04' else 'e' * 64
        return {'metrics': {'initial.adapter_sha256': initial}}, {}, Path(root)

    monkeypatch.setattr(lock_module, 'validate', validate)
    inputs = {
        'training_source': training_source,
        'deterministic_checks': ref(deterministic), 'smoke': ref(smoke),
        'split_manifest': str(split), 'runs': runs,
    }
    return inputs, calls, deterministic, evaluation_source, training_source


def test_make_lock_separates_training_and_evaluation_sources(tmp_path, monkeypatch):
    inputs, calls, _, evaluation_source, training_source = lock_inputs(tmp_path, monkeypatch)
    lock = lock_module.make_lock(inputs)
    assert lock['training_source'] == training_source
    assert lock['evaluation_source'] == evaluation_source
    assert calls and {source for _, source, _ in calls} == {training_source}
    assert lock['preopening_checks']['e0013_full_v1_train_encodability'] == {
        'attempt': 'ATT-0129-1',
        'manifest': ref(lock_module.E0013_AUDIT_ROOT / 'manifest.json'),
        'excluded_train': ref(lock_module.E0013_AUDIT_ROOT / 'excluded-train.json'),
        'contract': 'E-0013@3',
        'validator': {'name': 'v5_runtime', 'passed': True},
        'data_rows_per_epoch': 19443,
        'excluded_rows': lock_module.E0013_EXCLUSION,
        'excluded_piqa_5528_not_in_a2': True,
    }


def test_deterministic_check_must_bind_to_evaluation_source(tmp_path, monkeypatch):
    inputs, calls, deterministic_path, evaluation_source, training_source = lock_inputs(
        tmp_path, monkeypatch)
    write_json(deterministic_path, {
        'status': 'complete', 'source_commit': training_source,
        'metrics': {'no_channel_identity': True, 'no_channel_seeds': [3, 4, 5]},
        'validator': {'passed': True},
    })
    inputs['deterministic_checks'] = ref(deterministic_path)
    with pytest.raises(ValueError, match='final reviewed source'):
        lock_module.make_lock(inputs)
    assert calls == []
    assert evaluation_source != training_source
