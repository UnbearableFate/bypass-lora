"""Synthetic metadata-only receipt audit tests for E-0030 attempts."""
import json

import pytest

from seg_a2_attempt_audit import audit_attempts


LOCK_SHA = 'a' * 64
DECISION = 'D-1234'


def _write_attempt(root, seed, *, status='complete', opened=True, stage='a2_evaluation',
                   failure_reason=None):
    root.mkdir(parents=True, exist_ok=True)
    metrics = {
        'lock.sha256': LOCK_SHA,
        'exec.pbs_job_id': f'{seed}.server',
        'hardware': 'MIG-1g',
        'a2.opened': opened,
        'confirmation_decision': DECISION if stage == 'a2_evaluation' else None,
        'opening_protocol_decision': 'D-0072',
    }
    if failure_reason:
        metrics['failure_reason'] = failure_reason
    manifest = {
        'contract': 'E-0030@1', 'status': status, 'seed': seed, 'stage': stage,
        'opened': opened, 'metrics': metrics,
    }
    (root / 'manifest.json').write_text(json.dumps(manifest))
    if opened:
        receipt = {
            'lock_sha256': LOCK_SHA, 'seed': seed, 'arm': 'SEG-05',
            'pbs_job_id': f'{seed}.server', 'hardware': 'MIG-1g',
            'confirmation_decision': DECISION,
            'opening_protocol_decision': 'D-0072',
            'time_utc': '2026-11-03T00:00:00+00:00',
        }
        (root / 'a2-read.jsonl').write_text(json.dumps(receipt) + '\n')


def _fixture(root):
    root.mkdir()
    selected = {}
    attempts = []
    for seed in (3, 4, 5):
        attempt = f'ATT-{seed:04}-1'
        _write_attempt(root / attempt, seed)
        selected[str(seed)] = str(root / attempt)
        attempts.append(attempt)
    return attempts, selected


def test_audit_rejects_duplicate_complete_opened_attempt_for_a_seed(tmp_path):
    attempts, selected = _fixture(tmp_path / 'artifacts')
    duplicate = 'ATT-0039-1'
    _write_attempt(tmp_path / 'artifacts' / duplicate, 3)
    attempts.append(duplicate)
    with pytest.raises(ValueError, match='exactly one complete opened'):
        audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=tmp_path / 'artifacts',
                       confirmation_decision=DECISION)


def test_audit_accepts_partial_opened_retry_and_lists_it(tmp_path):
    attempts, selected = _fixture(tmp_path / 'artifacts')
    partial = 'ATT-0038-1'
    _write_attempt(tmp_path / 'artifacts' / partial, 3, status='partial', opened=True)
    attempts.append(partial)
    report = audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=tmp_path / 'artifacts',
                            confirmation_decision=DECISION)
    opened = report['opened_attempts_per_seed']['3']
    assert {entry['attempt_id'] for entry in opened} == {'ATT-0003-1', partial}
    assert sum(entry['category'] == 'complete_opened' for entry in opened) == 1
    assert sum(entry['category'] == 'partial_opened' for entry in opened) == 1
    assert partial in report['accounted_attempts']
    assert report['accounted_metadata_refs'][partial]['receipt'] is not None


@pytest.mark.parametrize('stage,failure_reason', [
    ('dev_replay', None),
    ('a2_evaluation', 'retry_profile_miss'),
])
def test_audit_rejects_unaccounted_replay_or_profile_miss(tmp_path, stage, failure_reason):
    attempts, selected = _fixture(tmp_path / 'artifacts')
    unaccounted = 'ATT-0040-1'
    _write_attempt(tmp_path / 'artifacts' / unaccounted, 4, status='partial', opened=False,
                   stage=stage, failure_reason=failure_reason)
    with pytest.raises(ValueError, match='unaccounted lock-matching attempt'):
        audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=tmp_path / 'artifacts',
                       confirmation_decision=DECISION)


@pytest.mark.parametrize('directory_exists', [False, True])
def test_audit_accounts_for_unopened_attempt_without_manifest(tmp_path, directory_exists):
    artifacts = tmp_path / 'artifacts'
    attempts, selected = _fixture(artifacts)
    missing = 'ATT-0099-1'
    if directory_exists:
        (artifacts/missing).mkdir()
    attempts.append(missing)
    report = audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                            confirmation_decision=DECISION)
    record = report['no_manifest_attempts'][0]
    assert record['category'] == 'no_manifest_unopened'
    assert record['seed'] is None and record['opened'] is False
    assert record['lock_associated'] is False
    assert report['accounted_metadata_refs'][missing] == {'manifest': None, 'receipt': None}


def _receipt_only(root, seed):
    _write_attempt(root, seed, status='partial')
    (root/'manifest.json').unlink()


def test_receipt_only_attempt_is_partial_and_cannot_replace_complete_result(tmp_path):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    partial = 'ATT-0097-1'
    _receipt_only(artifacts/partial, 3)
    attempts.append(partial)
    report = audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                            confirmation_decision=DECISION)
    record = next(r for r in report['opened_attempts_per_seed']['3'] if r['attempt_id'] == partial)
    assert record['category'] == 'partial_opened_no_manifest'
    assert record['status'] is None and record['opened'] is True
    assert record['manifest_ref'] is None
    assert report['accounted_metadata_refs'][partial]['manifest'] is None
    assert report['accounted_metadata_refs'][partial]['receipt'] is not None
    assert report['selected_attempts']['3'] == 'ATT-0003-1'


def test_receipt_only_selected_attempt_still_needs_a_complete_result(tmp_path):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    (artifacts/'ATT-0003-1'/'manifest.json').unlink()
    with pytest.raises(ValueError, match='exactly one complete opened'):
        audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                       confirmation_decision=DECISION)


def test_unaccounted_receipt_only_attempt_is_rejected(tmp_path):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    _receipt_only(artifacts/'ATT-0097-1', 3)
    with pytest.raises(ValueError, match='unaccounted lock-matching attempt'):
        audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                       confirmation_decision=DECISION)


def test_unaccounted_unattributable_truncated_receipt_cannot_be_ignored(tmp_path):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    root = artifacts/'ATT-0097-1'; root.mkdir()
    (root/'a2-read.jsonl').write_text('{"confirmation_decision": "D-1234", "hardware": "NVIDIA')
    with pytest.raises(ValueError, match='unaccounted lock-matching attempt'):
        audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                       confirmation_decision=DECISION)


@pytest.mark.parametrize('seed_recoverable', [True, False])
def test_truncated_receipt_is_conservatively_opened_without_inventing_fields(tmp_path, seed_recoverable):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    partial = 'ATT-0097-1'
    root = artifacts/partial
    _receipt_only(root, 3)
    path = root/'a2-read.jsonl'
    if seed_recoverable:
        full = json.dumps(json.loads(path.read_text()), sort_keys=True)
        prefix = full[:full.index('"time_utc"')]+ '"time_utc": "2026-11'
    else:
        prefix = '{"confirmation_decision": "D-1234", "hardware": "NVIDIA'
    path.write_text(prefix)
    attempts.append(partial)
    report = audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                            confirmation_decision=DECISION)
    record = next(r for r in report['no_manifest_attempts'] if r['attempt_id'] == partial)
    assert record['opened'] is True and record['receipt_truncated'] is True
    assert record['category'] == 'partial_opened_no_manifest'
    assert record['seed'] == (3 if seed_recoverable else None)
    if not seed_recoverable:
        assert report['opened_attempts_unknown_seed'][0]['attempt_id'] == partial
    # Even an unattributed partial opening cannot become the selected result.
    assert report['selected_attempts']['3'] == 'ATT-0003-1'


@pytest.mark.parametrize('field,value', [('seed', 6), ('confirmation_decision', 'D-9999'),
                                       ('opening_protocol_decision', 'D-0000'), ('time_utc', 'bad-time')])
def test_receipt_only_attempt_still_validates_known_metadata(tmp_path, field, value):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    partial = 'ATT-0097-1'
    root = artifacts/partial
    _receipt_only(root, 3)
    path = root/'a2-read.jsonl'
    receipt = json.loads(path.read_text()); receipt[field] = value
    path.write_text(json.dumps(receipt)+'\n')
    attempts.append(partial)
    with pytest.raises(ValueError):
        audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                       confirmation_decision=DECISION)


def test_complete_manifest_does_not_allow_a_truncated_receipt(tmp_path):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    (artifacts/'ATT-0003-1'/'a2-read.jsonl').write_text('{"seed": 3,')
    with pytest.raises(ValueError, match='malformed receipt'):
        audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                       confirmation_decision=DECISION)


def test_only_the_last_manifestless_receipt_line_may_be_truncated(tmp_path):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    partial = 'ATT-0097-1'
    root = artifacts/partial
    _receipt_only(root, 3)
    path = root/'a2-read.jsonl'
    path.write_text('{"seed": 3,\n'+path.read_text())
    attempts.append(partial)
    with pytest.raises(ValueError, match='malformed receipt'):
        audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                       confirmation_decision=DECISION)


def test_prior_failed_analysis_is_accounted_without_counting_as_a_seed_opening(tmp_path):
    artifacts = tmp_path/'artifacts'
    attempts, selected = _fixture(artifacts)
    attempt = 'ATT-0098-1'
    root = artifacts/attempt; root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'contract': 'E-0030@1',
        'status': 'partial', 'stage': 'analysis', 'metrics': {'lock.sha256': LOCK_SHA}}))
    attempts.append(attempt)
    report = audit_attempts(attempts, selected, LOCK_SHA, artifacts_root=artifacts,
                            confirmation_decision=DECISION)
    assert report['other_lock_matching_attempts'][0]['category'] == 'analysis_unopened'
    assert all(len(rows) == 1 for rows in report['opened_attempts_per_seed'].values())
