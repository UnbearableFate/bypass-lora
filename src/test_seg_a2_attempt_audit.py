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


def test_audit_requires_every_accounted_attempt_manifest(tmp_path):
    artifacts = tmp_path / 'artifacts'
    attempts, selected = _fixture(artifacts)
    attempts.append('ATT-0099-1')
    with pytest.raises(ValueError, match='manifest.json'):
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
