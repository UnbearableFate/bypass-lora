"""Receipt-only audit of every E-0030 attempt associated with one A2 lock.

This module reads attempt manifests and ``a2-read.jsonl`` receipts only. It
does not open condition predictions, evaluation audits, or any A2 text/labels.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

SEEDS = (3, 4, 5)
CONTRACT = 'E-0030@1'
ATTEMPT_RE = re.compile(r'ATT-\d+-\d+')
DECISION_RE = re.compile(r'D-\d{4}')
MAX_DISCOVERED_ATTEMPTS = 10_000


def _read_json(path):
    return json.loads(Path(path).read_text())


def _ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def _lock_name(manifest):
    metrics = manifest.get('metrics')
    return metrics.get('lock.sha256') if isinstance(metrics, dict) else None


def _parse_receipts(path):
    receipts = []
    if path.exists():
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}: malformed receipt line {line_number}') from exc
            if not isinstance(value, dict):
                raise ValueError(f'{path}: receipt line {line_number} must be an object')
            receipts.append(value)
    return receipts


def _manifest_record(root, attempt_id):
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'accounted attempt {attempt_id} has no manifest.json')
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f'accounted attempt {attempt_id} manifest must be an object')
    receipts_path = root / 'a2-read.jsonl'
    receipts = _parse_receipts(receipts_path)
    return manifest, manifest_path, receipts, receipts_path if receipts_path.is_file() else None


def _audit_record(attempt_id, root, manifest, manifest_path, receipts, receipt_path,
                  lock_sha256, expected_decision):
    metrics = manifest.get('metrics')
    if not isinstance(metrics, dict):
        metrics = {}
    manifest_matches = _lock_name(manifest) == lock_sha256
    matching_receipts = [r for r in receipts if r.get('lock_sha256') == lock_sha256]
    if not manifest_matches and not matching_receipts:
        return None
    if manifest.get('contract') != CONTRACT:
        raise ValueError(f'{attempt_id}: lock-matching manifest contract differs')

    stage = manifest.get('stage')
    analysis_attempt = stage == 'analysis'
    seed = manifest.get('seed')
    if manifest_matches and not analysis_attempt and (type(seed) is not int or seed not in SEEDS):
        raise ValueError(f'{attempt_id}: lock-matching manifest must identify seed 3, 4 or 5')
    if len(matching_receipts) > 1:
        raise ValueError(f'{attempt_id}: multiple receipts name the same A2 lock')
    if matching_receipts:
        receipt = matching_receipts[0]
        receipt_seed = receipt.get('seed')
        if type(receipt_seed) is not int or receipt_seed not in SEEDS:
            raise ValueError(f'{attempt_id}: lock-matching receipt must identify seed 3, 4 or 5')
        if manifest_matches and receipt_seed != seed:
            raise ValueError(f'{attempt_id}: receipt seed differs from lock-matching manifest')
        if not manifest_matches:
            raise ValueError(f'{attempt_id}: receipt names this lock but manifest does not')
        if (manifest.get('stage') != 'a2_evaluation' or
                receipt.get('pbs_job_id') != metrics.get('exec.pbs_job_id') or
                receipt.get('hardware') != metrics.get('hardware') or
                receipt.get('confirmation_decision') != metrics.get('confirmation_decision') or
                receipt.get('opening_protocol_decision') != metrics.get('opening_protocol_decision')):
            raise ValueError(f'{attempt_id}: receipt metadata differs from its evaluation manifest')
        decision = receipt.get('confirmation_decision')
        if not isinstance(decision, str) or DECISION_RE.fullmatch(decision) is None:
            raise ValueError(f'{attempt_id}: receipt confirmation decision is invalid')
        if expected_decision is not None and decision != expected_decision:
            raise ValueError(f'{attempt_id}: receipt confirmation decision differs from the expected decision')
        receipt_time = receipt.get('time_utc')
        if not isinstance(receipt_time, str):
            raise ValueError(f'{attempt_id}: receipt time is missing')
        try:
            datetime.fromisoformat(receipt_time.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f'{attempt_id}: receipt time is malformed') from exc

    opened = manifest.get('opened') is True or metrics.get('a2.opened') is True or bool(matching_receipts)
    if manifest.get('opened') is not None and manifest.get('opened') is not opened:
        raise ValueError(f'{attempt_id}: opened flag conflicts with receipt metadata')
    if metrics.get('a2.opened') is not None and metrics.get('a2.opened') is not opened:
        raise ValueError(f'{attempt_id}: a2.opened metric conflicts with receipt metadata')
    if opened and not matching_receipts:
        raise ValueError(f'{attempt_id}: opened manifest has no matching lock receipt')

    status = manifest.get('status')
    manifest_decision = metrics.get('confirmation_decision')
    if manifest_decision is not None:
        if not isinstance(manifest_decision, str) or DECISION_RE.fullmatch(manifest_decision) is None:
            raise ValueError(f'{attempt_id}: manifest confirmation decision is invalid')
        if expected_decision is not None and manifest_decision != expected_decision:
            raise ValueError(f'{attempt_id}: manifest confirmation decision differs from the expected decision')
    if status not in ('complete', 'partial'):
        raise ValueError(f'{attempt_id}: lock-matching manifest status must be complete or partial')
    failure_reason = metrics.get('failure_reason')
    if failure_reason == 'retry_profile_miss':
        if opened or status != 'partial' or stage != 'a2_evaluation':
            raise ValueError(f'{attempt_id}: retry_profile_miss must be unopened and partial')
    elif stage == 'dev_replay':
        if opened:
            raise ValueError(f'{attempt_id}: dev_replay must never open A2')
    elif analysis_attempt:
        if opened:
            raise ValueError(f'{attempt_id}: analysis attempt must never write an A2 read receipt')
    elif stage != 'a2_evaluation':
        raise ValueError(f'{attempt_id}: unsupported lock-matching attempt stage {stage!r}')

    if analysis_attempt:
        category = 'analysis_unopened'
    elif opened and status == 'complete':
        category = 'complete_opened'
    elif opened:
        category = 'partial_opened'
    elif failure_reason == 'retry_profile_miss':
        category = 'retry_profile_miss'
    elif stage == 'dev_replay':
        category = 'dev_replay'
    else:
        category = 'unopened_partial'
    return {
        'attempt_id': attempt_id,
        'root': str(root.resolve()),
        'seed': seed,
        'status': status,
        'stage': stage,
        'opened': opened,
        'category': category,
        'failure_reason': failure_reason,
        'confirmation_decision': (matching_receipts[0].get('confirmation_decision')
                                  if matching_receipts else manifest_decision),
        'manifest_ref': _ref(manifest_path),
        'receipt_ref': _ref(receipt_path) if receipt_path is not None else None,
        'receipt_count_for_lock': len(matching_receipts),
    }


def audit_attempts(accounted_attempts, evaluations, lock_sha256, *,
                   artifacts_root=Path('/work/xg24i002/x10041/bypass-lora/research/artifacts'),
                   confirmation_decision=None):
    """Validate opened-attempt uniqueness and return receipt metadata only.

    ``evaluations`` maps string seeds ``3``/``4``/``5`` to their selected
    absolute ATT roots. Any matching lock attempt discovered among the bounded
    immediate ATT directories must also be in ``accounted_attempts``.
    """
    if not isinstance(lock_sha256, str) or re.fullmatch(r'[0-9a-f]{64}', lock_sha256) is None:
        raise ValueError('lock_sha256 must be a lowercase 64-hex digest')
    if confirmation_decision is not None and (
            not isinstance(confirmation_decision, str) or
            DECISION_RE.fullmatch(confirmation_decision) is None):
        raise ValueError('confirmation_decision must match D-NNNN')
    if (not isinstance(accounted_attempts, list) or not accounted_attempts or
            any(not isinstance(attempt, str) or ATTEMPT_RE.fullmatch(attempt) is None
                for attempt in accounted_attempts) or
            len(set(accounted_attempts)) != len(accounted_attempts)):
        raise ValueError('accounted_attempts must be unique registered ATT IDs')
    if not isinstance(evaluations, dict) or set(evaluations) != {str(seed) for seed in SEEDS}:
        raise ValueError('evaluations must select one root for seeds 3, 4 and 5')

    artifacts_root = Path(artifacts_root).resolve()
    accounted = set(accounted_attempts)
    selected = {}
    for seed in SEEDS:
        root_value = evaluations[str(seed)]
        if not isinstance(root_value, (str, Path)) or not Path(root_value).is_absolute():
            raise ValueError(f'seed {seed} evaluation root must be an absolute path')
        root = Path(root_value).resolve()
        attempt_id = root.name
        if ATTEMPT_RE.fullmatch(attempt_id) is None or root.parent != artifacts_root:
            raise ValueError(f'seed {seed} evaluation root must be an immediate ATT directory under artifacts_root')
        if attempt_id not in accounted:
            raise ValueError(f'seed {seed} selected evaluation is absent from accounted_attempts')
        selected[seed] = (attempt_id, root)
    if len({attempt_id for attempt_id, _ in selected.values()}) != len(SEEDS):
        raise ValueError('each seed must select a distinct ATT evaluation root')

    if not artifacts_root.is_dir():
        raise ValueError('artifacts_root must be an existing directory')
    discovered_dirs = []
    for path in artifacts_root.iterdir():
        if path.is_dir() and not path.is_symlink() and ATTEMPT_RE.fullmatch(path.name):
            discovered_dirs.append(path.resolve())
            if len(discovered_dirs) > MAX_DISCOVERED_ATTEMPTS:
                raise ValueError('artifacts_root exceeds the bounded immediate ATT audit limit')

    accounted_roots = {}
    for attempt_id in sorted(accounted):
        accounted_roots[attempt_id] = artifacts_root / attempt_id
    for root in discovered_dirs:
        if root.name not in accounted:
            # Read only metadata to distinguish ordinary ATT outputs from a
            # forgotten replay/opening attempt associated with this lock.
            manifest_path = root / 'manifest.json'
            receipts_path = root / 'a2-read.jsonl'
            matching = False
            if manifest_path.is_file():
                candidate = _read_json(manifest_path)
                if not isinstance(candidate, dict):
                    raise ValueError(f'{root.name}: manifest must be an object')
                matching = _lock_name(candidate) == lock_sha256
            if not matching and receipts_path.is_file():
                matching = any(r.get('lock_sha256') == lock_sha256 for r in _parse_receipts(receipts_path))
            if matching:
                raise ValueError(f'unaccounted lock-matching attempt discovered: {root.name}')

    records = {}
    for attempt_id, root in accounted_roots.items():
        manifest, manifest_path, receipts, receipt_path = _manifest_record(root, attempt_id)
        record = _audit_record(attempt_id, root, manifest, manifest_path, receipts,
                               receipt_path, lock_sha256, confirmation_decision)
        if record is not None:
            records[attempt_id] = record

    opened_by_seed = {str(seed): [] for seed in SEEDS}
    matching_attempts_by_seed = {str(seed): [] for seed in SEEDS}
    decisions = set()
    for record in records.values():
        seed = record['seed']
        if seed in SEEDS:
            matching_attempts_by_seed[str(seed)].append(record)
        if record['opened']:
            opened_by_seed[str(seed)].append(record)
        if record['confirmation_decision'] is not None:
            decisions.add(record['confirmation_decision'])

    decisions.discard(None)
    if len(decisions) > 1 or (confirmation_decision is not None and decisions and
                              decisions != {confirmation_decision}):
        raise ValueError('lock-matching opened attempts disagree on confirmation decision')
    resolved_decision = (confirmation_decision if confirmation_decision is not None else
                         next(iter(decisions)) if decisions else None)

    for seed in SEEDS:
        selected_id = selected[seed][0]
        attempts = matching_attempts_by_seed[str(seed)]
        complete_opened = [record for record in attempts
                           if record['status'] == 'complete' and record['opened']]
        if len(complete_opened) != 1 or complete_opened[0]['attempt_id'] != selected_id:
            raise ValueError(f'seed {seed} must have exactly one complete opened attempt equal to selected root')
        for record in attempts:
            if record['opened'] and record['attempt_id'] != selected_id and record['status'] != 'partial':
                raise ValueError(f'seed {seed} non-selected opened attempt must be partial')

    def sorted_records(values):
        return sorted(values, key=lambda rec: rec['attempt_id'])

    accounted_refs = {}
    for attempt_id, root in accounted_roots.items():
        manifest_path = root / 'manifest.json'
        receipt_path = root / 'a2-read.jsonl'
        accounted_refs[attempt_id] = {
            'manifest': _ref(manifest_path),
            'receipt': _ref(receipt_path) if receipt_path.is_file() else None,
        }
    refs = [record['manifest_ref'] for record in records.values()]
    refs.extend(record['receipt_ref'] for record in records.values() if record['receipt_ref'] is not None)
    return {
        'lock_sha256': lock_sha256,
        'confirmation_decision': resolved_decision,
        'opened_attempts_per_seed': {seed: sorted_records(values) for seed, values in opened_by_seed.items()},
        'lock_matching_attempts_per_seed': {seed: sorted_records(values)
                                            for seed, values in matching_attempts_by_seed.items()},
        'other_lock_matching_attempts': sorted_records([r for r in records.values() if r['seed'] not in SEEDS]),
        'selected_attempts': {str(seed): selected[seed][0] for seed in SEEDS},
        'accounted_attempts': sorted(accounted),
        'metadata_refs': refs,
        'accounted_metadata_refs': accounted_refs,
        'discovered_attempts': len(discovered_dirs),
    }
