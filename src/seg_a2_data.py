"""E-0030 A2 lock verification and gated, ID-only split selection.

This module deliberately keeps model evaluation out of the reader.  The only
operation that opens the V1 source data is ``read_a2`` and it is gated by a
verified lock and an explicit ``unseal=True`` argument.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from rb_data import confirmation_manifest

CONTRACT = 'E-0030@1'
OPENING_PROTOCOL_DECISION = 'D-0072'
SEEDS = (3, 4, 5)
ARMS = ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07')
EVALUATION_ARMS = ('SEG-00', *ARMS)
MODEL_REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
SPLIT_MANIFEST_SHA256 = '34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc'
BASE_SEED = 20261101
STUDY = 'residual_la_v21'
EVALUATION = {
    'split': 'a2', 'tasks': ['piqa', 'boolq'], 'examples_per_task': 1000,
    'max_length': 512,
    'conditions': ['SEG-00', 'SEG-04', 'SEG-05', 'SEG-06', 'SEG-07',
                   'SEG-05/iv_swap', 'SEG-05/iv_off'],
}
BOOTSTRAP = {
    'seed': 20261101, 'repeats': 10000,
    'method': 'shared_boolq_rows_recompute_max_linear_quantile',
}


def _read(path):
    return json.loads(Path(path).read_text())


def _reference(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    if not data:
        raise ValueError(f'empty artifact: {path}')
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def _verify_reference(ref, label):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256', 'bytes'}:
        raise ValueError(f'{label} reference has an unexpected schema')
    if not isinstance(ref['path'], str) or not ref['path']:
        raise ValueError(f'{label} reference path is invalid')
    if re.fullmatch(r'[0-9a-f]{64}', str(ref['sha256'])) is None:
        raise ValueError(f'{label} reference SHA256 is invalid')
    if type(ref['bytes']) is not int or ref['bytes'] <= 0:
        raise ValueError(f'{label} reference byte count is invalid')
    path = Path(ref['path']).resolve()
    if not path.is_file() or _reference(path) != ref:
        raise ValueError(f'{label} reference hash/size binding changed')
    return path


def _valid_sha(value, length, label):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{%d}' % length, value) is None:
        raise ValueError(f'{label} must be a lowercase {length}-digit SHA')


def verify_lock(path, sha256):
    """Authenticate the complete E-0030 lock and all pinned input artifacts."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('A2 lock file is missing')
    _valid_sha(sha256, 64, 'expected lock SHA256')
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != sha256:
        raise ValueError('A2 lock SHA256 differs from the caller pin')
    lock = _read(path)
    required = {'contract', 'evaluation_source', 'training_source', 'model_revision',
                'tokenizer_revision', 'split_manifest', 'ids_sha256', 'runs',
                'evaluation', 'bootstrap', 'derangement', 'preopening_checks'}
    if not isinstance(lock, dict) or set(lock) != required:
        raise ValueError('A2 lock schema differs from E-0030@1')
    if lock['contract'] != CONTRACT:
        raise ValueError('A2 lock contract is not E-0030@1')
    _valid_sha(lock['evaluation_source'], 40, 'evaluation_source')
    _valid_sha(lock['training_source'], 40, 'training_source')
    env_commit = os.environ.get('SR_CODE_COMMIT')
    if env_commit is None or lock['evaluation_source'] != env_commit:
        raise ValueError('A2 evaluation source differs from SR_CODE_COMMIT')
    if lock['model_revision'] != MODEL_REVISION or lock['tokenizer_revision'] != MODEL_REVISION:
        raise ValueError('A2 model/tokenizer revision differs from the pinned base revision')

    split_path = _verify_reference(lock['split_manifest'], 'A2 split manifest')
    if lock['split_manifest']['sha256'] != SPLIT_MANIFEST_SHA256:
        raise ValueError('A2 split manifest SHA256 differs from the sealed D-0025 file')
    _valid_sha(lock['ids_sha256'], 64, 'ids_sha256')
    manifest = _read(split_path)
    if (not isinstance(manifest, dict) or manifest.get('sealed') is not True or
            manifest.get('model_evaluated') is not False or
            manifest.get('ids_sha256') != lock['ids_sha256']):
        raise ValueError('A2 split manifest identity differs from the lock')

    if lock['evaluation'] != EVALUATION:
        raise ValueError('A2 evaluation settings differ from E-0030')
    if lock['bootstrap'] != BOOTSTRAP:
        raise ValueError('A2 bootstrap settings differ from E-0030')
    # derive_seed lives in the lightweight data module; no model stack is imported.
    from v1_data import derive_seed
    expected_seed = derive_seed(BASE_SEED, STUDY, 'A2', 'iv_swap')
    if lock['derangement'] != {'seed': expected_seed,
                               'method': 'within_task_fixed_derangement'}:
        raise ValueError('A2 derangement settings differ from E-0030')

    runs = lock['runs']
    if not isinstance(runs, dict) or set(runs) != {str(seed) for seed in SEEDS}:
        raise ValueError('A2 lock must bind all three new seeds')
    for seed in SEEDS:
        by_arm = runs[str(seed)]
        if not isinstance(by_arm, dict) or set(by_arm) != set(ARMS):
            raise ValueError(f'A2 lock seed {seed} must bind the four registered arms')
        for arm in ARMS:
            entry = by_arm[arm]
            if not isinstance(entry, dict) or set(entry) != {
                    'root', 'manifest', 'predictions', 'checkpoint'}:
                raise ValueError(f'{arm} seed {seed} run entry schema is invalid')
            root = Path(entry['root']).resolve()
            if not root.is_dir():
                raise ValueError(f'{arm} seed {seed} run root is missing')
            for field in ('manifest', 'predictions'):
                ref_path = _verify_reference(entry[field], f'{arm} seed {seed} {field}')
                if root not in ref_path.parents:
                    raise ValueError(f'{arm} seed {seed} {field} escapes its run root')
            producer = _read(entry['manifest']['path'])
            expected_identity = {
                'contract': CONTRACT, 'status': 'complete', 'arm': arm,
                'seed': seed, 'phase': 'SEG', 'stage': 'train',
            }
            if (any(producer.get(key) != value for key, value in expected_identity.items()) or
                    producer.get('validator', {}).get('passed') is not True or
                    producer.get('metrics', {}).get('source.commit') != lock['training_source'] or
                    producer.get('metrics', {}).get('model_revision') != MODEL_REVISION or
                    producer.get('metrics', {}).get('tokenizer_revision') != MODEL_REVISION):
                raise ValueError(f'{arm} seed {seed} producer manifest identity differs from the lock')
            checkpoint = entry['checkpoint']
            if not isinstance(checkpoint, dict) or set(checkpoint) != {'config.json', 'adapter.pt'}:
                raise ValueError(f'{arm} seed {seed} checkpoint inventory is incomplete')
            for name, ref in checkpoint.items():
                checkpoint_path = _verify_reference(ref, f'{arm} seed {seed} checkpoint {name}')
                if checkpoint_path.name != name or root not in checkpoint_path.parents:
                    raise ValueError(f'{arm} seed {seed} checkpoint path escapes its run root')

    checks = lock['preopening_checks']
    if not isinstance(checks, dict) or checks.get('passed') is not True:
        raise ValueError('A2 pre-opening checks are not recorded as passed')
    if checks.get('opening_protocol_decision') != OPENING_PROTOCOL_DECISION:
        raise ValueError('A2 lock lacks the D-0072 opening protocol')
    return lock


def load_a2(original_splits, manifest):
    """Return V1 splits with each task's A2 rows in sealed manifest order."""
    expected = confirmation_manifest(original_splits)
    if manifest != expected:
        raise ValueError('A2 manifest differs from regenerated D-0025 selection')
    result = {'splits': {}, 'metadata': dict(original_splits['metadata'])}
    for task in ('piqa', 'boolq'):
        rows = original_splits['splits'][task]['train']
        lookup = {row['id']: row for row in rows}
        ids = manifest['ids'][task]
        if len(ids) != 1000 or len(ids) != len(set(ids)) or not set(ids) <= set(lookup):
            raise ValueError(f'{task} A2 IDs are missing, duplicated, or incomplete')
        result['splits'][task] = dict(original_splits['splits'][task])
        held = set(ids)
        result['splits'][task]['train'] = [row for row in rows if row['id'] not in held]
        result['splits'][task]['a2'] = [lookup[row_id] for row_id in ids]
        if [row['id'] for row in result['splits'][task]['a2']] != ids:
            raise ValueError(f'{task} A2 row order differs from sealed manifest')
    result['metadata']['a2_ids_sha256'] = manifest['ids_sha256']
    return result


def confirmation_decision(value):
    if not isinstance(value, str) or re.fullmatch(r'D-\d{4}', value) is None:
        raise ValueError('A2 opening requires a confirmation decision D-NNNN')
    return value


def read_a2(lock_path, lock_sha256, data_dir, output, seed, arm, hardware, *,
            unseal=False, confirmation_decision_id=None):
    """Verify, log the opening receipt, then load and select the A2 rows.

    The caller must run the per-seed dev gate before calling this function.
    No model or evaluator is imported here.
    """
    if not unseal:
        raise ValueError('A2 access requires explicit unseal=True')
    decision = confirmation_decision(confirmation_decision_id)
    if seed not in SEEDS or arm not in EVALUATION_ARMS:
        raise ValueError('A2 read seed/arm is not registered')
    if not isinstance(hardware, str) or not hardware.strip():
        raise ValueError('A2 read receipt requires the assigned hardware description')
    lock_path = Path(lock_path).resolve()
    lock = verify_lock(lock_path, lock_sha256)
    receipt_path = Path(output).resolve()
    receipt_path.mkdir(parents=True, exist_ok=True)
    receipt = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'pbs_job_id': os.environ.get('PBS_JOBID'),
        'lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        'seed': seed, 'arm': arm, 'hardware': hardware,
        'evaluated_conditions': EVALUATION['conditions'],
        'confirmation_decision': decision,
        'opening_protocol_decision': OPENING_PROTOCOL_DECISION,
    }
    with (receipt_path / 'a2-read.jsonl').open('a') as handle:
        handle.write(json.dumps(receipt, sort_keys=True, separators=(',', ':')) + '\n')

    # Import and source-data access occur only after the verification and receipt.
    from v1_run import prepare
    original_splits = prepare(data_dir)
    manifest = _read(lock['split_manifest']['path'])
    return lock, load_a2(original_splits, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--arm', choices=EVALUATION_ARMS, required=True)
    parser.add_argument('--hardware', required=True)
    parser.add_argument('--confirmation-decision', required=True)
    parser.add_argument('--unseal', action='store_true')
    args = parser.parse_args()
    read_a2(args.lock, args.sha256, args.data_dir, args.output, args.seed,
            args.arm, args.hardware, unseal=args.unseal,
            confirmation_decision_id=args.confirmation_decision)


if __name__ == '__main__':
    main()
