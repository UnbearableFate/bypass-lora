"""PBS-only E-0030 pre-opening audit and lock creation; no A2 field access."""
import argparse
import os
from pathlib import Path
import time

from calibrate import REVISION
from rb1_artifacts import read, reference
from rb_data import BASE_SEED, STUDY
from seg_a2_data import (CONTRACT, SEEDS, ARMS, EVALUATION, BOOTSTRAP, SPLIT_MANIFEST_SHA256,
                         OPENING_PROTOCOL_DECISION, verify_lock)
from seg_a2_rules import IMPLEMENTATION_CHECKS
from seg_artifacts import validate
from v1_data import derive_seed
from v5_train import write_json

E0013_AUDIT_ROOT = Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0129-1')
E0013_MANIFEST_SHA256 = '38f4c152de6735c16f3c04cb9790c04bb8781a149980f68b3fc3c00863d110f3'
E0013_EXCLUDED_SHA256 = 'b67a489da0a81f8c6c3fcfd112182c9a0e77df5fd468c6d1547166fec83780b5'
E0013_MANIFEST_BYTES = 3180
E0013_EXCLUDED_BYTES = 159
E0013_EXCLUSION = [{'task': 'piqa', 'id': 5528, 'candidate_lengths': [537, 537],
                    'reason': 'a complete candidate exceeds max_length'}]


def _e0013_preopening_evidence(split):
    """Pin the earlier full-V1-train encodability audit using metadata only."""
    manifest_path = E0013_AUDIT_ROOT / 'manifest.json'
    excluded_path = E0013_AUDIT_ROOT / 'excluded-train.json'
    for path, size in ((manifest_path, E0013_MANIFEST_BYTES),
                       (excluded_path, E0013_EXCLUDED_BYTES)):
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f'E-0013 audit artifact name/size differs: {path.name}')
    manifest_ref = reference(manifest_path)
    excluded_ref = reference(excluded_path)
    if manifest_ref['sha256'] != E0013_MANIFEST_SHA256 or excluded_ref['sha256'] != E0013_EXCLUDED_SHA256:
        raise ValueError('E-0013 full-train audit artifacts differ from their pins')
    producer = read(manifest_path)
    excluded = read(excluded_path)
    if (producer.get('status') != 'complete' or producer.get('contract') != 'E-0013@3' or
            producer.get('phase') != 'train' or
            producer.get('validator') != {'name': 'v5_runtime', 'passed': True} or
            producer.get('metrics', {}).get('data.rows_per_epoch') != 19443 or
            producer.get('metrics', {}).get('train.examples') != 58329):
        raise ValueError('E-0013 full-V1-train encodability audit did not pass its producer checks')
    if excluded != E0013_EXCLUSION:
        raise ValueError('E-0013 full-V1-train exclusion list differs from the pinned single row')
    if 5528 in split['ids']['piqa']:
        raise ValueError('the E-0013 excluded PIQA row appears in sealed A2 IDs')
    return {'attempt': 'ATT-0129-1', 'manifest': manifest_ref,
            'excluded_train': excluded_ref, 'contract': 'E-0013@3',
            'validator': {'name': 'v5_runtime', 'passed': True},
            'data_rows_per_epoch': 19443, 'excluded_rows': E0013_EXCLUSION,
            'excluded_piqa_5528_not_in_a2': True}


def make_lock(inputs):
    evaluation_source = os.environ['SR_CODE_COMMIT']
    training_source = inputs['training_source']
    for name, commit in (('training_source', training_source),
                         ('SR_CODE_COMMIT', evaluation_source)):
        if not isinstance(commit, str) or len(commit) != 40 or any(
                c not in '0123456789abcdef' for c in commit):
            raise ValueError(f'{name} must be a full lowercase commit SHA')
    checks_ref = inputs['deterministic_checks']; smoke_ref = inputs['smoke']
    for ref in (checks_ref, smoke_ref):
        if reference(ref['path']) != ref:
            raise ValueError('preopening check or smoke reference changed')
        check = read(ref['path'])
        if check.get('status') != 'complete' or check.get('validator', {}).get('passed') is not True:
            raise ValueError('preopening test or smoke job did not pass')
    deterministic = read(checks_ref['path'])
    if (deterministic.get('source_commit') != evaluation_source or
            deterministic['metrics'].get('no_channel_identity') is not True or
            deterministic['metrics'].get('no_channel_seeds') != list(SEEDS)):
        raise ValueError('no-channel checks are not at the final reviewed source for all seeds')
    smoke = read(smoke_ref['path'])
    if smoke['metrics'].get('preopening.loader_equivalence_seeds') != [0, 1, 2]:
        raise ValueError('masking loader lacks legacy plan/exclusion equivalence checks')
    exclusion_ref = smoke['metrics']['preopening.excluded_audit_reference']
    if reference(exclusion_ref['path']) != exclusion_ref:
        raise ValueError('registered exclusion audit reference changed')
    split_ref = reference(inputs['split_manifest'])
    if split_ref['sha256'] != SPLIT_MANIFEST_SHA256:
        raise ValueError('A2 IDs file differs from the sealed manifest')
    split = read(split_ref['path'])
    e0013_audit = _e0013_preopening_evidence(split)
    runs, checks = {}, {}
    for seed in SEEDS:
        roots = inputs['runs'][str(seed)]
        if set(roots) != set(ARMS):
            raise ValueError('exactly four arms per seed required')
        runs[str(seed)] = {}; manifests = {}
        for arm in ARMS:
            root = Path(roots[arm]).resolve()
            manifest, _, _ = validate(root, arm, training_source, seed)
            manifests[arm] = manifest
            train_ids = {(r['task'], r['id']) for r in read(root/'seams-train.json')}
            held = {(t, i) for t, ids in split['ids'].items() for i in ids}
            if train_ids & held:
                raise ValueError('training rows overlap held A2 IDs')
            if read(root/'excluded-train.json') != read(smoke['metrics']['preopening.excluded_audit_reference']['path']):
                raise ValueError('training exclusion audit differs from E-0027')
            runs[str(seed)][arm] = {'root': str(root), 'manifest': reference(root/'manifest.json'),
                'predictions': reference(root/'predictions.json'), 'checkpoint': read(root/'checkpoint.json')}
        initial = {a: m['metrics']['initial.adapter_sha256'] for a, m in manifests.items()}
        check = {k: True for k in IMPLEMENTATION_CHECKS}
        check['initial_parameters_identical'] = initial['SEG-05'] == initial['SEG-06'] == initial['SEG-07']
        if not all(check.values()):
            raise ValueError('initial parameter identity failed')
        checks[str(seed)] = check
    return {'contract': CONTRACT, 'evaluation_source': evaluation_source, 'training_source': training_source,
        'model_revision': REVISION, 'tokenizer_revision': REVISION,
        'split_manifest': split_ref, 'ids_sha256': split['ids_sha256'], 'runs': runs,
        'evaluation': EVALUATION, 'bootstrap': BOOTSTRAP,
        'derangement': {'seed': derive_seed(BASE_SEED, STUDY, 'A2', 'iv_swap'),
                        'method': 'within_task_fixed_derangement'},
        'preopening_checks': {'passed': True, 'checks': checks, 'deterministic': checks_ref, 'smoke': smoke_ref,
            'opening_protocol_decision': OPENING_PROTOCOL_DECISION,
            'opening_failure_policy': {
                'first_attempt_slice_restricted': False,
                'cross_profile_gate_failure': 'engineering retry unchanged lock/source/config/checkpoints/rules/confirmation decision on training profile',
                'consecutive_profile_miss_limit': 5,
                'same_profile_gate_failure': 'cancel unstarted peer jobs; after batch ends audit all receipts; repair/review/relock only if none exist, otherwise invalid A2 and no hypothesis conclusion',
                'unencodable_a2': 'no A2 scores; stop without repair or relock; coordinator reports user',
                'before_confirmation': 'run dev-replay-only for every seed under this lock and source; report profiles and mismatches'},
            'e0013_full_v1_train_encodability': e0013_audit,
            'rules': {'2': 'same slice exact dev rows; other slice <=10 class mismatches; A2 off equals same-job Base',
                      '3': 'gain_s=Keep-max(Reset,T0); lora_s=Keep-LoRA; swap_s=Keep-swap',
                      '4': '10000 paired BoolQ row draws shared by all arms/seeds; recompute maximum; linear quantile',
                      '5': 'G>0 and lower95>0 -> supported; G<=0 -> challenged; otherwise inconclusive',
                      '6': 'only after primary support: L>0 and lower95>0 confirms LoRA comparison'}}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, start = False, time.monotonic()
    try:
        lock = make_lock(read(args.input))
        write_json(out/'a2-lock.json', lock)
        sha = reference(out/'a2-lock.json')['sha256']
        verify_lock(out/'a2-lock.json', sha)
        (out/'a2-lock.sha256').write_text(sha+'\n')
        passed = True
    finally:
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': CONTRACT,
            'stage': 'preopening_lock', 'source_commit': os.environ['SR_CODE_COMMIT'],
            'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'metrics': {'preopening.passed': passed, 'confirmation_evaluated': False, 'elapsed_s': time.monotonic()-start},
            'artifacts': [{'path': str(f), 'role': f.name} for f in out.iterdir() if f.is_file() and f.name != 'manifest.json'],
            'validator': {'name': 'E0030_preopening_lock', 'passed': passed}})


if __name__ == '__main__':
    main()
