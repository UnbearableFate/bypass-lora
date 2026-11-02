"""PBS-only E-0030 pre-opening audit and lock creation; no A2 field access."""
import argparse
import os
from pathlib import Path
import time

from calibrate import REVISION
from rb1_artifacts import read, reference
from rb_data import BASE_SEED, STUDY
from seg_a2_data import CONTRACT, SEEDS, ARMS, EVALUATION, BOOTSTRAP, SPLIT_MANIFEST_SHA256, verify_lock
from seg_a2_rules import IMPLEMENTATION_CHECKS
from seg_artifacts import validate
from v1_data import derive_seed
from v5_train import write_json


def make_lock(inputs):
    source = os.environ['SR_CODE_COMMIT']
    if inputs['training_source'] != source:
        raise ValueError('training/evaluation/analysis source must be the same reviewed commit')
    checks_ref = inputs['deterministic_checks']; smoke_ref = inputs['smoke']
    for ref in (checks_ref, smoke_ref):
        if reference(ref['path']) != ref:
            raise ValueError('preopening check or smoke reference changed')
        check = read(ref['path'])
        if check.get('status') != 'complete' or check.get('validator', {}).get('passed') is not True:
            raise ValueError('preopening test or smoke job did not pass')
    deterministic = read(checks_ref['path'])
    if (deterministic.get('source_commit') != source or
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
    runs, checks = {}, {}
    for seed in SEEDS:
        roots = inputs['runs'][str(seed)]
        if set(roots) != set(ARMS):
            raise ValueError('exactly four arms per seed required')
        runs[str(seed)] = {}; manifests = {}
        for arm in ARMS:
            root = Path(roots[arm]).resolve()
            manifest, _, _ = validate(root, arm, source, seed)
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
    return {'contract': CONTRACT, 'evaluation_source': source, 'training_source': source,
        'model_revision': REVISION, 'tokenizer_revision': REVISION,
        'split_manifest': split_ref, 'ids_sha256': split['ids_sha256'], 'runs': runs,
        'evaluation': EVALUATION, 'bootstrap': BOOTSTRAP,
        'derangement': {'seed': derive_seed(BASE_SEED, STUDY, 'A2', 'iv_swap'),
                        'method': 'within_task_fixed_derangement'},
        'preopening_checks': {'passed': True, 'checks': checks, 'deterministic': checks_ref, 'smoke': smoke_ref,
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
