"""One locked E-0030 seed job: dev reproduction gate, then A2 evaluation."""
import argparse
import gc
import os
from pathlib import Path
import time

import torch

from rb1_artifacts import read, reference
from seg_a2_data import ARMS, SEEDS, CONTRACT, verify_lock, read_a2, confirmation_decision
from seg_a2_predata import prepare_preopening
from seg_artifacts import validate, validate_rows
from seg_eval import evaluate
from seg_data import encode_segments
from seg_model import make_model
from seg_train import tokenizer
from v5_train import write_json


def dev_gate(actual, expected, hardware, training_hardware):
    """Same slice: exact rows; other slice: <=10 class changes on paired rows."""
    validate_rows(expected)
    validate_rows(actual, [(r['task'], r['id'], r['gold']) for r in expected])
    if not hardware or not training_hardware:
        raise ValueError('dev gate requires measured hardware profiles')
    flips = sum(a['prediction'] != b['prediction'] for a, b in zip(actual, expected))
    same = hardware == training_hardware
    identical = actual == expected
    return {'passed': identical if same else flips <= 10,
            'same_hardware': same, 'prediction_mismatches': flips,
            'rows_identical': identical, 'hardware': hardware,
            'training_hardware': training_hardware}


class A2Unencodable(ValueError):
    """A2 was read but no A2 model score has been computed."""


def preencode_a2(splits, tok, output):
    counts = {t: 0 for t in ('piqa', 'boolq')}
    for task in counts:
        for row in splits['splits'][task]['a2']:
            try:
                encode_segments(tok, row)
            except ValueError as exc:
                write_json(output/'a2-encodability.json', {'passed': False,
                    'status': 'unencodable_no_model_scored', 'task': task, 'id': row['id'],
                    'reason': str(exc), 'checked': counts})
                raise A2Unencodable('A2 unencodable; no A2 model scored') from exc
            counts[task] += 1
    write_json(output/'a2-encodability.json', {'passed': True, 'checked': counts,
                                            'status': 'encodable_no_model_scored_yet'})


def evaluate_a2_conditions(lock, splits, tok, adapters, seed, output):
    # Complete the deterministic encoding check before loading any A2 model.
    preencode_a2(splits, tok, output)
    conditions = {}
    for label in lock['evaluation']['conditions']:
        arm, _, kind = label.partition('/')
        model = (make_model('SEG-00') if arm == 'SEG-00' else
                 make_model(arm, adapters[arm], seed=seed)).eval()
        m, p, audit = evaluate(model, arm, splits, tok, kind=kind or 'normal', split_name='a2', donor_phase='A2')
        conditions[label] = {'metrics': m, 'predictions': p, 'audit': audit}
        write_json(output/'conditions.json', conditions)
        del model; gc.collect(); torch.cuda.empty_cache()
    return conditions


def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, values, start = False, {}, time.monotonic()
    try:
        if not args.unseal:
            raise ValueError('A2 evaluation requires --unseal after coordinator confirmation')
        lock = verify_lock(args.lock, args.lock_sha256)
        decision = confirmation_decision(args.confirmation_decision)
        values['confirmation_decision'] = decision
        hardware = torch.cuda.get_device_name()
        entries = lock['runs'][str(args.seed)]
        manifests, adapters = {}, {}
        for arm in ARMS:
            manifests[arm], _, adapters[arm] = validate(entries[arm]['root'], arm, lock['training_source'], args.seed)
        tok = tokenizer()
        dev = prepare_preopening(args.data_dir, Path(lock['split_manifest']['path']))
        model = make_model('SEG-05', adapters['SEG-05'], seed=args.seed).eval()
        dm, dp, da = evaluate(model, 'SEG-05', dev, tok)
        gate = dev_gate(dp, read(entries['SEG-05']['predictions']['path']), hardware,
                        manifests['SEG-05']['metrics']['hardware'])
        write_json(out/'dev-reproduction.json', {'check': gate, 'metrics': dm, 'predictions': dp, 'audit': da})
        values.update({'hardware': hardware, 'exec.pbs_job_id': os.environ.get('PBS_JOBID'),
                      'lock.sha256': args.lock_sha256, 'path.dev_gate': gate,
                      'source.commit': os.environ['SR_CODE_COMMIT']})
        if not gate['passed']:
            values['failure_reason'] = 'dev_reproduction_gate_failed'
            raise ValueError('pre-opening dev reproduction gate failed; A2 remains unopened by this job')
        del model; gc.collect(); torch.cuda.empty_cache()
        _, a2 = read_a2(args.lock, args.lock_sha256, args.data_dir, out, args.seed,
                        'SEG-05', hardware, unseal=True, confirmation_decision_id=decision)
        try:
            conditions = evaluate_a2_conditions(lock, a2, tok, adapters, args.seed, out)
        except A2Unencodable:
            values['failure_reason'] = 'a2_unencodable_no_model_scored'
            raise
        off_ok = conditions['SEG-05/iv_off']['predictions'] == conditions['SEG-00']['predictions']
        values.update({'off.identical_to_Base_seg': off_ok, 'conditions': len(conditions), 'a2_rows': 2000})
        write_json(out/'provenance.json', {'lock': reference(args.lock), 'seed': args.seed,
            'training_source': lock['training_source'], 'evaluation_source': os.environ['SR_CODE_COMMIT'],
            'inputs': entries, 'hardware': hardware, 'job_id': os.environ['PBS_JOBID'],
            'confirmation_decision': decision})
        if not off_ok:
            raise ValueError('A2 iv_off differs from same-job Base_seg; A2 results invalid')
        passed = True
    finally:
        values['a2.opened'] = (out/'a2-read.jsonl').exists()
        values['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': CONTRACT,
            'arm': 'A2', 'seed': args.seed, 'stage': 'a2_evaluation', 'metrics': values,
            'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'artifacts': [{'path': str(out/name), 'role': name} for name in
                          ('dev-reproduction.json', 'conditions.json', 'provenance.json', 'a2-read.jsonl',
                           'a2-encodability.json')
                          if (out/name).is_file()],
            'validator': {'name': 'segmented_locked_a2_evaluation', 'passed': passed}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--lock-sha256', required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--unseal', action='store_true')
    p.add_argument('--confirmation-decision', required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
