"""Locked same-checkpoint E-0025/E-0027 boundary interventions, dev only."""
import argparse
import gc
import os
from pathlib import Path
import time

import torch

from rb1_artifacts import read, reference
from seg_artifacts import validate
from seg_eval import evaluate
from seg_model import make_model, KEEP
from seg_train import tokenizer, original_backbone
from seg_model import contract
from v1_run import prepare
from v5_train import write_json


def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, values, start = False, {}, time.monotonic()
    try:
        _, refs, adapter = validate(args.training_run, args.arm, args.training_source, args.seed)
        tok = tokenizer(); splits = prepare(args.data_dir)
        model = make_model(args.arm, adapter, seed=args.seed).eval()
        with original_backbone(model) as base:
            bm, bp, ba = evaluate(base, 'SEG-00', splits, tok)
        conditions = {'Base_seg': {'metrics': bm, 'predictions': bp, 'audit': ba}}
        for kind in ('normal', 'iv_off', 'iv_swap', 'iv_reset'):
            m, p, audit = evaluate(model, args.arm, splits, tok, kind=kind)
            conditions[kind] = {'metrics': m, 'predictions': p, 'audit': audit}
        if conditions['iv_off']['predictions'] != bp:
            raise ValueError('segmented iv_off differs from same-allocation Base_seg')
        train = read(Path(args.training_run)/'predictions.json')
        values['normal.training_prediction_flips'] = sum(a['prediction'] != b['prediction'] for a, b in zip(train, conditions['normal']['predictions']))
        # D-0018/D-0019: cross-job differences do not cause a rerun/failure.
        values['normal.training_scores_identical'] = train == conditions['normal']['predictions']
        values.update({'off.identical_to_Base_seg': True, 'conditions': 5, 'dev_rows': 2000})
        write_json(out/'conditions.json', conditions)
        write_json(out/'provenance.json', {'training': refs, 'checkpoint_files': read(Path(args.training_run)/'checkpoint.json'),
            'training_source': args.training_source, 'evaluation_source': os.environ['SR_CODE_COMMIT'],
            'donor_phase': 'R1', 'donor_rule': 'RB1 within-task fixed-seed derangement; segment1 only, no candidate or gold access'})
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': contract(args.seed),
            'arm': args.arm, 'seed': args.seed, 'phase': 'SEG', 'stage': 'intervention',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': values,
            'artifacts': [{'path': str(p), 'role': p.name} for p in (out/'conditions.json', out/'provenance.json') if p.exists()],
            'validator': {'name': 'segmented_boundary_interventions', 'passed': passed}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=KEEP, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--training-source', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
