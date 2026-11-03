"""E-0031 zero-training checkpoint reevaluation and H-QVU frozen interventions."""
import argparse
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb1_artifacts import read, reference, validate_formal_run
from rb1_intervene import evaluate_history
from rb_eta_select import SPLIT_SHA256, validate_calibration
from rb_tap_identity import prediction_keys
from rb_train import load_data, make_model
from v1_run import evaluate
from v5_train import write_json

LEGACY = {'RB1-01':'ATT-0239-1', 'RB1-02':'ATT-0240-1',
          'RB1-07':'ATT-0245-1', 'RB1-09':'ATT-0247-1'}


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    passed, metrics = False, {}
    tick = time.monotonic()
    try:
        if reference(args.split_manifest)['sha256'] != SPLIT_SHA256:
            raise ValueError('reevaluation split differs from RB1')
        source = args.training_source if args.arm == 'H-QVU' else None
        root = args.training_run.resolve()
        if args.arm in LEGACY and root.name != LEGACY[args.arm]:
            raise ValueError('E-0031 pins each legacy checkpoint producer')
        origin, refs, adapter = validate_formal_run(root, args.arm, source)
        if origin['metrics']['split_manifest.sha256'] != SPLIT_SHA256:
            raise ValueError('checkpoint trained on a different split')
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        tok.pad_token_id = tok.eos_token_id
        splits, _ = load_data(args.data_dir, args.split_manifest, tok)
        model = make_model(args.arm, checkpoint=adapter).eval()
        metrics.update({'source.commit': os.environ['SR_CODE_COMMIT'], 'model_revision': REVISION,
            'tokenizer_revision': REVISION, 'split_manifest.sha256': SPLIT_SHA256,
            'hardware': torch.cuda.get_device_name(), 'exec.pbs_job_id': os.environ['PBS_JOBID'],
            'environment.lock_sha256': reference(Path(__file__).resolve().parents[1]/'uv.lock')['sha256']})
        if args.arm in LEGACY:
            start = time.monotonic()
            values, rows = evaluate(model, splits, tok, args.eval_limit, split_names=('dev',))
            metrics.update(values)
            metrics['dev_eval.seconds'] = time.monotonic()-start
            pinned = read(root/'predictions.json')
            if args.eval_limit:
                pinned = [r for task in ('piqa','boolq')
                    for r in [p for p in pinned if p['task'] == task][:args.eval_limit]]
            identity = lambda rs: [(r['task'],r['id'],r['gold']) for r in rs]
            if identity(rows) != identity(pinned):
                raise ValueError('reevaluation row membership differs from pinned dev')
            flips = sum(a['prediction'] != b['prediction'] for a,b in zip(rows,pinned))
            metrics['compatibility'] = {'predictions_equal': flips == 0, 'flips': flips,
                'same_profile': metrics['hardware'] == origin['metrics']['hardware'],
                'training_hardware': origin['metrics']['hardware'], 'evaluation_hardware': metrics['hardware']}
            write_json(out/'predictions.json', rows)
            start = time.monotonic()
            values, rows = evaluate(model, splits, tok, args.eval_limit, split_names=('calibration',))
            metrics.update(values)
            metrics['calibration.seconds'] = time.monotonic()-start
            if not args.eval_limit:
                validate_calibration(rows, metrics)
            write_json(out/'calibration-predictions.json', rows)
            if metrics['compatibility']['same_profile'] and flips:
                raise ValueError('same-profile checkpoint dev predictions differ from accepted RB1')
        else:
            conditions = {}
            for kind in (None, 'iv_swap'):
                start = time.monotonic()
                values, rows, _ = evaluate_history(model, splits, tok, kind, args.eval_limit, trace_reads=False)
                conditions[kind or 'normal'] = {'metrics': values, 'predictions': rows,
                    'elapsed_s': time.monotonic()-start}
                write_json(out/'conditions.json', conditions)
            pinned = read(root/'predictions.json')
            if prediction_keys(conditions['normal']['predictions']) != prediction_keys(pinned):
                raise ValueError('normal intervention evaluation differs from trained checkpoint dev')
            metrics['normal.identical_to_training'] = True
        write_json(out/'provenance.json', {'training_run': refs, 'split_manifest': reference(args.split_manifest),
            'evaluation_source': os.environ['SR_CODE_COMMIT'], 'training_source': origin['metrics']['source.commit']})
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-tick
        write_json(out/'manifest.json', {'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'], 'contract':'E-0031@1', 'arm':args.arm,
            'stage':'smoke' if args.eval_limit else 'intervention' if args.arm == 'H-QVU' else 'checkpoint_evaluation',
            'phase':'R1', 'seed':0, 'metrics':metrics,
            'artifacts':[{'path':str(p),'role':p.name} for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'],
            'validator':{'name':'e0031_checkpoint_evaluation','passed':passed}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=(*LEGACY, 'H-QVU'), required=True)
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--training-source')
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--eval-limit', type=int)
    run(p.parse_args())
