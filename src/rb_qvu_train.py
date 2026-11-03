"""E-0031 fixed seed/order/LR training and exact same-allocation identity checks."""
import argparse
import gc
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb1_artifacts import read, reference, verify_lr_lock
from rb1_intervene import evaluate_history, original_backbone
from rb_data import plan_batches
from rb_eta_select import ORDER_SHA256, SPLIT_SHA256, validate_calibration
from rb_tap_check import initial_state
from rb_tap_identity import prediction_keys
from rb_train import E0031_ARMS, PARAMETERS, load_data, make_model, train
from v1_run import evaluate, preserved_evaluation_state
from v5_train import plan_metadata, write_json


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    passed, metrics = False, {}
    tick = time.monotonic()
    try:
        if args.stage != 'smoke' and (args.smoke_steps is not None or args.eval_limit is not None):
            raise ValueError('partial evaluation or budget is smoke-only')
        if args.stage == 'train' and args.arm in ('U-25','T0-U-25'):
            if not args.conditional_analysis:
                raise ValueError('conditional training requires the recorded rule6 analysis')
            release = read(args.conditional_analysis)
            if (release.get('contract') != 'E-0031@1' or
                    release.get('source_commit') != os.environ['SR_CODE_COMMIT'] or
                    release.get('rules',{}).get('6_conditional_controls',{}).get('triggered') is not True):
                raise ValueError('rule6 did not release conditional controls for this source')
            metrics['conditional_analysis'] = reference(args.conditional_analysis)
        lock = verify_lr_lock(args.lr_lock)
        if lock['locked_lrs']['RBc-C'] != .001 or reference(args.split_manifest)['sha256'] != SPLIT_SHA256:
            raise ValueError('E-0031 fixes the accepted split and LR1e-3')
        if lock['split_manifest_sha256'] != SPLIT_SHA256:
            raise ValueError('LR lock split differs')
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        tok.pad_token_id = tok.eos_token_id
        splits, excluded = load_data(args.data_dir, args.split_manifest, tok)
        batches = plan_batches(splits, tok, 'R1', 0, [(r['task'], r['id']) for r in excluded], 2000000)
        plan = plan_metadata(batches)
        if (plan['plan.steps'] != 13115 or plan['plan.examples'] != 26230 or
                plan['plan.order_sha256'] != ORDER_SHA256 or plan['plan.input_tokens'] != 2000069):
            raise ValueError('training plan differs from locked RB1 order and budget')
        if args.stage == 'smoke':
            batches = batches[:args.smoke_steps or 80]
        metrics.update(plan_metadata(batches))
        metrics.update({'source.commit': os.environ['SR_CODE_COMMIT'], 'model_revision': REVISION,
            'tokenizer_revision': REVISION, 'environment.lock_sha256': reference(Path(__file__).resolve().parents[1]/'uv.lock')['sha256'],
            'split_manifest.sha256': SPLIT_SHA256, 'lr_lock.sha256': reference(args.lr_lock)['sha256'],
            'learning_rate': .001, 'failed': False, 'hardware': torch.cuda.get_device_name(),
            'exec.pbs_job_id': os.environ['PBS_JOBID'], 'torch': torch.__version__, 'cuda': torch.version.cuda})
        write_json(out/'config.json', dict(vars(args), phase='R1', seed=0, lr=.001, three_epochs=False))
        write_json(out/'excluded-train.json', excluded)
        start = time.monotonic()
        model = make_model(args.arm)
        metrics['load.seconds'] = time.monotonic()-start
        metrics['init.parameters.sha256'] = initial_state(model)[0]
        metrics['params.stored'] = sum(p.numel() for p in model.adapter_parameters())
        metrics['params.trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        def monitor(step, current):
            values, rows = evaluate(current, splits, tok, args.eval_limit, split_names=('dev',))
            write_json(out/f'dev-step-{step}.json', {'step': step, 'metrics': values, 'predictions': rows})
            return rows
        with preserved_evaluation_state(model):
            start = time.monotonic()
            initial = monitor(0, model)
            with original_backbone(model) as backbone:
                values, base_rows = evaluate(backbone, splits, tok, args.eval_limit, split_names=('dev',))
            write_json(out/'dev-base-same-allocation.json', {'metrics': values, 'predictions': base_rows,
                'job_id': os.environ['PBS_JOBID'], 'hardware': metrics['hardware'],
                'source_commit': os.environ['SR_CODE_COMMIT']})
            metrics['init.identical_to_Base'] = prediction_keys(initial) == prediction_keys(base_rows)
            metrics['init_eval.seconds'] = time.monotonic()-start
            if not metrics['init.identical_to_Base']:
                raise ValueError('step0 predictions differ from same-allocation Base')
        metrics.update(train(model, batches, tok, .001, out, monitor if args.stage == 'train' else None))
        if any(metrics[key] != PARAMETERS[args.arm] for key in ('params.stored', 'params.trainable', 'params.with_grad')):
            raise ValueError('stored/trainable/gradient counts differ from contract')
        model.export_adapter(out/'adapter')
        write_json(out/'checkpoint.json', {name: reference(out/'adapter'/name) for name in ('config.json', 'adapter.pt')})
        start = time.monotonic()
        values, dev = evaluate(model, splits, tok, args.eval_limit, split_names=('dev',))
        metrics.update(values)
        metrics['dev_eval.seconds'] = time.monotonic()-start
        write_json(out/'predictions.json', dev)
        start = time.monotonic()
        values, calibration = evaluate(model, splits, tok, args.eval_limit, split_names=('calibration',))
        metrics.update(values)
        metrics['calibration.seconds'] = time.monotonic()-start
        if args.stage == 'train':
            validate_calibration(calibration, metrics)
        write_json(out/'calibration-predictions.json', calibration)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        start = time.monotonic()
        model = make_model(args.arm, checkpoint=out/'adapter')
        values, reloaded = evaluate(model, splits, tok, args.eval_limit, split_names=('dev', 'calibration'))
        metrics['reload.identical'] = reloaded == dev+calibration and all(metrics[k] == v for k, v in values.items())
        metrics['reload.seconds'] = time.monotonic()-start
        if not metrics['reload.identical']:
            raise ValueError('checkpoint reload predictions or scores differ')
        start = time.monotonic()
        off_values, off_rows, _ = evaluate_history(model, splits, tok, 'iv_off', args.eval_limit, trace_reads=False)
        metrics['iv_off.identical_to_Base'] = prediction_keys(off_rows) == prediction_keys(base_rows)
        metrics['iv_off.seconds'] = time.monotonic()-start
        write_json(out/'iv-off.json', {'metrics': off_values, 'predictions': off_rows})
        if not metrics['iv_off.identical_to_Base']:
            raise ValueError('iv_off differs from same-allocation Base')
        passed = True
    finally:
        metrics['failed'] = not passed
        metrics['elapsed_s'] = time.monotonic()-tick
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'contract': 'E-0031@1',
            'arm': args.arm, 'family': 'RBc-C', 'stage': args.stage, 'phase': 'R1', 'seed': 0,
            'metrics': metrics, 'artifacts': [{'path': str(p), 'role': p.name}
                for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'],
            'validator': {'name': 'e0031_training', 'passed': passed}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=E0031_ARMS, required=True)
    p.add_argument('--stage', choices=('smoke', 'train'), required=True)
    p.add_argument('--lr-lock', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke-steps', type=int)
    p.add_argument('--eval-limit', type=int)
    p.add_argument('--conditional-analysis', type=Path)
    run(p.parse_args())
