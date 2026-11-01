"""E-0013@3 V5 training and LR selection. Execute only in a PBS allocation.

Only the three locked V1 train source files are read. The evaluation and adapter
formats come directly from V1; this module owns the new seeds and training plan.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import transformers
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION, load_base
from bypass_model import AdapterConfig, BranchModel
from v1_lora import LoRAConfig, LoRAModel
from v1_data import audit_training_exclusions, derive_seed, encode_training_example
from v1_run import (EvaluationNonfinite, TrainingDiverged, batch_tensors, evaluate,
                    example_mean_objective, lr_factor, prepare,
                    preserved_evaluation_state, token_losses)


ARMS = {'V5-C': 524288, 'V5-F': 524288, 'V5-L': 630784}
LAYERS = (6, 13, 20, 27)
GRID = (1e-4, 3e-4, 1e-3)
CONTRACT = 'E-0013@3'
TRAIN_ROWS = 19443


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(arm, seed, checkpoint=None):
    base = load_base()
    # load_base resets its diagnostic seed, so reset immediately before adapters.
    torch.manual_seed(derive_seed(20261029, 'concat_loop_v2', 'V5', 'init', seed))
    if arm == 'base':
        return base
    if arm == 'V5-L':
        config = LoRAConfig(layers=tuple(range(28)), rank=1, alpha=1)
        model = (LoRAModel.load_adapter(base, checkpoint) if checkpoint else
                 LoRAModel(base, config))
    else:
        config = AdapterConfig(layers=LAYERS, rank=32 if arm == 'V5-C' else 64,
                               branch='tokenwise_silu',
                               fusion='C_up' if arm == 'V5-C' else 'F_out')
        model = (BranchModel.load_adapter(base, checkpoint) if checkpoint else
                 BranchModel(base, config))
    if model.config != config:
        raise ValueError('checkpoint configuration differs from the V5 arm')
    params = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == ARMS[arm]
    assert all(p.dtype == torch.float32 for p in params)
    return model


def training_stream(splits, seed, epoch, excluded=()):
    """V1 task shuffle/alternation, with only the V5 training seed replaced."""
    rng = random.Random(derive_seed(20261029, 'concat_loop_v2', 'V5', 'train', seed) + epoch)
    excluded = set(excluded)
    tasks = {}
    for task in ('piqa', 'boolq'):
        rows = list(splits['splits'][task]['train'])
        rng.shuffle(rows)
        tasks[task] = [r for r in rows if (task, r['id']) not in excluded]
    positions = {'piqa': 0, 'boolq': 0}
    result = []
    while any(positions[t] < len(tasks[t]) for t in tasks):
        for task in tasks:
            pos = positions[task]
            if pos < len(tasks[task]):
                result.append(tasks[task][pos])
                positions[task] += 1
    return result


def plan_batches(splits, tok, seed, excluded, token_budget=None):
    """Carry pending examples across epochs, exactly as V1 plan_batches does."""
    batches, pending, cache, tokens = [], [], {}, 0
    for epoch in range(3):
        stream = training_stream(splits, seed, epoch, excluded)
        for index, row in enumerate(stream):
            key = (row['task'], row['id'])
            if key not in cache:
                cache[key] = encode_training_example(tok, row, max_length=512)
            encoded = dict(cache[key], epoch=epoch, epoch_end=index == len(stream)-1)
            pending.append(encoded)
            if len(pending) == 2:
                batches.append(pending)
                tokens += sum(len(e['input_ids']) for e in pending)
                pending = []
                if token_budget is not None and tokens >= token_budget:
                    return batches
    if pending:
        batches.append(pending)
    return batches


def plan_metadata(batches):
    order = [(e['task'], e['id'], e['epoch']) for batch in batches for e in batch]
    end_steps = {str(e['epoch']+1): step for step, batch in enumerate(batches, 1)
                 for e in batch if e['epoch_end']}
    return {'plan.steps': len(batches), 'plan.examples': len(order),
            'plan.input_tokens': sum(len(e['input_ids']) for batch in batches for e in batch),
            'plan.order_sha256': hashlib.sha256(json.dumps(order).encode()).hexdigest(),
            'plan.epoch_end_steps': end_steps}


def smoke_batches(batches, per_epoch=12):
    # Include each actual epoch boundary as well as enough steps for steady timing.
    selected, first = [], 0
    for end in plan_metadata(batches)['plan.epoch_end_steps'].values():
        indices = sorted(set(range(first, min(first+per_epoch, end))) | {end-1})
        selected.extend(batches[i] for i in indices)
        first = end
    return selected


def raw_best(candidates):
    finite = [(float(lr), float(nll)) for lr, nll in candidates.items()
              if nll is not None and math.isfinite(nll)]
    if not finite:
        raise ValueError('no finite LR candidate')
    return min(finite, key=lambda pair: (pair[1], pair[0]))


def extension_lr(candidates):
    if set(candidates) != set(GRID):
        raise ValueError('endpoint expansion requires exactly the initial grid')
    best, _ = raw_best(candidates)
    return {1e-4: 3e-5, 1e-3: 3e-3}.get(best)


def select_lr(candidates):
    _, best = raw_best(candidates)
    if best <= 0:
        raise ValueError('calibration NLL must be positive')
    return min(lr for lr, nll in candidates.items() if nll is not None
               and math.isfinite(nll) and (nll-best)/best <= .01)


def train(model, batches, tok, lr, output, monitor):
    """V1 example-mean objective; monitoring after epoch-last optimizer steps."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    totals = {'train.steps': 0, 'train.input_tokens': 0, 'train.loss_tokens': 0,
              'train.padded_tokens': 0, 'train.examples': 0}
    durations, grad_seen, monitoring_seconds = [], set(), 0.
    start = time.monotonic()
    try:
        with (output/'steps.jsonl').open('w') as log:
            for step, examples in enumerate(batches):
                torch.cuda.synchronize()
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                summed = 0.
                for e in examples:
                    losses, _ = token_losses(model, batch_tensors([e], tok.pad_token_id))
                    totals['train.examples'] += 1
                    totals['train.input_tokens'] += len(e['input_ids'])
                    totals['train.loss_tokens'] += losses.numel()
                    if not losses.numel() or not torch.isfinite(losses).all():
                        raise TrainingDiverged('nonfinite or empty loss', dict(totals))
                    example_mean_objective(losses, len(examples)).backward()
                    summed += losses.detach().mean().item()
                norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=False).item()
                grad_seen.update(id(p) for p in params if p.grad is not None)
                if not math.isfinite(norm):
                    raise TrainingDiverged('nonfinite gradient norm', dict(totals))
                rate = lr * lr_factor(step, len(batches))
                for group in optimizer.param_groups:
                    group['lr'] = rate
                optimizer.step()
                totals['train.steps'] += 1
                torch.cuda.synchronize()
                duration = time.monotonic()-tick
                durations.append(duration)
                log.write(json.dumps({'step': step+1, 'example_mean_nll': summed/len(examples),
                                      'preclip_norm': norm, 'lr': rate, 'seconds': duration})+'\n')
                if (step+1) % 100 == 0:
                    log.flush()
                    write_json(output/'usage.json', totals)
                for epoch in [e['epoch']+1 for e in examples if e['epoch_end']]:
                    if monitor is not None:
                        tick = time.monotonic()
                        with preserved_evaluation_state(model):
                            monitor(epoch, step+1, model)
                        torch.cuda.synchronize()
                        monitoring_seconds += time.monotonic()-tick
    finally:
        write_json(output/'usage.json', totals)
    totals.update({'train.seconds': time.monotonic()-start-monitoring_seconds,
                   'train.steady_step_seconds': sum(durations[10:])/len(durations[10:]) if len(durations)>10 else None,
                   'train.peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                   'monitor.seconds': monitoring_seconds,
                   'params.with_grad': sum(p.numel() for p in params if id(p) in grad_seen)})
    return totals


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists() or (out/'manifest.json').exists():
        raise ValueError('refusing to overwrite an existing run')
    metrics, passed, stage = {'learning_rate': args.lr}, False, 'preparation'
    start = time.monotonic()
    try:
        if args.phase == 'train':
            lock = json.loads(args.lr_lock.read_text())
            if lock['contract'] != CONTRACT or lock['locked_lrs'][args.arm] != args.lr:
                raise ValueError('training LR differs from the committed lock')
            metrics['lr_lock.sha256'] = file_sha256(args.lr_lock)
        write_json(out/'config.json', vars(args))
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        tok.pad_token_id = tok.eos_token_id
        splits = prepare(args.data_dir)
        excluded = audit_training_exclusions(splits, tok)
        if [(r['task'], r['id'], r['candidate_lengths']) for r in excluded] != [('piqa', 5528, [537, 537])]:
            raise ValueError('V1 training exclusion changed')
        excluded_ids = [(r['task'], r['id']) for r in excluded]
        rows = len(training_stream(splits, args.seed, 0, excluded_ids))
        assert rows == TRAIN_ROWS
        metrics['data.rows_per_epoch'] = rows
        write_json(out/'data-metadata.json', splits['metadata'])
        write_json(out/'excluded-train.json', excluded)
        batches = plan_batches(splits, tok, args.seed, excluded_ids,
                               200000 if args.phase == 'calibration' else None)
        metrics.update(plan_metadata(batches))
        if args.phase != 'calibration':
            assert metrics['plan.steps'] == 29165 and metrics['plan.examples'] == 58329
            assert metrics['plan.epoch_end_steps'] == {'1': 9722, '2': 19443, '3': 29165}
        if args.phase == 'smoke':
            batches = smoke_batches(batches)
            metrics.update({key.replace('plan.', 'smoke.'): val for key, val in plan_metadata(batches).items()})
        tick = time.monotonic()
        model = make_model(args.arm, args.seed)
        torch.cuda.synchronize()
        metrics['load.seconds'] = time.monotonic()-tick
        metrics['hardware'] = torch.cuda.get_device_name()
        metrics['torch'] = torch.__version__
        metrics['transformers'] = transformers.__version__
        metrics['cuda'] = torch.version.cuda
        metrics['params.trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics['params.stored'] = sum(p.numel() for p in model.parameters())
        limit = 32 if args.phase == 'smoke' else None
        if args.phase == 'smoke':
            with preserved_evaluation_state(model), torch.no_grad():
                model.eval()
                batch = batch_tensors([batches[0][0]], tok.pad_token_id)
                inputs = {k:v for k,v in batch.items() if k != 'labels'}
                base = make_model('base', args.seed)
                metrics['init.identical'] = torch.equal(model(**inputs, use_cache=False).logits,
                                                       base(**inputs, use_cache=False).logits)
                assert metrics['init.identical']
                del base
                gc.collect()
                torch.cuda.empty_cache()
        final_eval = {}
        def monitor(epoch, step, current):
            nonlocal stage
            stage = 'dev_monitoring'
            tick = time.monotonic()
            values, preds = evaluate(current, splits, tok, limit, split_names=('dev',))
            torch.cuda.synchronize()
            write_json(out/f'dev-epoch-{epoch}.json', {'epoch': epoch, 'step': step,
                       'seconds': time.monotonic()-tick, 'metrics': values})
            if epoch == 3:
                final_eval.update(metrics=values, predictions=preds)
            stage = 'training'
        if args.arm != 'base':
            stage = 'training'
            torch.cuda.reset_peak_memory_stats()
            metrics.update(train(model, batches, tok, args.lr, out,
                                 None if args.phase == 'calibration' else monitor))
            assert metrics['params.with_grad'] == ARMS[args.arm]
        if args.phase == 'calibration':
            stage = 'calibration_evaluation'
            tick = time.monotonic()
            values, _ = evaluate(model, splits, tok, split_names=('calibration',))
            torch.cuda.synchronize()
            metrics.update(values)
            metrics['eval.seconds'] = time.monotonic()-tick
            passed = True
            return
        stage = 'saving'
        if args.arm != 'base':
            tick = time.monotonic()
            model.export_adapter(out/'adapter')
            torch.cuda.synchronize()
            metrics['save.seconds'] = time.monotonic()-tick
            write_json(out/'checkpoint.json', {name: {'path': str(out/'adapter'/name),
                       'sha256': file_sha256(out/'adapter'/name),
                       'bytes': (out/'adapter'/name).stat().st_size} for name in ('config.json', 'adapter.pt')})
        else:
            stage = 'base_evaluation'
            tick = time.monotonic()
            values, preds = evaluate(model, splits, tok, split_names=('dev',))
            metrics['eval.seconds'] = time.monotonic()-tick
            final_eval.update(metrics=values, predictions=preds)
        metrics.update(final_eval['metrics'])
        with (out/'predictions.jsonl').open('w') as handle:
            for pred in final_eval['predictions']:
                handle.write(json.dumps(pred, allow_nan=False)+'\n')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        stage = 'reload_evaluation'
        tick = time.monotonic()
        model = make_model(args.arm, args.seed, None if args.arm == 'base' else out/'adapter')
        torch.cuda.synchronize()
        metrics['reload.load_seconds'] = time.monotonic()-tick
        tick = time.monotonic()
        values, preds = evaluate(model, splits, tok, limit, split_names=('dev',))
        torch.cuda.synchronize()
        metrics['reload.eval_seconds'] = time.monotonic()-tick
        metrics['reload.identical'] = values == final_eval['metrics'] and preds == final_eval['predictions']
        assert metrics['reload.identical'], 'fresh-Base reload differs from the final evaluation'
        passed = True
    except (TrainingDiverged, EvaluationNonfinite) as exc:
        metrics.update(getattr(exc, 'metrics', {}))
        metrics['failure.stage'], metrics['failure.reason'] = stage, str(exc)
        if args.phase == 'calibration' and stage in ('training', 'calibration_evaluation'):
            metrics['run.failed_lr'] = True
            passed = True
        else:
            raise
    except Exception as exc:
        metrics['failure.stage'], metrics['failure.reason'] = stage, repr(exc)
        raise
    finally:
        if (out/'usage.json').exists():
            for key, value in json.loads((out/'usage.json').read_text()).items():
                metrics.setdefault(key, value)
        metrics['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial',
                   'contract': CONTRACT, 'phase': args.phase,
                   'spec_digest': os.environ['SR_SPEC_DIGEST'], 'arm': args.arm, 'seed': args.seed,
                   'metrics': metrics, 'artifacts': [{'path': str(p), 'role': p.stem}
                   for p in out.glob('*') if p.is_file() and p.name != 'manifest.json'],
                   'validator': {'name': 'v5_runtime', 'passed': passed}})


def lock_lrs(args):
    grouped, refs = {arm: {} for arm in ARMS}, []
    for path in args.manifests:
        manifest = json.loads(path.read_text())
        m = manifest['metrics']
        if (manifest['status'] != 'complete' or not manifest['validator']['passed']
                or manifest['seed'] != 0 or manifest['phase'] != 'calibration'
                or manifest['contract'] != CONTRACT):
            raise ValueError(f'invalid calibration manifest: {path}')
        lr, arm = m['learning_rate'], manifest['arm']
        if lr in grouped[arm]:
            raise ValueError('duplicate LR candidate')
        grouped[arm][lr] = None if m.get('run.failed_lr') else m['calibration.nll']
        refs.append({'path': str(path.resolve()), 'sha256': file_sha256(path), 'arm': arm, 'lr': lr})
    locked = {}
    for arm, candidates in grouped.items():
        initial = {lr: candidates[lr] for lr in GRID}
        extension = extension_lr(initial)
        expected = set(GRID) | ({extension} if extension else set())
        if set(candidates) != expected:
            raise ValueError(f'{arm}: required candidates {expected}, got {set(candidates)}')
        locked[arm] = select_lr(candidates)
    write_json(args.output, {'contract': CONTRACT, 'locked_lrs': locked,
               'calibration_manifests': refs,
               'rule': 'expand raw finite initial-grid minimum once; smallest LR within 1% of final finite NLL minimum'})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--arm', choices=['base', *ARMS], required=True)
    p.add_argument('--phase', choices=['smoke', 'calibration', 'train', 'base'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-lock', type=Path)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('lock')
    p.add_argument('--manifests', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'lock':
        lock_lrs(args)
        return
    if (args.phase == 'train' and (args.seed not in (1, 2, 3) or args.lr_lock is None)
            or args.phase in ('smoke', 'calibration', 'base') and args.seed != 0
            or (args.phase == 'base') != (args.arm == 'base')
            or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error('invalid V5 phase, seed, arm or locked learning rate')
    # Path arguments are serialized in the run configuration.
    args.output = args.output.resolve()
    run(args)


if __name__ == '__main__':
    main()
