"""Formal V2 M-line training and checkpoint evaluation (E-0009/E-0010).

Run this program only in an allocated GPU job. No confirmation split is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from bypass_model import BranchModel
from calibrate import MODEL, REVISION, load_base
from v2_memory import CONTROLS, collate, config, evaluate_batch, train_forward
from v2_schedule import DATA_HASHES, load_data, lr_multiplier, make_schedule


EVAL_BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 8


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@contextmanager
def adapted(base, cfg=None, checkpoint=None):
    # load_base seeds its own loader; seed after loading to pin the adapter.
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = (BranchModel.load_adapter(base, checkpoint) if checkpoint is not None
             else BranchModel(base, cfg))
    try:
        for parameter in model.adapter_parameters():
            if parameter.dtype != torch.float32:
                raise ValueError('trainable adapter parameters must be FP32')
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        yield model
    finally:
        for i, layer in model.adapters().items():
            for handle in layer.handles:
                handle.remove()
            base.model.layers[i] = layer.original
        base.zero_grad(set_to_none=True)


def _summary(records, prior_token):
    count = len(records)
    if not count:
        return {'count': 0, 'em': None, 'nll': None}
    result = {'count': count, 'em': sum(r['correct'] for r in records)/count,
              'nll': sum(r['nll'] for r in records)/count,
              'subject_copy_rate': sum(r['subject_copy'] for r in records)/count,
              'mean_answer_rank': sum(r['answer_rank'] for r in records)/count}
    if prior_token is not None:
        result['prior_em'] = sum(r['target'] == prior_token for r in records)/count
    return result


@torch.no_grad()
def evaluate(model, rows, pad_id, output, split, controls=CONTROLS, prior_token=None):
    """Fixed global cyclic donors make shuffled independent of batch boundaries.

    Diagnostic scalar means are weighted by valid question tokens, matching the
    offline evaluator's all-valid-question scope. EM/NLL and strata are episode
    means. Denominator percentiles remain explicitly per-batch diagnostics.
    """
    if split not in ('dev', 'calibration') or len(rows) < 2:
        raise ValueError('evaluate only released dev/calibration with >=2 owners')
    if any(r['identity']['split'] != split for r in rows):
        raise ValueError('evaluation split mismatch')
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    all_metrics, all_predictions = {}, {}
    for mode in controls:
        if mode not in CONTROLS:
            raise ValueError('invalid control')
        records, diag_sum, diag_weight, batch_percentiles = [], defaultdict(float), defaultdict(int), []
        for start in range(0, len(rows), EVAL_BATCH_SIZE):
            selected = rows[start:start+EVAL_BATCH_SIZE]
            batch = collate(selected, pad_id, device)
            if mode == 'shuffled':
                donors = [rows[(i-1) % len(rows)] for i in range(start, start+len(selected))]
                donor_batch = collate(donors, pad_id, device)
                details, logits = evaluate_batch(model, batch, mode, donor_batch=donor_batch)
            else:
                details, logits = evaluate_batch(model, batch, mode)
            losses = F.cross_entropy(logits.float(), batch.labels, reduction='none')
            predictions = logits.argmax(-1)
            ranks = (logits > logits.gather(1, batch.labels[:, None])).sum(-1)+1
            # Subject tokens are public query content; use the tokenizer once in
            # main to annotate their single-token ID rather than adding labels.
            for row, pred, loss, rank in zip(selected, predictions.tolist(), losses.tolist(), ranks.tolist()):
                records.append({'episode_id': row['episode_id'], 'prediction': pred,
                                'target': row['query']['answer_token_id'], 'nll': loss,
                                'correct': pred == row['query']['answer_token_id'],
                                'answer_rank': rank, 'subject_copy': pred == row.get('_subject_token_id'),
                                'answer_ne_subject': row['query']['answer_text'].strip() != row['query']['head'],
                                'relation': row['query']['relation'],
                                'target_fact_position': row['query']['target_fact_index'],
                                'support_order': row['axes']['support_order']})
            weight = int(batch.question_mask.sum())
            for key, value in details.items():
                if key.startswith('_') or key in ('nll', 'em') or not isinstance(value, (float, int)) or isinstance(value, bool):
                    continue
                if key.startswith('read.denominator.p'):
                    continue
                diag_sum[key] += float(value)*weight
                diag_weight[key] += weight
            batch_percentiles.append({k: v for k, v in details.items() if k.startswith('read.denominator.p')})
        metrics = _summary(records, prior_token)
        metrics['answer_ne_subject'] = _summary([r for r in records if r['answer_ne_subject']], prior_token)
        metrics['strata'] = {}
        for axis in ('relation', 'target_fact_position', 'support_order'):
            groups = defaultdict(list)
            for record in records:
                groups[str(record[axis])].append(record)
            metrics['strata'][axis] = {name: _summary(group, prior_token) for name, group in sorted(groups.items())}
        metrics['diagnostics'] = {k: diag_sum[k]/diag_weight[k] for k in diag_sum}
        metrics['diagnostic_weighting'] = 'valid_question_tokens'
        metrics['denominator_percentiles_by_batch'] = batch_percentiles
        if mode == 'shuffled':
            metrics['donor_rule'] = 'global ordered split: recipient i receives support (i-1) modulo split size'
            metrics['diagnostics']['address.target_available'] = False
        with (output/f'{split}.{mode}.predictions.jsonl').open('w') as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
        all_metrics[mode] = metrics
        all_predictions[mode] = [r['prediction'] for r in records]
    write_json(output/f'{split}.metrics.json', all_metrics)
    return all_metrics, all_predictions


def train(model, rows, schedule, pad_id, lr, microbatch_size, output, dev, prior_token):
    params = list(model.adapter_parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.)
    curves, grad_norms, history, seen_gradients = {}, [], [], set()
    milestones = {math.ceil(schedule['steps']*fraction): str(int(fraction*100)) for fraction in (.25, .5, .75)}
    torch.cuda.reset_peak_memory_stats()
    training_seconds, input_tokens, training_peak, curve_seconds = 0., 0, 0, 0.
    for step, indices in enumerate(schedule['batches']):
        if len(indices) != TRAIN_BATCH_SIZE:
            raise ValueError('schedule must contain complete batch-8 steps')
        model.train()
        model.model.eval()
        optimizer.zero_grad(set_to_none=True)
        step_lr = lr*lr_multiplier(step, schedule['steps'], schedule['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr'] = step_lr
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        step_loss = 0.
        for offset in range(0, TRAIN_BATCH_SIZE, microbatch_size):
            selected = [rows[i] for i in indices[offset:offset+microbatch_size]]
            batch = collate(selected, pad_id, next(model.parameters()).device)
            loss, _, support_counts, query_counts = train_forward(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            (loss*(len(selected)/TRAIN_BATCH_SIZE)).backward()
            step_loss += float(loss.detach())*len(selected)/TRAIN_BATCH_SIZE
            input_tokens += batch.input_tokens
            if support_counts[14]['side']['reads'] or query_counts[14]['side']['writes']:
                raise AssertionError('support/query protocol isolation failed')
            if support_counts[14]['side']['writes'] != int(batch.write_mask.sum()):
                raise AssertionError('each fact must be written exactly once')
        for name, parameter in model.adapters()[14].side.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError('nonfinite adapter gradient: '+name)
                if torch.count_nonzero(parameter.grad):
                    seen_gradients.add(name)
        if any(parameter.grad is not None for name, parameter in model.model.named_parameters()
               if '.side.' not in name):
            raise AssertionError('frozen backbone accumulated gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        if not math.isfinite(norm):
            raise RuntimeError('nonfinite gradient norm')
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic()-tick
        training_peak = max(training_peak, torch.cuda.max_memory_allocated())
        training_seconds += elapsed
        grad_norms.append(norm)
        history.append({'step': step+1, 'tokens': input_tokens, 'loss': step_loss,
                        'lr': step_lr, 'gradient_norm': norm, 'seconds': elapsed})
        with (output/'train.jsonl').open('a') as handle:
            handle.write(json.dumps(history[-1], allow_nan=False)+'\n')
        write_json(output/'progress.json', history[-1])
        if step+1 in milestones:
            milestone = milestones[step+1]
            curve_start = time.monotonic()
            curves[milestone], _ = evaluate(model, dev, pad_id, output/f'curve-{milestone}', 'dev',
                                           ('normal', 'zero', 'shuffled'), prior_token)
            torch.cuda.synchronize()
            curve_seconds += time.monotonic()-curve_start
    if input_tokens != schedule['tokens']:
        raise AssertionError('actual token count differs from planned budget stop')
    names = dict(model.adapters()[14].side.named_parameters())
    stable_times = [record['seconds'] for record in history[1:]] or [history[0]['seconds']]
    return {'steps': len(history), 'tokens': input_tokens, 'training_seconds': training_seconds,
            'learning_curve_evaluation_seconds': curve_seconds,
            'stable_step_seconds': sum(stable_times)/len(stable_times),
            'tokens_per_second': input_tokens/training_seconds,
            'peak_allocated_bytes': training_peak,
            'gradient_norm_mean': sum(grad_norms)/len(grad_norms),
            'gradient_norm_max': max(grad_norms),
            'clip_fraction': sum(n > 1. for n in grad_norms)/len(grad_norms),
            'stored_parameters': sum(p.numel() for p in names.values()),
            'trainable_parameters': sum(p.numel() for p in names.values() if p.requires_grad),
            'parameters_with_nonzero_gradient': sum(names[n].numel() for n in seen_gradients),
            'parameter_names_with_nonzero_gradient': sorted(seen_gradients), 'learning_curves': curves}


def run_arm(base, train_rows, dev, calibration, schedule, pad_id, args, output, tap, backend, fusion, prior):
    output.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    cfg = config(tap, backend, fusion)
    initial_batch = collate([train_rows[i] for i in schedule['batches'][0]], pad_id,
                            next(base.parameters()).device)
    with torch.no_grad():
        mask = initial_batch.question_mask
        reference = base(initial_batch.question_ids, attention_mask=mask,
                         position_ids=(mask.long().cumsum(-1)-1).clamp_min(0), use_cache=False).logits
        last = mask.long().sum(-1)-1
        reference = reference[torch.arange(len(last), device=last.device), last].float()
    with adapted(base, cfg=cfg) as model:
        model.eval()
        with torch.no_grad():
            _, initial_logits, _, _ = train_forward(model, initial_batch)
            initial_delta = float((initial_logits-reference).abs().max())
        if initial_delta != 0.:
            raise AssertionError(f'zero exit differs from Base: {initial_delta}')
        del initial_logits, reference, initial_batch
        result = {'config': asdict(cfg), 'lr': args.lr,
                  'initialization_max_abs': initial_delta,
                  'schedule': {k: v for k, v in schedule.items() if k not in ('order', 'batches')}}
        result['train'] = train(model, train_rows, schedule, pad_id, args.lr, args.microbatch_size, output, dev, prior)
        result['evaluation'] = {}
        predictions = {}
        torch.cuda.synchronize()
        eval_start = time.monotonic()
        for split, rows in (('dev', dev), ('calibration', calibration)):
            if rows is not None:
                result['evaluation'][split], predictions[split] = evaluate(model, rows, pad_id, output, split, prior_token=prior)
        torch.cuda.synchronize()
        result['evaluation_seconds'] = time.monotonic()-eval_start
        save_start = time.monotonic()
        model.export_adapter(output/'adapter')
        torch.cuda.synchronize()
        result['checkpoint_save_seconds'] = time.monotonic()-save_start
    # A new wrapper reads the saved config and every tensor, exercising the actual
    # reload API while reusing the immutable frozen backbone to fit on MIG.
    reload_start = time.monotonic()
    with adapted(base, checkpoint=output/'adapter') as reloaded:
        torch.cuda.synchronize()
        reload_load_seconds = time.monotonic()-reload_start
        if asdict(reloaded.config) != asdict(cfg):
            raise AssertionError('checkpoint configuration differs')
        reload_metrics, reload_predictions = evaluate(reloaded, dev, pad_id, output/'reload', 'dev', ('normal',), prior)
        torch.cuda.synchronize()
    if reload_predictions['normal'] != predictions['dev']['normal']:
        raise AssertionError('checkpoint reload changed per-episode dev predictions')
    result['checkpoint_reload'] = {'predictions_identical': True, 'dev_normal': reload_metrics['normal'],
                                   'load_seconds': reload_load_seconds,
                                   'load_and_evaluation_seconds': time.monotonic()-reload_start}
    result['wall_seconds'] = time.monotonic()-tick
    result['mig_instance_hours'] = result['wall_seconds']/3600
    result['node_hours'] = result['mig_instance_hours']*.25
    write_json(output/'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--token-budget', type=int, default=4_000_000)
    parser.add_argument('--microbatch-size', type=int, default=8)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--smoke-eval-limit', type=int, default=0)
    parser.add_argument('--tap', choices=('direct_hidden', 'qknorm_pre_rope', 'all'), default='qknorm_pre_rope')
    parser.add_argument('--backend', choices=('kernel_matrix', 'soft_slot', 'all'), default='kernel_matrix')
    parser.add_argument('--fusion', choices=('F_out', 'C_up', 'C_dual', 'all'), default='F_out')
    parser.add_argument('--eval-splits', choices=('dev', 'both'), default='both')
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0 or args.token_budget <= 0:
        parser.error('finite positive lr and token budget required')
    if args.microbatch_size < 1 or args.microbatch_size > TRAIN_BATCH_SIZE:
        parser.error('microbatch size must be 1..8')
    if not 0 <= args.smoke_steps <= 64:
        parser.error('smoke steps must be 0 (disabled) or 1..64')
    if args.smoke_eval_limit and (not args.smoke_steps or args.smoke_eval_limit < 2):
        parser.error('smoke eval limit must be >=2 and requires smoke steps')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists() or (args.output/'metrics.json').exists():
        parser.error('output already contains run results')
    started = time.monotonic()
    results, passed, failure = {}, False, None
    try:
        train_rows = load_data(args.data_dir/'train.jsonl', 'train')
        dev = load_data(args.data_dir/'dev.jsonl', 'dev')
        calibration = load_data(args.data_dir/'calibration.jsonl', 'calibration') if args.eval_splits == 'both' else None
        schedule = make_schedule(train_rows, args.token_budget)
        if args.smoke_steps:
            schedule['batches'] = schedule['batches'][:args.smoke_steps]
            schedule['steps'] = len(schedule['batches'])
            schedule['tokens'] = sum(len(train_rows[i]['support_input_ids']) + len(train_rows[i]['question_input_ids'])
                                     for indices in schedule['batches'] for i in indices)
            schedule['warmup_steps'] = math.ceil(.05*schedule['steps'])
            schedule['smoke_steps'] = args.smoke_steps
        if args.smoke_eval_limit:
            dev = dev[:args.smoke_eval_limit]
            if calibration is not None:
                calibration = calibration[:args.smoke_eval_limit]
        write_json(args.output/'schedule.json', {k: v for k, v in schedule.items() if k not in ('order', 'batches')})
        print(json.dumps({'planned_steps': schedule['steps'], 'planned_tokens': schedule['tokens'],
                          'order_sha256': schedule['order_sha256']}), flush=True)
        prior = Counter(r['query']['answer_token_id'] for r in train_rows).most_common(1)[0][0]
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for rows in (dev, calibration):
            for row in rows or []:
                ids = tokenizer.encode(' '+row['query']['head'], add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError('query subject is not a single token')
                row['_subject_token_id'] = ids[0]
        base = load_base()
        taps = ('direct_hidden', 'qknorm_pre_rope') if args.tap == 'all' else (args.tap,)
        backends = ('kernel_matrix', 'soft_slot') if args.backend == 'all' else (args.backend,)
        fusions = ('F_out', 'C_up', 'C_dual') if args.fusion == 'all' else (args.fusion,)
        for tap, backend, fusion in itertools.product(taps, backends, fusions):
            arm = '.'.join((tap, backend, fusion))
            results[arm] = run_arm(base, train_rows, dev, calibration, schedule, pad_id, args,
                                   args.output/arm, tap, backend, fusion, prior)
            write_json(args.output/'metrics.json', results)
        passed = True
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        wall = time.monotonic()-started
        artifacts = [{'path': str(p.resolve()), 'role': p.name} for p in args.output.rglob('*') if p.is_file() and p.name != 'manifest.json']
        write_json(args.output/'manifest.json', {
            'status': 'complete' if passed else 'failed', 'spec_digest': os.environ.get('SR_SPEC_DIGEST'),
            'model': MODEL, 'revision': REVISION, 'seed': 0, 'batch_size': TRAIN_BATCH_SIZE,
            'data_dir': str(args.data_dir.resolve()),
            'data_sha256': {split: DATA_HASHES[split] for split in ('train', 'dev', 'calibration')
                            if split != 'calibration' or args.eval_splits == 'both'},
            'evaluation_batch_size': EVAL_BATCH_SIZE, 'microbatch_size': args.microbatch_size,
            'token_budget': args.token_budget, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'smoke_steps': args.smoke_steps, 'smoke_eval_limit': args.smoke_eval_limit,
            'metrics': {'arms_completed': len(results), 'wall_seconds': wall,
                        'mig_instance_hours': wall/3600, 'node_hours': wall/3600*.25},
            'failure': failure, 'artifacts': artifacts,
            'validator': {'name': 'v2_train', 'passed': passed}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
