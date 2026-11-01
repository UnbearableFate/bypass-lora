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
from v2_memory import CONTROLS, DIAGNOSTIC_CONTROLS, collate, config, evaluate_batch, train_forward
from v2r_data import METADATA_SHA256, load_variant
from v2_schedule import (DATA_HASHES, load_data, lr_multiplier, make_curriculum_schedule, make_schedule,
                         make_step_schedule)


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


QUANTILES = (0., .05, .25, .5, .75, .95, 1.)
LAST_FIELDS = {'_last_argmax': 'last_argmax_slot', '_last_target_weight': 'last_target_weight',
               '_last_n_eff': 'last_n_eff'}
RATIO_NAMES = tuple(f'{a}_to_{b}' for a in ('fusion', 'prior') for b in ('residual', 'post_attention'))


def distribution(values):
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    if not len(values):
        return None
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite diagnostic distribution')
    return {'n': len(values), 'mean': float(values.mean()),
            'quantiles': {str(q): float(v) for q, v in zip(QUANTILES, torch.quantile(values, torch.tensor(QUANTILES)))}}


def last_position_summary(records, ratios_all):
    """E-0014 last-question-position read and E-0012 L4 increment/residual ratios."""
    present = [r for r in records if r.get('last_argmax_slot') is not None and r['last_argmax_slot'] >= 0]
    result = {'n': len(present)}
    if present:
        width = max(r['n_facts'] for r in present)
        counts = [0]*width
        for record in present:
            counts[record['last_argmax_slot']] += 1
        result.update(argmax_slot_counts=counts,
                      argmax_slot_frequencies=[c/len(present) for c in counts],
                      argmax_most_common_slot_rate=max(counts)/len(present),
                      n_eff_mean=sum(r['last_n_eff'] for r in present)/len(present))
        if all(r.get('last_target_weight') is not None for r in present):
            result.update(argmax_target_count=sum(r['last_argmax_slot'] == r['target_fact_position'] for r in present),
                          argmax_same_relation_count=sum(r['last_argmax_relation'] == r['relation'] for r in present),
                          target_weight_mean=sum(r['last_target_weight'] for r in present)/len(present))
            result['argmax_target_rate'] = result['argmax_target_count']/len(present)
            result['argmax_same_relation_rate'] = result['argmax_same_relation_count']/len(present)
    for name in RATIO_NAMES:
        values = [r[name+'_last'] for r in records if r.get(name+'_last') is not None]
        if values:
            result[name] = distribution(values)
    all_positions = {name: distribution(torch.cat(chunks)) for name, chunks in ratios_all.items() if chunks}
    return result, all_positions


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
def evaluate(model, rows, pad_id, output, split, controls=CONTROLS, prior_token=None, capture=None):
    """Fixed global cyclic donors make shuffled independent of batch boundaries.

    E-0015: controls may include DIAGNOSTIC_CONTROLS; ``capture`` (a dict) receives
    each mode's last-position F_out increments [episodes, d] in row order.

    Diagnostic scalar means are weighted by valid question tokens, matching the
    offline evaluator's all-valid-question scope. EM/NLL and strata are episode
    means. Reconstruction error uses the maximum across batches; denominator
    percentiles remain explicitly per-batch diagnostics.
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
        if mode not in CONTROLS + DIAGNOSTIC_CONTROLS:
            raise ValueError('invalid control')
        records, diag_sum, diag_weight, batch_percentiles = [], defaultdict(float), defaultdict(int), []
        diag_max, ratios_all, increments = {}, defaultdict(list), []
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
            for i, row in enumerate(selected):
                record = records[len(records)-len(selected)+i]
                for key, field in LAST_FIELDS.items():
                    if details.get(key) is not None:
                        record[field] = details[key][i]
                if record.get('last_argmax_slot') is not None:
                    record['n_facts'] = len(row['facts'])
                    slot = record['last_argmax_slot']
                    # Relation of the argmax slot in the recipient's own support;
                    # undefined for shuffled donor memory.
                    record['last_argmax_relation'] = (row['facts'][slot]['relation']
                                                      if slot >= 0 and mode != 'shuffled' else None)
                for name in RATIO_NAMES:
                    if details.get(f'_{name}_last') is not None:
                        record[name+'_last'] = details[f'_{name}_last'][i]
            for name in RATIO_NAMES:
                if details.get(f'_{name}_all') is not None:
                    ratios_all[name].append(details[f'_{name}_all'])
            if capture is not None:
                increments.append(details['_last_fusion'])
            weight = int(batch.question_mask.sum())
            for key, value in details.items():
                if key.startswith('_') or key in ('nll', 'em') or not isinstance(value, (float, int)) or isinstance(value, bool):
                    continue
                if key.startswith('read.denominator.p'):
                    continue
                if key == 'read.reconstruction_max_abs':
                    diag_max[key] = max(diag_max.get(key, 0.), float(value))
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
        metrics['diagnostics'].update(diag_max)
        metrics['diagnostic_weighting'] = 'valid_question_tokens; read.reconstruction_max_abs is split maximum'
        metrics['denominator_percentiles_by_batch'] = batch_percentiles
        metrics['last_position'], metrics['all_question_positions'] = last_position_summary(records, ratios_all)
        if mode == 'shuffled':
            metrics['donor_rule'] = 'global ordered split: recipient i receives support (i-1) modulo split size'
            metrics['diagnostics']['address.target_available'] = False
        with (output/f'{split}.{mode}.predictions.jsonl').open('w') as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
        all_metrics[mode] = metrics
        all_predictions[mode] = [r['prediction'] for r in records]
        if capture is not None:
            capture[mode] = torch.cat(increments)
    write_json(output/f'{split}.metrics.json', all_metrics)
    return all_metrics, all_predictions


def train(model, rows, schedule, pad_id, lr, microbatch_size, output, dev, prior_token, *,
          forward_options=None, log_name='train.jsonl', learning_curves=True, monitors=None):
    """Defaults reproduce V2. E-0014 options: forward_options go to train_forward;
    only adapter parameters with requires_grad are optimized (R4 freezing);
    monitors maps a completed step to (name, rows) for a calibration normal check."""
    params = [p for p in model.adapter_parameters() if p.requires_grad]
    forward_options = forward_options or {}
    monitors = monitors or {}
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.)
    curves, grad_norms, history, seen_gradients = {}, [], [], set()
    milestones = ({math.ceil(schedule['steps']*fraction): str(int(fraction*100)) for fraction in (.25, .5, .75)}
                  if learning_curves else {})
    monitor_results, components = {}, {}
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
            log = {} if forward_options.get('address_weight') else None
            options = dict(forward_options, log=log) if log is not None else forward_options
            loss, _, support_counts, query_counts = train_forward(model, batch, **options)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            (loss*(len(selected)/TRAIN_BATCH_SIZE)).backward()
            step_loss += float(loss.detach())*len(selected)/TRAIN_BATCH_SIZE
            for key, value in (log or {}).items():
                components[key] = components.get(key, 0.) + value*len(selected)/TRAIN_BATCH_SIZE
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
        if components:
            history[-1].update(components)
            components = {}
        with (output/log_name).open('a') as handle:
            handle.write(json.dumps(history[-1], allow_nan=False)+'\n')
        write_json(output/'progress.json', history[-1])
        if step+1 in milestones:
            milestone = milestones[step+1]
            curve_start = time.monotonic()
            curves[milestone], _ = evaluate(model, dev, pad_id, output/f'curve-{milestone}', 'dev',
                                           ('normal', 'zero', 'shuffled'), prior_token)
            torch.cuda.synchronize()
            curve_seconds += time.monotonic()-curve_start
        if step+1 in monitors:
            name, monitor_rows = monitors[step+1]
            curve_start = time.monotonic()
            metrics, _ = evaluate(model, monitor_rows, pad_id, output/f'monitor-{name}', 'calibration',
                                  ('normal',), None)
            monitor_results[name] = {'step': step+1, 'em': metrics['normal']['em'],
                                     'nll': metrics['normal']['nll'], 'count': metrics['normal']['count']}
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
            'parameter_names_with_nonzero_gradient': sorted(seen_gradients), 'learning_curves': curves,
            **({'monitors': monitor_results} if monitors else {})}


def clear_stale_gradients(module):
    """Drop gradients of frozen parameters (R4 stage-1 prior grads) so later
    nonzero-gradient statistics count only parameters trained in that stage."""
    for parameter in module.parameters():
        if not parameter.requires_grad:
            parameter.grad = None


def _side_tensors(model, prior_path):
    """Copies of the prior-path (True) or memory-path (False) side parameters."""
    return {name: p.detach().clone() for name, p in model.adapters()[14].side.named_parameters()
            if name.startswith('prior.') == prior_path}


def run_arm(base, train_rows, dev, calibration, schedule, pad_id, args, output, tap, backend, fusion, prior, *,
            prior_stage=None, monitors=None, extra_dev=None):
    """prior_stage: (rows, schedule) for the R4 path trained alone at zero memory, then
    frozen; extra_dev: {label: (rows, prior_token)} evaluated with normal/zero/shuffled."""
    output.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    cfg = config(tap, backend, fusion, args.prior_rank)
    if bool(prior_stage) != bool(args.prior_rank):
        raise ValueError('a prior path needs its own training stage and vice versa')
    forward_options = {'address_weight': args.address_loss_weight} if args.address_loss_weight else {}
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
                  'schedule': {k: v for k, v in schedule.items() if k not in ('order', 'batches')},
                  'forward_options': forward_options}
        if prior_stage:
            prior_rows, prior_schedule = prior_stage
            side = model.adapters()[14].side
            memory_before = _side_tensors(model, False)
            for name, parameter in side.named_parameters():
                parameter.requires_grad_(name.startswith('prior.'))
            stage_start = time.monotonic()
            result['prior_stage'] = {
                'lr': args.prior_lr,
                'schedule': {k: v for k, v in prior_schedule.items() if k not in ('order', 'batches')},
                'train': train(model, prior_rows, prior_schedule, pad_id, args.prior_lr, args.microbatch_size,
                               output, dev, prior, forward_options={'zero_memory': True},
                               log_name='prior_train.jsonl', learning_curves=False)}
            changed = [n for n, t in _side_tensors(model, False).items() if not torch.equal(t, memory_before[n])]
            if changed:
                raise AssertionError(f'memory path changed during the prior stage: {changed}')
            result['prior_stage']['dev'], _ = evaluate(model, dev, pad_id, output/'prior-stage', 'dev',
                                                       ('normal', 'zero'), prior)
            torch.cuda.synchronize()
            result['prior_stage']['seconds'] = time.monotonic()-stage_start
            for name, parameter in side.named_parameters():
                parameter.requires_grad_(not name.startswith('prior.'))
            clear_stale_gradients(side)
            prior_frozen = _side_tensors(model, True)
        result['train'] = train(model, train_rows, schedule, pad_id, args.lr, args.microbatch_size, output, dev, prior,
                                forward_options=forward_options, monitors=monitors)
        if prior_stage:
            changed = [n for n, t in _side_tensors(model, True).items() if not torch.equal(t, prior_frozen[n])]
            if changed:
                raise AssertionError(f'frozen prior path changed during memory training: {changed}')
            result['prior_stage']['frozen_during_memory_training'] = True
        result['evaluation'] = {}
        predictions = {}
        torch.cuda.synchronize()
        eval_start = time.monotonic()
        for split, rows in (('dev', dev), ('calibration', calibration)):
            if rows is not None:
                result['evaluation'][split], predictions[split] = evaluate(model, rows, pad_id, output, split, prior_token=prior)
        for label, (rows, token) in (extra_dev or {}).items():
            result['evaluation'][label], _ = evaluate(model, rows, pad_id, output/label, 'dev',
                                                      ('normal', 'zero', 'shuffled'), token)
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
    parser.add_argument('--backend', choices=('kernel_matrix', 'soft_slot', 'centered_cos_slot', 'all'),
                        default='kernel_matrix')
    parser.add_argument('--fusion', choices=('F_out', 'C_up', 'C_dual', 'all'), default='F_out')
    parser.add_argument('--eval-splits', choices=('dev', 'both'), default='both')
    # E-0014 options; every default reproduces V2 (token budget, E=16 train, no extras).
    parser.add_argument('--step-budget', type=int, default=0, help='train this many batch-8 steps instead of a token budget')
    parser.add_argument('--variant-data', action='append', default=[], metavar='E=DIR',
                        help='E-0014 fact-count variant data directory (E in 1,4,8)')
    parser.add_argument('--train-facts', type=int, choices=(1, 4, 8, 16), default=16)
    parser.add_argument('--curriculum', default='', metavar='E:STEPS,...',
                        help='contiguous step segments by facts per episode; sums to --step-budget')
    parser.add_argument('--extra-dev', type=int, action='append', default=[], choices=(1, 4, 8),
                        help='also evaluate this variant dev with normal/zero/shuffled')
    parser.add_argument('--prior-rank', type=int, default=0)
    parser.add_argument('--prior-steps', type=int, default=0)
    parser.add_argument('--prior-lr', type=float, default=0.)
    parser.add_argument('--address-loss-weight', type=float, default=0.)
    args = parser.parse_args()
    variant_dirs = {}
    for item in args.variant_data:
        facts, _, directory = item.partition('=')
        if facts not in ('1', '4', '8') or not directory or int(facts) in variant_dirs:
            parser.error('--variant-data takes unique E=DIR with E in 1,4,8')
        variant_dirs[int(facts)] = Path(directory)
    segments = []
    if args.curriculum:
        for item in args.curriculum.split(','):
            facts, _, steps = item.partition(':')
            if facts not in ('1', '4', '8', '16') or not steps.isdigit() or int(steps) < 1:
                parser.error('--curriculum takes E:STEPS items with E in 1,4,8,16')
            segments.append((int(facts), int(steps)))
        if sum(n for _, n in segments) != args.step_budget or args.train_facts != 16:
            parser.error('curriculum steps must sum to --step-budget and replace --train-facts')
    needed = {f for f, _ in segments} | set(args.extra_dev) | ({args.train_facts} if not segments else set())
    if any(f != 16 and f not in variant_dirs for f in needed):
        parser.error('every used variant needs --variant-data')
    if args.step_budget < 0 or (args.train_facts != 16 and not args.step_budget):
        parser.error('variant training requires a positive --step-budget')
    if bool(args.prior_rank) != bool(args.prior_steps) or bool(args.prior_rank) != (args.prior_lr > 0):
        parser.error('--prior-rank, --prior-steps and --prior-lr are given together')
    if args.prior_rank < 0 or args.prior_steps < 0 or not math.isfinite(args.prior_lr):
        parser.error('invalid prior path options')
    if args.address_loss_weight and (args.address_loss_weight < 0 or not math.isfinite(args.address_loss_weight)
                                     or args.backend != 'soft_slot'):
        parser.error('the address loss needs a finite positive weight and --backend soft_slot')
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
        variants = {f: {s: load_variant(variant_dirs[f], f, s) for s in ('train', 'calibration', 'dev')}
                    for f in sorted(needed - {16})}
        v2_train_rows, prior_stage, monitors = train_rows, None, None
        if segments:
            calibration_by_facts = {f: (variants[f]['calibration'][1] if f != 16 else
                                        load_data(args.data_dir/'calibration.jsonl', 'calibration'))
                                    for f, _ in segments}
            train_rows, schedule = make_curriculum_schedule(
                [(f'e{f}', variants[f]['train'][1] if f != 16 else v2_train_rows, n) for f, n in segments])
            monitors = {part['last_step']: (part['label'], calibration_by_facts[f])
                        for part, (f, _) in zip(schedule['curriculum'], segments)}
        elif args.step_budget:
            if args.train_facts != 16:
                train_rows = variants[args.train_facts]['train'][1]
            schedule = make_step_schedule(train_rows, args.step_budget)
        else:
            schedule = make_schedule(train_rows, args.token_budget)
        if args.prior_rank:
            prior_stage = (v2_train_rows, make_step_schedule(v2_train_rows, args.prior_steps))
        extra_dev = {f'e{f}': (variants[f]['dev'][1], variants[f]['dev'][0]['prior']['answer_token_id'])
                     for f in args.extra_dev}
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
            extra_dev = {k: (rows[:args.smoke_eval_limit], token) for k, (rows, token) in extra_dev.items()}
            if monitors:
                monitors = {k: (name, rows[:args.smoke_eval_limit]) for k, (name, rows) in monitors.items()}
        write_json(args.output/'schedule.json', {k: v for k, v in schedule.items() if k not in ('order', 'batches')})
        print(json.dumps({'planned_steps': schedule['steps'], 'planned_tokens': schedule['tokens'],
                          'order_sha256': schedule['order_sha256']}), flush=True)
        # The E=16 dev prior token always comes from the V2 train split (as in V2).
        prior = Counter(r['query']['answer_token_id'] for r in v2_train_rows).most_common(1)[0][0]
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for rows in (dev, calibration, *(r for r, _ in extra_dev.values())):
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
                                   args.output/arm, tap, backend, fusion, prior, prior_stage=prior_stage,
                                   monitors=monitors, extra_dev=extra_dev)
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
            'token_budget': None if args.step_budget else args.token_budget,
            'options': {'step_budget': args.step_budget, 'train_facts': args.train_facts,
                        'curriculum': segments, 'extra_dev': args.extra_dev, 'prior_rank': args.prior_rank,
                        'prior_steps': args.prior_steps, 'prior_lr': args.prior_lr,
                        'address_loss_weight': args.address_loss_weight},
            'variant_data': {str(f): {'dir': str(d.resolve()), 'metadata_sha256': METADATA_SHA256[f]}
                             for f, d in variant_dirs.items()},
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'smoke_steps': args.smoke_steps, 'smoke_eval_limit': args.smoke_eval_limit,
            'metrics': {'arms_completed': len(results), 'wall_seconds': wall,
                        'mig_instance_hours': wall/3600, 'node_hours': wall/3600*.25},
            'failure': failure, 'artifacts': artifacts,
            'validator': {'name': 'v2_train', 'passed': passed}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
