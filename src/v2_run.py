"""Small V2 M-line training/evaluation entry point; execute only inside PBS.

This implementation task authorizes smoke steps only. Larger studies need their
own contract and explicit command. No training data is selected using dev scores.
"""
import argparse
from contextlib import contextmanager
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from bypass_model import BranchModel
from calibrate import load_base, MODEL, REVISION
from v2_memory import (config, collate, train_forward, evaluate_batch, CONTROLS)


@contextmanager
def adapted(base, cfg, seed=2100042):
    torch.manual_seed(seed)
    model = BranchModel(base, cfg)
    try:
        yield model
    finally:
        for i, layer in model.adapters().items():
            for handle in layer.handles:
                handle.remove()
            base.model.layers[i] = layer.original
        base.zero_grad(set_to_none=True)


def fixture(tokenizer, n_facts=2):
    """Copy positive control: identical question, contrasting episode answers.

    This is a tiny engineering fixture, not E-0005 dev or scientific evidence.
    Each fact repeats the same name, making the read unambiguously learnable.
    """
    rows = []
    for i, name in enumerate(('Alice', 'Bob')):
        answer = ' '+name
        labels = tokenizer.encode(answer, add_special_tokens=False)
        if len(labels) != 1:
            raise ValueError('fixture answer is not one token')
        text, facts = '', []
        for j in range(n_facts):
            text += f'Item {j} is {name}'
            start = len(text)
            text += ';'
            facts.append({'head': f'Item{j}', 'relation': 'is', 'tail': name, 'delimiter_char_span': [start, len(text)]})
            text += '\n'
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        for fact in facts:
            start, end = fact['delimiter_char_span']
            indices = [j for j, (a, b) in enumerate(encoded['offset_mapping']) if a <= start and b >= end]
            if len(indices) != 1:
                raise ValueError('fixture delimiter must map uniquely')
            fact['delimiter_token_index'] = indices[0]
        question = 'Which name appears in the facts? Answer:'
        rows.append({'generator_version': 'T-0006-copy-fixture-v1', 'task': 'copy', 'layout': 'fixture',
                     'base_seed': 2100042, 'split': 'fixture', 'index': i, 'episode_id': f'copy-{i}',
                     'content_sha256': hashlib.sha256(text.encode()).hexdigest(), 'support_text': text,
                     'support_input_ids': encoded['input_ids'], 'question_text': question,
                     'question_input_ids': tokenizer.encode(question, add_special_tokens=False), 'facts': facts,
                     'query': {'head': 'Item0', 'relation': 'is', 'target_fact_index': 0,
                               'answer_text': answer, 'answer_token_id': labels[0]},
                     'axes': {'hop': 1, 'n_facts': n_facts, 'n_distractors': n_facts-1,
                              'name_pool': 'seen', 'support_order': 'natural'}})
    return rows


def read_rows(path, limit):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    if len(rows) < 2:
        raise ValueError('need at least two episode owners')
    return rows


def train_steps(model, batch, steps, lr, progress_path):
    model.train()
    # Backbone dropout is pinned to zero by the loaded Qwen config; no stochastic
    # layers are introduced. Keep frozen backbone in eval mode explicitly.
    model.model.eval()
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=lr, weight_decay=0.)
    times, losses, grads, clipped = [], [], [], []
    startup = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, _, support_counts, query_counts = train_forward(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        params = list(model.adapter_parameters())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError('nonfinite adapter gradient')
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        grads.append(float(norm)); clipped.append(float(norm > 1))
        if step < 2:
            startup.append({n: float(p.grad.norm()) if p.grad is not None else None
                            for n, p in model.adapters()[14].side.named_parameters()})
        if any(p.grad is not None for n, p in model.model.named_parameters() if '.side.' not in n):
            raise AssertionError('frozen backbone accumulated gradients')
        if support_counts[14]['side']['reads'] != 0 or query_counts[14]['side']['writes'] != 0:
            raise AssertionError('M-line phase isolation failed')
        if support_counts[14]['side']['writes'] != int(batch.write_mask.sum()):
            raise AssertionError('wrong support write count')
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.monotonic()-start)
        losses.append(float(loss.detach()))
        progress_path.write_text(json.dumps({'steps': step+1, 'input_tokens': batch.input_tokens*(step+1),
                                             'step_seconds': times, 'losses': losses})+'\n')
    stable = times[1:] or times
    mean = sum(stable)/len(stable)
    return {'steps': steps, 'input_tokens': batch.input_tokens*steps, 'losses': losses,
            'step_seconds': times, 'stable_step_seconds': mean,
            'tokens_per_second': batch.input_tokens/mean,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'gradient_norm': grads, 'clip_fraction': sum(clipped)/len(clipped), 'startup_gradients': startup}


def json_diagnostics(details):
    return {k: v for k, v in details.items() if not k.startswith('_')}


def run_arm(base, rows, pad_id, tap, backend, fusion, steps, lr, out):
    out.mkdir(parents=True, exist_ok=True)
    batch = collate(rows, pad_id)
    with torch.no_grad():
        reference = base(batch.question_ids, attention_mask=batch.question_mask,
                         position_ids=(batch.question_mask.long().cumsum(-1)-1).clamp_min(0), use_cache=False).logits
    with adapted(base, config(tap, backend, fusion)) as model:
        model.eval()
        with torch.no_grad():
            loss, logits, _, _ = train_forward(model, batch)
            last = batch.question_mask.sum(-1)-1
            reference_last = reference[torch.arange(len(last), device=last.device), last].float()
            delta = float((logits-reference_last).abs().max())
            if delta != 0:
                raise AssertionError(f'zero exit differs from Base: {delta}')
        del reference
        result = {'initialization_max_abs': delta,
                  'parameters': sum(p.numel() for p in model.adapter_parameters())}
        result['train'] = train_steps(model, batch, steps, lr, out/'train-progress.json')
        model.eval()
        result['controls'] = {}
        probe_vectors = {}
        for mode in CONTROLS:
            diagnostics, _ = evaluate_batch(model, batch, mode)
            probe_vectors[mode] = {k: v for k, v in diagnostics.items() if k.startswith('_')}
            result['controls'][mode] = json_diagnostics(diagnostics)
        normal = result['controls']['normal']['nll']
        result['positive_control_pass'] = (normal < result['controls']['zero']['nll'] and
                                           normal < result['controls']['shuffled']['nll'])
        torch.save(probe_vectors, out/'probe-vectors.pt')
        model.export_adapter(out/'adapter')
        # Reload exact tensors into this same adapter: verifies the actual export
        # names/layout without reloading a second frozen backbone on the MIG.
        original = {i: {k: v.detach().clone() for k, v in layer.side.state_dict().items()} for i, layer in model.adapters().items()}
        weights = torch.load(out/'adapter'/'adapter.pt', weights_only=True)
        for i, layer in model.adapters().items():
            layer.side.load_state_dict(weights[str(i)], strict=True)
            if any(not torch.equal(v, original[i][k]) for k, v in layer.side.state_dict().items()):
                raise AssertionError('checkpoint roundtrip mismatch')
        result['checkpoint_identical'] = True
        (out/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--data')
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--fixture-facts', type=int, default=2)
    parser.add_argument('--tap', choices=('all', 'direct_hidden', 'qknorm_pre_rope'), default='all')
    parser.add_argument('--backend', choices=('all', 'kernel_matrix', 'soft_slot'), default='all')
    parser.add_argument('--fusion', choices=('all', 'F_out', 'C_dual'), default='all')
    parser.add_argument('--require-positive', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.steps <= 64:
        parser.error('this entry point is limited to 1..64 smoke steps')
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    results, success = {}, False
    start = time.monotonic()
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        rows = read_rows(args.data, args.limit) if args.data else fixture(tok, args.fixture_facts)
        (out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        base = load_base()
        choices = [('direct_hidden', 'qknorm_pre_rope') if args.tap == 'all' else (args.tap,),
                   ('kernel_matrix', 'soft_slot') if args.backend == 'all' else (args.backend,),
                   ('F_out', 'C_dual') if args.fusion == 'all' else (args.fusion,)]
        for tap, backend, fusion in itertools.product(*choices):
            arm = '.'.join((tap, backend, fusion))
            print('Starting '+arm, flush=True)
            results[arm] = run_arm(base, rows, tok.pad_token_id, tap, backend, fusion, args.steps, args.lr, out/arm)
            (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        success = not args.require_positive or all(v['positive_control_pass'] for v in results.values())
    finally:
        metrics = {'elapsed_s': time.monotonic()-start,
                   'train.input_tokens': sum(json.loads(p.read_text())['input_tokens'] for p in out.glob('*/train-progress.json')),
                   'arms_completed': len(results)}
        for arm, result in results.items():
            metrics[arm+'.tokens_per_second'] = result['train']['tokens_per_second']
            metrics[arm+'.positive_control_pass'] = result['positive_control_pass']
        (out/'manifest.json').write_text(json.dumps({'status': 'complete' if success else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics,
            'artifacts': [{'path': str(out/'results.json'), 'role': 'arm_metrics'}] if results else [],
            'validator': {'name': 'v2_smoke', 'passed': success}}, indent=2)+'\n')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
