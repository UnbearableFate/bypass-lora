"""E-0008 context-retained LoRA positive control. Run only in PBS."""
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import load_base
from v1_lora import LoRAModel, LoRAConfig
from v2_data import MODEL, REVISION, encode
from v2_eval import prompt_tokens, left_pad, score_logits
from v2_schedule import load_data, make_schedule, lr_multiplier

def final_logits(model, inputs):
    return model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]


def answer_losses(model, inputs, targets):
    return score_logits(final_logits(model, inputs), targets)[1]


def make_model(checkpoint=None):
    base = load_base()
    torch.manual_seed(0)
    return (LoRAModel.load_adapter(base, checkpoint) if checkpoint else
            LoRAModel(base, LoRAConfig()))


@torch.inference_mode()
def evaluate(model, tok, rows, split, constant, output):
    model.eval()
    records = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        inputs = left_pad([prompt_tokens(tok, r, 'F', split=split) for r in batch],
                          tok.pad_token_id, next(model.parameters()).device)
        logits = final_logits(model, inputs)
        targets = torch.tensor([r['query']['answer_token_id'] for r in batch], device=logits.device)
        preds, losses = score_logits(logits, targets)
        ranks = 1 + (logits > logits.gather(1, targets[:, None])).sum(-1)
        for row, pred, loss, rank in zip(batch, preds.tolist(), losses.tolist(), ranks.tolist()):
            subject = encode(tok, ' ' + row['query']['head'])
            if len(subject) != 1:
                raise ValueError('subject must be single-token')
            target = row['query']['answer_token_id']
            records.append(dict(episode_id=row['episode_id'], prediction=pred, target=target,
                                nll=loss, rank=rank, subject=subject[0], self_edge=target == subject[0]))
    output.write_text(''.join(json.dumps(r) + '\n' for r in records))
    result = {}
    for name, subset in [('all', records), ('nonself', [r for r in records if not r['self_edge']])]:
        if not subset:
            raise ValueError('empty evaluation slice')
        n = len(subset)
        result.update({f'{name}.examples': n,
                       f'{name}.em': sum(r['prediction'] == r['target'] for r in subset) / n,
                       f'{name}.nll': sum(r['nll'] for r in subset) / n,
                       f'{name}.subject_copy': sum(r['prediction'] == r['subject'] for r in subset) / n,
                       f'{name}.rank_mean': sum(r['rank'] for r in subset) / n,
                       f'{name}.prior_em': sum(r['target'] == constant for r in subset) / n})
    return result, records


def train(model, tok, rows, schedule, lr, output, intermediate=None):
    batches = schedule['batches']
    params = list(model.adapter_parameters())
    assert all(p.dtype == torch.float32 for p in params)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.)
    tokens = 0
    tick = time.monotonic()
    for step, indices in enumerate(batches, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_lr = lr * lr_multiplier(step - 1, len(batches), schedule['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr'] = step_lr
        batch = [rows[i] for i in indices]
        inputs = left_pad([prompt_tokens(tok, r, 'F', split='train') for r in batch],
                          tok.pad_token_id, next(model.parameters()).device)
        targets = torch.tensor([r['query']['answer_token_id'] for r in batch], device=inputs['input_ids'].device)
        losses = answer_losses(model, inputs, targets)
        loss = losses.mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        tokens += sum(len(r['support_input_ids']) + len(r['question_input_ids']) for r in batch)
        with (output / 'learning.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(step=step, tokens=tokens, loss=loss.item(),
                                         lr=step_lr, grad_norm=norm.item())) + '\n')
        if intermediate:
            intermediate(step, model)
    return {'train.steps': len(batches), 'train.tokens': tokens,
            'train.seconds': time.monotonic() - tick}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--tokens', type=int, choices=(200000, 4000000), default=200000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--data-phase', choices=('V2', 'V3'), default='V2')
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('positive finite learning rate required')
    if args.data_phase == 'V3' and args.tokens != 200000:
        raise ValueError('V3 data selection is restricted to the 0.2M-token pilot')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    metrics, passed = {}, False
    stage = 'setup'
    try:
        read_data = load_data
        if args.data_phase == 'V3':
            from v3_pilot_data import load_data as read_data
        train_rows = read_data(args.data_dir / 'train.jsonl', 'train')
        calibration = read_data(args.data_dir / 'calibration.jsonl', 'calibration')
        dev = read_data(args.data_dir / 'dev.jsonl', 'dev') if args.tokens == 4000000 or args.smoke else None
        schedule = (make_schedule(train_rows, args.tokens, phase='V3') if args.data_phase == 'V3'
                    else make_schedule(train_rows, args.tokens))
        (out / 'schedule.json').write_text(json.dumps(schedule) + '\n')
        if args.smoke:
            schedule = {**schedule, 'batches': schedule['batches'][:4], 'warmup_steps': 1}
            calibration, dev = calibration[:32], dev[:32]
        counts = Counter(r['query']['answer_token_id'] for r in train_rows)
        constant = min(counts, key=lambda token: (-counts[token], token))
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        model = make_model()
        torch.cuda.reset_peak_memory_stats()
        checkpoints = {math.ceil(len(schedule['batches']) * fraction) for fraction in (.25, .5, .75)}
        def intermediate(step, current):
            nonlocal stage
            if dev is not None and step in checkpoints:
                stage = 'intermediate_evaluation'
                values, _ = evaluate(current, tok, dev, 'dev', constant, out / f'dev-step-{step}.jsonl')
                with (out / 'curve.jsonl').open('a') as handle:
                    handle.write(json.dumps({'step': step, 'metrics': values}) + '\n')
                stage = 'training'
        stage = 'training'
        metrics.update(train(model, tok, train_rows, schedule, args.lr, out, intermediate))
        metrics['train.tokens_per_second'] = metrics['train.tokens'] / metrics['train.seconds']
        stage = 'checkpoint_save'
        model.export_adapter(out / 'adapter')
        eval_rows = {'calibration': calibration}
        if dev is not None:
            eval_rows['dev'] = dev
        originals = {}
        stage = 'original_evaluation'
        for split, rows in eval_rows.items():
            values, records = evaluate(model, tok, rows, split, constant, out / f'{split}.jsonl')
            originals[split] = records
            metrics.update({f'{split}.{key}': val for key, val in values.items()})
            if args.data_phase == 'V3':
                from v3_pilot_data import stratify
                metrics.update({f'{split}.{key}': val for key, val in stratify(rows, records, constant).items()})
        metrics['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        stage = 'checkpoint_reload'
        model = make_model(out / 'adapter')
        stage = 'reload_evaluation'
        for split, rows in eval_rows.items():
            _, repeated = evaluate(model, tok, rows, split, constant, out / f'{split}-reloaded.jsonl')
            if repeated != originals[split]:
                raise AssertionError('checkpoint reload differs per episode')
        metrics['reload.identical'] = True
        metrics['calibration.gate'] = metrics['calibration.all.em'] >= metrics['calibration.all.prior_em'] + .20
        metrics['failed_lr'] = False
        if args.smoke and metrics['train.steps'] != 4:
            raise AssertionError('smoke requires four completed steps')
        passed = True
    except Exception as exc:
        metrics['failure.stage'] = stage
        nonfinite = isinstance(exc, (ValueError, RuntimeError)) and (
            'nonfinite' in str(exc).lower() or 'non-finite' in str(exc).lower())
        # Only non-smoke training/original-evaluation divergence is a scientific
        # failed LR. Reload, curve, setup and smoke failures must propagate.
        if nonfinite and not args.smoke and stage in ('training', 'original_evaluation'):
            metrics.update({'failed_lr': True, 'failure': str(exc)})
            passed = True
        else:
            metrics['error'] = repr(exc)
            raise
    finally:
        metrics['elapsed_s'] = time.monotonic() - start
        report = {'metrics': metrics, 'lr': args.lr, 'tokens_budget': args.tokens,
                  'data_phase': args.data_phase,
                  'smoke': args.smoke, 'model': MODEL, 'revision': REVISION,
                  'torch': torch.__version__, 'cuda': torch.version.cuda, 'eval_batch': 16}
        (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        artifacts = [{'path': str(p.resolve()), 'role': p.name}
                     for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json']
        (out / 'manifest.json').write_text(json.dumps({'status': 'complete' if passed else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics, 'artifacts': artifacts,
            'validator': {'name': 'v2_fline', 'passed': passed}}, indent=2) + '\n')


if __name__ == '__main__':
    main()
