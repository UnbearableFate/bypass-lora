"""Frozen Base, full-vocabulary one-token EM/NLL on released V2 dev only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from calibrate import load_base
from v2_data import MODEL, REVISION, VERSION, encode, file_digest


def score_logits(logits, targets):
    if logits.ndim != 2 or targets.shape != logits.shape[:1]:
        raise ValueError('expected one full-vocabulary logit vector per target')
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite vocabulary logits')
    losses = F.cross_entropy(logits.float(), targets, reduction='none')
    if not torch.isfinite(losses).all():
        raise ValueError('nonfinite NLL')
    return logits.argmax(-1), losses


def prompt_tokens(tok, row, line):
    if row['identity']['split'] != 'dev' or row['identity']['layout']['name_pool'] != 'seen':
        raise ValueError('Base evaluation is restricted to seen-name dev')
    if line not in ('M', 'F'):
        raise ValueError('unknown line')
    prompt = (row['support_text'] if line == 'F' else '') + row['question_text']
    ids = encode(tok, prompt)
    target = row['query']['answer_token_id']
    if encode(tok, prompt + row['query']['answer_text']) != ids + [target]:
        raise ValueError('single-token answer boundary mismatch')
    return ids


def left_pad(sequences, pad_id, device):
    length = max(map(len, sequences))
    if not length or any(not x for x in sequences):
        raise ValueError('empty prompt')
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, -len(seq):] = torch.tensor(seq, device=device)
        mask[i, -len(seq):] = 1
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    return {'input_ids': ids, 'attention_mask': mask, 'position_ids': positions}


@torch.inference_mode()
def evaluate(model, tok, rows, output, batch_size=8):
    if batch_size < 1 or not rows:
        raise ValueError('positive batch size and nonempty dev are required')
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Base must be frozen')
    device = next(model.parameters()).device
    metrics = {}
    with (output / 'predictions.jsonl').open('w') as handle:
        for line in ('M', 'F'):
            correct, total_nll = 0, 0.
            tick = time.monotonic()
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                prompts = [prompt_tokens(tok, row, line) for row in batch]
                inputs = left_pad(prompts, tok.pad_token_id, device)
                # Last position is valid in every left-padded prompt. Compute
                # only its vocabulary projection, without support KV reuse.
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                targets = torch.tensor([r['query']['answer_token_id'] for r in batch], device=device)
                predictions, losses = score_logits(logits, targets)
                for row, pred, loss in zip(batch, predictions.tolist(), losses.tolist()):
                    correct += pred == row['query']['answer_token_id']
                    total_nll += loss
                    handle.write(json.dumps({'episode_id': row['episode_id'], 'line': line,
                                             'prediction': pred, 'target': row['query']['answer_token_id'],
                                             'nll': loss}) + '\n')
            metrics[line + '.em'] = correct / len(rows)
            metrics[line + '.nll'] = total_nll / len(rows)
            metrics[line + '.examples'] = len(rows)
            metrics[line + '.elapsed_s'] = time.monotonic() - tick
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--metadata-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--smoke-limit', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    metadata_path = args.data_dir / 'metadata.json'
    if file_digest(metadata_path) != args.metadata_sha256:
        raise ValueError('metadata digest mismatch')
    metadata = json.loads(metadata_path.read_text())
    if metadata['version'] != VERSION or metadata['tokenizer_revision'] != REVISION:
        raise ValueError('data generator/tokenizer version mismatch')
    dev_path = args.data_dir / 'dev.jsonl'
    if file_digest(dev_path) != metadata['splits']['dev']['sha256']:
        raise ValueError('dev digest mismatch')
    rows = [json.loads(line) for line in dev_path.open()]
    if len(rows) != metadata['sizes']['dev'] or len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('invalid dev size or duplicate IDs')
    if args.smoke_limit < 0:
        raise ValueError('negative smoke limit')
    if args.smoke_limit:
        rows = rows[:args.smoke_limit]
    elif len(rows) != 1000:
        raise ValueError('full evaluation requires the contract dev size 1000')
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    load_start = time.monotonic()
    model = load_base()
    metrics = {'model_load_s': time.monotonic() - load_start}
    metrics.update(evaluate(model, tok, rows, args.output, args.batch_size))
    constant = metadata['prior']['answer_token_id']
    metrics['prior.dev_em'] = sum(r['query']['answer_token_id'] == constant for r in rows) / len(rows)
    metrics['M.within_prior_plus_5pp'] = metrics['M.em'] <= metrics['prior.dev_em'] + .05
    metrics['F.above_prior_plus_20pp'] = metrics['F.em'] >= metrics['prior.dev_em'] + .20
    metrics['elapsed_s'] = time.monotonic() - tick
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('nonfinite summary')
    report = {'metrics': metrics, 'metadata_sha256': args.metadata_sha256,
              'dev_sha256': metadata['splits']['dev']['sha256'], 'batch_size': args.batch_size,
              'smoke_limit': args.smoke_limit, 'model': MODEL, 'revision': REVISION,
              'torch': torch.__version__, 'cuda': torch.version.cuda}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest = {'status': 'complete', 'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics,
                'artifacts': [{'path': str((args.output / f).resolve()), 'role': f}
                              for f in ('metrics.json', 'predictions.jsonl')],
                'validator': {'name': 'v2_eval', 'passed': True}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
