"""Frozen Base difficulty check on released T2 dev, using V2 vocabulary scoring."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import load_base
from v2_eval import left_pad, prompt_tokens, score_logits
from v3_data import MODEL, REVISION, VERSION, encode, file_digest


def load_released_data(directory, metadata_sha256, split='dev'):
    """Check data gates and integrity before any model is loaded or called."""
    if split not in ('train', 'calibration', 'dev'):
        raise ValueError('sealed slices are restricted')
    directory = Path(directory)
    path = directory / 'metadata.json'
    if file_digest(path) != metadata_sha256:
        raise ValueError('metadata digest mismatch')
    metadata = json.loads(path.read_text())
    if metadata['version'] != VERSION or metadata['phase'] != 'V3' or metadata['tokenizer_revision'] != REVISION:
        raise ValueError('T2 generator/phase/tokenizer mismatch')
    if not all(metadata['shortcuts'][s]['gates_passed'] for s in ('calibration', 'dev')):
        raise ValueError('structural shortcut gates failed; model evaluation is forbidden')
    entry = metadata['splits'][split]
    if entry['sealed'] or metadata['name_pools_sha256'] != file_digest(directory / 'name_pools.json'):
        raise ValueError('sealed data or name-pool mismatch')
    path = directory / (split + '.jsonl')
    if file_digest(path) != entry['sha256']:
        raise ValueError('released split digest mismatch')
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != metadata['sizes'][split] or len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('split size or unique identities mismatch')
    for row in rows:
        ident = row['identity']
        if (ident['split'] != split or ident['phase'] != 'V3' or ident['generator_version'] != VERSION
                or ident['task'] != 'T2' or ident['layout']['name_pool'] != 'seen'
                or ident['layout']['hop'] not in (1, 2, 3)):
            raise ValueError('row is not released in-distribution T2')
    return metadata, rows


def grouped_scores(records):
    groups = defaultdict(list)
    for row in records:
        hop, order = row['hop'], row['support_order']
        for group in ('all', f'hop={hop}', f'order={order}', f'hop={hop},order={order}'):
            groups[group].append(row)
    return {key: {'examples': len(rows),
                  **{metric: sum(r[metric] for r in rows) / len(rows)
                     for metric in ('em', 'nll', 'subject_copy', 'prior')}}
            for key, rows in groups.items()}


@torch.inference_mode()
def evaluate(model, tok, rows, output, constant, batch_size=8):
    if not rows or batch_size < 1:
        raise ValueError('nonempty dev and positive batch size are required')
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Base must be frozen')
    device = next(model.parameters()).device
    metrics, report = {}, {}
    with (output / 'predictions.jsonl').open('w') as handle:
        for line in ('M', 'F'):
            records = []
            tick = time.monotonic()
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                inputs = left_pad([prompt_tokens(tok, row, line) for row in batch], tok.pad_token_id, device)
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                targets = torch.tensor([r['query']['answer_token_id'] for r in batch], device=device)
                predictions, losses = score_logits(logits, targets)
                for row, pred, loss in zip(batch, predictions.tolist(), losses.tolist()):
                    subject = encode(tok, ' ' + row['query']['head'])
                    if len(subject) != 1:
                        raise ValueError('subject must use the same validated single-token pool')
                    record = {'episode_id': row['episode_id'], 'line': line,
                              'hop': row['axes']['hop'], 'support_order': row['axes']['support_order'],
                              'prediction': pred, 'target': row['query']['answer_token_id'],
                              'em': pred == row['query']['answer_token_id'], 'nll': loss,
                              'subject_copy': pred == subject[0], 'prior': row['query']['answer_token_id'] == constant}
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
                    records.append(record)
            groups = grouped_scores(records)
            report[line] = groups
            metrics[line + '.elapsed_s'] = time.monotonic() - tick
            for group, values in groups.items():
                for metric, value in values.items():
                    metrics[f'{line}.{group}.{metric}'] = value
            if line == 'M':
                for group, values in groups.items():
                    metrics[f'M.{group}.within_prior_plus_5pp'] = values['em'] <= values['prior'] + .05
    return metrics, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--metadata-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--smoke-limit', type=int, default=0)
    args = parser.parse_args()
    start = time.monotonic()
    metadata, rows = load_released_data(args.data_dir, args.metadata_sha256)
    if args.smoke_limit < 0:
        raise ValueError('negative smoke limit')
    if args.smoke_limit:
        rows = rows[:args.smoke_limit]
    elif len(rows) != 1000:
        raise ValueError('full evaluation requires the contract dev size 1000')
    args.output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tick = time.monotonic()
    model = load_base()
    load_s = time.monotonic() - tick
    metrics, groups = evaluate(model, tok, rows, args.output, metadata['prior']['answer_token_id'], args.batch_size)
    metrics.update(model_load_s=load_s, elapsed_s=time.monotonic() - start, smoke_limit=args.smoke_limit)
    report = {'metrics': metrics, 'groups': groups, 'metadata_sha256': args.metadata_sha256,
              'dev_sha256': metadata['splits']['dev']['sha256'], 'model': MODEL, 'revision': REVISION,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'batch_size': args.batch_size}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    manifest = {'status': 'complete', 'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics,
                'artifacts': [{'path': str((args.output / name).resolve()), 'role': name}
                              for name in ('metrics.json', 'predictions.jsonl')],
                'validator': {'name': 'v3_eval', 'passed': True}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
