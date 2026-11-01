"""Independent clean/read ridge probes on disjoint train and held-out episodes.

Offline only. Inputs must be separately generated train and held-out JSONL files;
fits both heads on training vectors, never on held-out labels or ordinary logits.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from bypass_model import BranchModel
from calibrate import load_base, MODEL, REVISION
from v2_memory import collate, evaluate_batch, independent_probes
from v2_run import read_rows


def validate_probe_splits(train, heldout):
    if any(e['split'] != 'train' for e in train):
        raise ValueError('probe fitting requires training split')
    if any(e['split'] == 'train' for e in heldout):
        raise ValueError('probe evaluation requires held-out split')
    for key in ('episode_id', 'content_sha256'):
        if {e[key] for e in train} & {e[key] for e in heldout}:
            raise ValueError('overlapping probe '+key)


@torch.no_grad()
def vectors(model, rows, pad_id, batch_size):
    clean, read, labels = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = collate(rows[start:start+batch_size], pad_id)
        diagnostics, _ = evaluate_batch(model, batch)
        clean.append(diagnostics['_clean_vectors'])
        read.append(diagnostics['_read_vectors'])
        labels.append(batch.labels.cpu())
    return torch.cat(clean), torch.cat(read), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--heldout', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    success, metrics = False, {}
    start = time.monotonic()
    try:
        train, heldout = read_rows(args.train, args.limit), read_rows(args.heldout, args.limit)
        validate_probe_splits(train, heldout)
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        model = BranchModel.load_adapter(load_base(), args.checkpoint).eval()
        fit_vectors = vectors(model, train, tok.pad_token_id, args.batch_size)
        eval_vectors = vectors(model, heldout, tok.pad_token_id, args.batch_size)
        metrics.update(independent_probes(*fit_vectors, *eval_vectors))
        metrics['diagnostic.fit_input_tokens'] = sum(len(e['support_input_ids'])+len(e['question_input_ids']) for e in train)
        metrics['diagnostic.fit_vector_presentations'] = 2*len(train)
        metrics['diagnostic.heldout_input_tokens'] = sum(len(e['support_input_ids'])+len(e['question_input_ids']) for e in heldout)
        success = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({'status': 'complete' if success else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics, 'artifacts': [],
            'validator': {'name': 'independent_probes', 'passed': success}}, indent=2)+'\n')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
