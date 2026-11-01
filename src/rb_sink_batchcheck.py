"""T-0027 engineering check: where batched gold-NLL differences come from.

On the first rows of each dev task, compare batch-one gold NLL and answer-position
n_eff with (a) the same row right-padded alone, (b) the same row repeated with no
padding, and (c) length-sorted right-padded batches. Base and one trained RB1
checkpoint. No result here enters E-0024; it decides how E-0024 is run.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb1_intervene import original_backbone
from rb_sink_read import TASKS, answer_mask, gold_index
from rb_train import make_model
from v1_data import encode_candidates
from v1_run import batch_tensors, prepare


def forward(model, encoded, pad_id, traced, extra_pad=0):
    batch = batch_tensors(encoded, pad_id)
    if extra_pad:
        width = batch['input_ids'].shape[1] + extra_pad
        for key, fill in (('input_ids', pad_id), ('labels', -100), ('attention_mask', 0)):
            batch[key] = F.pad(batch[key], (0, extra_pad), value=fill)
        assert batch['input_ids'].shape[1] == width
    kwargs = {'trace_reads': True} if traced else {}
    output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False, **kwargs)
    shifted = batch['labels'][:, 1:]
    nll = [float(F.cross_entropy(output.logits[b, :-1][shifted[b] != -100].float(),
                                 shifted[b][shifted[b] != -100], reduction='sum'))
           for b in range(len(encoded))]
    neff = None
    if traced:
        answer = answer_mask(batch['labels'], batch['attention_mask'])
        neff = {layer: [float(d['read_metrics'][0]['n_eff'][b][answer[b]].double().mean())
                        for b in range(len(encoded))] for layer, d in output.diagnostics.items()}
    return nll, neff


def gaps(reference, other):
    diff = [abs(a-b) for a, b in zip(reference, other)]
    return {'max': max(diff), 'mean': sum(diff)/len(diff)}


@torch.no_grad()
def compare(model, encoded, pad_id, traced, batch_size):
    single = [forward(model, [e], pad_id, traced) for e in encoded]
    padded = [forward(model, [e], pad_id, traced, extra_pad=17) for e in encoded]
    repeated = [forward(model, [e]*4, pad_id, traced) for e in encoded]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]['input_ids']))
    batched_nll, batched_neff = [None]*len(encoded), {}
    for start in range(0, len(order), batch_size):
        idx = order[start:start+batch_size]
        nll, neff = forward(model, [encoded[i] for i in idx], pad_id, traced)
        for j, i in enumerate(idx):
            batched_nll[i] = nll[j]
            for layer, values in (neff or {}).items():
                batched_neff.setdefault(layer, [None]*len(encoded))[i] = values[j]
    base_nll = [s[0][0] for s in single]
    result = {'nll': {'pad_only': gaps(base_nll, [p[0][0] for p in padded]),
                      'repeat_no_pad': gaps(base_nll, [r[0][0] for r in repeated]),
                      'repeat_copies_agree': max(max(r[0])-min(r[0]) for r in repeated),
                      'sorted_batches': gaps(base_nll, batched_nll)}}
    if traced:
        result['n_eff'] = {}
        for layer in single[0][1]:
            ref = [s[1][layer][0] for s in single]
            result['n_eff'][layer] = {
                'pad_only': gaps(ref, [p[1][layer][0] for p in padded]),
                'repeat_no_pad': gaps(ref, [r[1][layer][0] for r in repeated]),
                'sorted_batches': gaps(ref, batched_neff[layer])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--arm', default='RB1-02')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--rows', type=int, default=48)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from rb1_artifacts import validate_formal_run
    _, _, adapter = validate_formal_run(args.training_run, args.arm)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    splits = prepare(args.data_dir)
    model = make_model(args.arm, phase='R1', seed=0, checkpoint=adapter).eval()
    report = {'arm': args.arm, 'rows_per_task': args.rows, 'hardware': torch.cuda.get_device_name()}
    for task in TASKS:
        rows = splits['splits'][task]['dev'][:args.rows]
        encoded = [encode_candidates(tok, row)[gold_index(row)] for row in rows]
        with original_backbone(model) as base:
            report[f'{task}.Base'] = compare(base, encoded, tok.pad_token_id, False, 8)
        report[f'{task}.{args.arm}'] = compare(model, encoded, tok.pad_token_id, True, 8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
