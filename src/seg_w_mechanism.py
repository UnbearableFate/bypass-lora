"""E-0032 rule 7 (descriptive): segment-1 share of the KernelMatrix read mass.

For each trained Keep checkpoint and each dev row (gold candidate only), segment
1 is written under the arm's registered mask; segment 2 then reads with the
carried state. At every segment-2 answer position t (the positions whose
logits score the candidate tokens) and every adapter layer, the KernelMatrix
read normalizes by rho_t . c_t, where rho_t = ELU1(RMSNorm(qbar_t) Rq) is the
first (only, S1) reader query and c_t = c1 + sum_{u<=t} ELU1(RMSNorm(kbar_u))
(inclusive write before read). The reported share is (rho_t . c1)/(rho_t . c_t)
(eps excluded). It is averaged over a row's answer positions, then over rows,
per task and layer. Descriptive only; no rule depends on it.
"""
import argparse
import os
from pathlib import Path
import time

import torch

from bypass_core import elu_plus_one, rmsnorm_no_affine
from rb1_artifacts import read, reference
from seg_data import encode_segments
from seg_eval import tensors, arm_write, write_tensor
from seg_model import make_model, first_forward, KEEP, WRITE_RULE
from seg_train import tokenizer
from v1_run import prepare
from v5_train import write_json


@torch.no_grad()
def shares(model, arm, tok, splits, limit=None):
    model.eval()
    layers = model.config.layers
    captured = {}
    handles = [model.adapters()[i].side.projection.register_forward_hook(
        lambda module, args, output, i=i: captured.__setitem__(i, output)) for i in layers]
    result = {}
    try:
        for task in ('piqa', 'boolq'):
            sums = {i: 0. for i in layers}
            pooled = {i: [0., 0] for i in layers}
            rows = splits['splits'][task]['dev'][:limit]
            for row in rows:
                gold = int(row['label']) if task == 'piqa' else (0 if row['label'] else 1)
                encoded = encode_segments(tok, row)[gold]
                first, mask, second = tensors(encoded, tok.pad_token_id)
                write = write_tensor(arm_write(tok, arm, encoded['segment1_ids']), first)
                memory = first_forward(model, first, mask, write).cache.memory
                captured.clear()
                model(second['input_ids'], attention_mask=second['attention_mask'], initial_memory=memory,
                      use_cache=False)
                valid = second['attention_mask'][0].bool()
                start, end = encoded['prompt_length']-1, int(valid.sum())-1   # positions predicting candidate tokens
                if end <= start:
                    raise ValueError('segmented candidate has no answer positions')
                for i in layers:
                    side = model.adapters()[i].side
                    qbar, kbar, _ = captured[i]
                    reader = side.reader
                    rho = elu_plus_one(rmsnorm_no_affine(qbar[0].float(), reader.eps_norm) @ reader.Rq)
                    keys = elu_plus_one(rmsnorm_no_affine(kbar[0].float(), reader.eps_norm))
                    keys = torch.where(valid[:, None], keys, torch.zeros_like(keys))
                    c1 = memory[i].c[0]
                    total = c1[None]+keys.cumsum(0)
                    share = (rho @ c1)/(rho*total).sum(-1)
                    selected = share[start:end]
                    if not torch.isfinite(selected).all():
                        raise ValueError('nonfinite read-mass share')
                    sums[i] += selected.mean().item()
                    pooled[i][0] += selected.sum().item(); pooled[i][1] += selected.numel()
            result[task] = {str(i): {'row_mean': sums[i]/len(rows), 'position_pooled': pooled[i][0]/pooled[i][1],
                                     'rows': len(rows), 'positions': pooled[i][1]} for i in layers}
    finally:
        for handle in handles:
            handle.remove()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, values, start = False, {}, time.monotonic()
    try:
        spec = read(args.input)
        values['input'] = reference(args.input)
        values['hardware'] = torch.cuda.get_device_name()
        tok = tokenizer(); splits = prepare(args.data_dir)
        models = {}
        for entry in spec['models']:
            arm, seed, run = entry['arm'], entry['seed'], Path(entry['run'])
            if arm not in KEEP or arm == 'SEG-02':
                raise ValueError('rule 7 describes r128 Keep checkpoints only')
            manifest = read(run/'manifest.json')
            if manifest.get('status') != 'complete' or manifest.get('arm') != arm or manifest.get('seed') != seed:
                raise ValueError(f'{run} is not a complete {arm} seed-{seed} run')
            checkpoint = read(run/'checkpoint.json')
            if any(reference(ref['path'])['sha256'] != ref['sha256'] for ref in checkpoint.values()):
                raise ValueError(f'{run} checkpoint changed')
            tick = time.monotonic()
            model = make_model(arm, run/'adapter', seed=seed)
            models[entry['label']] = {'arm': arm, 'seed': seed, 'run': str(run), 'checkpoint': checkpoint,
                                      'write_rule': WRITE_RULE.get(arm, 'all'),
                                      'shares': shares(model, arm, tok, splits), 'seconds': time.monotonic()-tick}
            del model; torch.cuda.empty_cache()
        write_json(out/'mechanism.json', {'definition': __doc__, 'models': models})
        values['models'] = len(models)
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0032@1',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'source_commit': os.environ['SR_CODE_COMMIT'],
            'stage': 'rule7_mechanism', 'metrics': values,
            'artifacts': [{'path': str(out/'mechanism.json'), 'role': 'mechanism.json'}] if (out/'mechanism.json').exists() else [],
            'validator': {'name': 'e0032_rule7_read_mass', 'passed': passed}})


if __name__ == '__main__':
    main()
