"""E-0029 S1 read weights by prefix length on a fixed training-side subset."""
import torch

from bypass_core import elu_plus_one, rmsnorm_no_affine
from v1_run import batch_tensors


PREFIX_BINS = ((1, 1), (2, 8), (9, 32), (33, 128), (129, 512))


@torch.no_grad()
def read_statistics(qbar, kbar, reader, valid):
    if qbar.shape != kbar.shape or valid.shape != qbar.shape[:2]:
        raise ValueError('read statistic shapes differ')
    q = elu_plus_one(rmsnorm_no_affine(qbar.float(), reader.eps_norm) @ reader.Rq)
    k = elu_plus_one(rmsnorm_no_affine(kbar.float(), reader.eps_norm))
    length = q.shape[1]
    visible = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    visible = visible[None] & valid[:, None, :] & valid[:, :, None]
    weights = torch.where(visible, q @ k.transpose(-1, -2), 0.)
    mass = weights.sum(-1, keepdim=True)
    normalized = weights / mass.clamp_min(torch.finfo(weights.dtype).tiny)
    actual = weights / (mass + reader.eps_read)
    return {'self_weight': normalized.diagonal(dim1=-2, dim2=-1),
        'maximum_weight': normalized.max(-1).values,
        'n_eff': normalized.square().sum(-1).clamp_min(torch.finfo(weights.dtype).tiny).reciprocal(),
        'actual_self_weight': actual.diagonal(dim1=-2, dim2=-1),
        'raw_weight_sum': actual.sum(-1), 'denominator': mass.squeeze(-1) + reader.eps_read,
        'prefix_length': valid.long().cumsum(-1)}


@torch.no_grad()
def diagnose(model, examples, tok):
    """No dev or confirmation data. Caller preserves training/RNG state."""
    if model.config.steps != 1 or model.config.mode != 'inclusive':
        raise ValueError('diagnostics require inclusive S1')
    model.eval()
    totals, handles, current = {}, [], {}
    for index, layer in model.adapters().items():
        def hook(module, args, result, index=index, reader=layer.side.reader):
            current[index] = read_statistics(result[0], result[1], reader, current['valid'])
        handles.append(layer.side.projection.register_forward_hook(hook))
    try:
        for example in examples:
            batch = batch_tensors([example], tok.pad_token_id)
            valid = batch['attention_mask'].bool()
            current['valid'] = valid
            model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False)
            answer = torch.zeros_like(valid)
            answer[:, :-1] = batch['labels'][:, 1:] != -100
            for index in model.adapters():
                stats = current[index]
                for scope, mask in (('all_valid_input', valid), ('answer_prediction', answer & valid)):
                    for low, high in PREFIX_BINS:
                        chosen = mask & (stats['prefix_length'] >= low) & (stats['prefix_length'] <= high)
                        count = int(chosen.sum())
                        if not count:
                            continue
                        key = example['task'], index, scope, low, high
                        total = totals.setdefault(key, {'count': 0, 'sums': {}})
                        total['count'] += count
                        for name, values in stats.items():
                            if name != 'prefix_length':
                                total['sums'][name] = total['sums'].get(name, 0.) + values[chosen].double().sum().item()
    finally:
        for handle in handles:
            handle.remove()
    return {'scope': 'fixed first 64 I-G training examples; all arms and checkpoints share IDs',
        'records': [{'task': key[0], 'layer': key[1], 'scope': key[2], 'prefix_bin': list(key[3:]),
            'count': value['count'], 'means': {name: total/value['count'] for name, total in value['sums'].items()}}
            for key, value in sorted(totals.items())],
        'weight_convention': 'self/max/n_eff normalized to sum one; actual_self/raw_weight_sum include eps'}
