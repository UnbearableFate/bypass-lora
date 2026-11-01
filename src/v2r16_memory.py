"""E-0016 decoder-first and last-position latent-slot protocols.

Legacy V2 operators are unchanged. Extra supervision is used only by the
explicit teacher protocol and teacher evaluation controls. Marginal training
never reads target_fact_index.
"""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from v2_memory import (CONTROLS, Snapshot, answer_logits, build_support,
                       capture_projection, collate, control_snapshot, read_vectors,
                       slot_columns)
from v2_memory import train_forward as legacy_forward
from v2r16_objective import slot_marginal

ADDRESS = {'projection.A_q', 'projection.A_k', 'reader.Rq'}
DECODER = {'projection.A_v', 'reader.Ro', 'fusion.P'}
TF_CONTROLS = ('tf_normal', 'tf_shuffled', 'tf_wrong', 'zero')


def freeze_for_address(model):
    side = model.adapters()[14].side
    if set(dict(side.named_parameters())) != ADDRESS | DECODER:
        raise ValueError('E-0016 requires the six-parameter-group V2-03 adapter')
    for name, parameter in side.named_parameters():
        parameter.requires_grad_(name in ADDRESS)
        parameter.grad = None
    return sorted(ADDRESS)


def compact_snapshot(snapshot):
    """Public valid-slot order only; compact views retain the support graph."""
    state = snapshot.memory[14]
    if state.backend != 'soft_slot' or not bool((state.counts > 0).all()):
        raise ValueError('compact views require nonempty SoftmaxSlot memory')
    width = int(state.counts.max())
    keys, values, valid = [], [], []
    for row in range(state.batch_size):
        columns = state.slot_valid[row].nonzero().flatten()
        n = len(columns)
        keys.append(F.pad(state.slot_keys[row, columns], (0, 0, 0, width-n)))
        values.append(F.pad(state.slot_values[row, columns], (0, 0, 0, width-n)))
        valid.append(torch.arange(width, device=columns.device) < n)
    return Snapshot(snapshot.memory, torch.stack(keys), torch.stack(values), torch.stack(valid))


def single_slot(snapshot, fact_indices):
    """Keep exactly the selected fact without renumbering its original column."""
    state = snapshot.memory[14]
    columns = slot_columns(state.slot_valid, fact_indices)
    valid = torch.zeros_like(state.slot_valid).scatter(1, columns[:, None], True)
    memory = replace(state, slot_valid=valid, counts=torch.ones_like(state.counts))
    compact_valid = torch.zeros_like(snapshot.valid).scatter(1, fact_indices[:, None], True)
    return Snapshot({14: memory}, snapshot.keys, snapshot.values, compact_valid)


def _query(model, ids, mask, snapshot, *, strategy='standard', mode='normal', target=None,
           forced_values=None, diagnostics=False):
    """One question forward; a scoped hook replaces only the requested increment.

    For last-position protocols the pre-hook also masks native reader execution
    outside the last valid question position. The attention mask stays intact.
    The per-slot forward and its normal softmax address use identical pre-read
    query features, because the sole intervention layer is layer 14.
    """
    if (model.config.layers != (14,) or model.config.backend != 'soft_slot' or model.config.steps != 1
            or model.config.readout != 'increment' or model.config.fusion != 'F_out'):
        raise ValueError('E-0016 is single-layer, single-read SoftmaxSlot')
    if strategy not in ('standard', 'hard', 'soft', 'forced'):
        raise ValueError('unknown read strategy')
    side, layer = model.adapters()[14].side, model.adapters()[14]
    row = torch.arange(len(ids), device=ids.device)
    last = mask.long().sum(-1)-1
    target = torch.zeros_like(last) if target is None else target
    last_only = strategy != 'standard'
    read_mask = torch.zeros_like(mask).scatter(1, last[:, None], True) if last_only else mask
    cache, handles = {}, []
    if diagnostics:
        def residual(module, args, kwargs):
            cache['residual_norm'] = (args[0] if args else kwargs['hidden_states']).float().norm(dim=-1)
        handles.append(layer.register_forward_pre_hook(residual, with_kwargs=True))
    with capture_projection(side) as seen:
        def restrict_reads(module, args):
            if args[4] != 'query':
                raise ValueError('question-only hook reached a support forward')
            return (*args[:3], read_mask, *args[4:])

        def replace_read(module, args, output):
            z, state, accounting = output
            qbar = seen['q']
            oracle = mode if mode in ('oracle_key', 'oracle_value') else 'normal'
            raw, weights, scores, _ = read_vectors(qbar, snapshot, side.reader, 'soft_slot', target, oracle)
            eligible = snapshot.valid[:, None].expand_as(scores)
            log_scores = scores.masked_fill(~eligible, -torch.inf)
            empty = ~eligible.any(-1, keepdim=True)
            log_scores = torch.where(empty, torch.zeros_like(log_scores), log_scores)
            log_weights = log_scores.log_softmax(-1).masked_fill(~eligible, -torch.inf)
            if strategy == 'hard' and mode != 'oracle_value':
                chosen = weights.argmax(-1)
                raw = snapshot.values.gather(1, chosen[..., None].expand(-1, -1, qbar.shape[-1]))
                raw = raw * snapshot.valid.any(-1)[:, None, None]
            elif strategy == 'forced':
                if forced_values is None or forced_values.shape != qbar[:, 0].shape:
                    raise ValueError('forced per-slot values must match question rows')
                raw = forced_values[:, None].expand_as(qbar)
            if strategy in ('hard', 'forced') or mode in ('oracle_key', 'oracle_value'):
                # Match the original S=1 increment, including cancellation rounding.
                z = (qbar + side.reader.eta * (raw * side.reader.read_gain @ side.reader.Ro))-qbar
                z = z * read_mask[..., None]
            cache.update(weights=weights, log_weights=log_weights, z=z)
            if diagnostics:
                cache['fusion_ratio'] = (z @ side.fusion.P).norm(dim=-1) / cache['residual_norm'].clamp_min(1e-12)
            return z, state, accounting

        if last_only:
            handles.append(side.register_forward_pre_hook(restrict_reads))
        handles.append(side.register_forward_hook(replace_read))
        try:
            output = model(ids, mask, memory=snapshot.memory, phase='query')
        finally:
            for handle in handles:
                handle.remove()
    cache['last'] = last
    return output, cache


def _marginal_from_support(model, batch, snapshot, support_counts, *, diagnostics=False):
    n = snapshot.valid.shape[1]
    if not bool(snapshot.valid.all()):
        raise ValueError('marginal training requires equal nonempty fact counts')
    owners = torch.arange(len(batch.labels), device=batch.labels.device).repeat_interleave(n)
    expanded = snapshot.select(owners)
    output, cache = _query(model, batch.question_ids.index_select(0, owners),
                           batch.question_mask.index_select(0, owners), expanded,
                           strategy='forced', forced_values=snapshot.values.reshape(-1, snapshot.values.shape[-1]))
    logits = answer_logits(output, batch.question_mask.index_select(0, owners)).reshape(len(batch.labels), n, -1)
    last = cache['last']
    expanded_row = torch.arange(len(owners), device=last.device)
    log_weights = cache['log_weights'][expanded_row, last][::n]
    loss, mixture = slot_marginal(log_weights, logits, batch.labels, snapshot.valid)
    return loss, mixture, support_counts, output.accounting


def train_forward(model, batch, *, protocol='standard'):
    if protocol == 'standard':
        return legacy_forward(model, batch)
    if protocol not in ('teacher', 'marginal'):
        raise ValueError('unknown training protocol')
    support, support_counts = build_support(model, batch)
    snapshot = compact_snapshot(support.select(batch.owners))
    if protocol == 'marginal':
        # No target_fact_index or other slot labels enter this branch.
        return _marginal_from_support(model, batch, snapshot, support_counts)
    target = torch.tensor([r['query']['target_fact_index'] for r in batch.rows], device=batch.labels.device)
    snapshot = single_slot(snapshot, target)
    output = model(batch.question_ids, batch.question_mask, memory=snapshot.memory, phase='query')
    logits = answer_logits(output, batch.question_mask)
    return F.cross_entropy(logits, batch.labels), logits, support_counts, output.accounting


@torch.no_grad()
def evaluate_batch(model, batch, mode='normal', donor_batch=None, *, strategy='standard'):
    if model.training or mode not in (*CONTROLS, *TF_CONTROLS):
        raise ValueError('evaluation needs eval mode and a registered control')
    original, support_counts = build_support(model, batch)
    original = compact_snapshot(original).select(batch.owners)
    shuffled = mode in ('shuffled', 'tf_shuffled')
    if shuffled:
        if donor_batch is None or len(donor_batch.rows) != len(batch.rows):
            raise ValueError('shuffled requires one explicit global donor per query')
        if any(a['episode_id'] == b['episode_id'] for a, b in zip(batch.rows, donor_batch.rows)):
            raise ValueError('donor must differ from recipient')
        donor, _ = build_support(model, donor_batch)
        snapshot = compact_snapshot(donor).select(donor_batch.owners)
    else:
        snapshot = control_snapshot(original, mode if mode in CONTROLS else 'normal')
    target = torch.tensor([r['query']['target_fact_index'] for r in batch.rows], device=batch.labels.device)
    if mode.startswith('tf_'):
        chosen = (target+1) % original.valid.sum(-1) if mode == 'tf_wrong' else target
        snapshot = single_slot(snapshot, chosen)
    if strategy == 'marginal' and mode != 'zero':
        loss, logits, _, _ = _marginal_from_support(model, batch, snapshot, support_counts)
        _, cache = _query(model, batch.question_ids, batch.question_mask, snapshot,
                           strategy='soft', mode=mode, target=target, diagnostics=True)
    else:
        read_strategy = 'soft' if strategy == 'marginal' else strategy
        output, cache = _query(model, batch.question_ids, batch.question_mask, snapshot,
                               strategy=read_strategy, mode=mode, target=target, diagnostics=True)
        logits = answer_logits(output, batch.question_mask)
    if not torch.isfinite(logits).all():
        raise RuntimeError('nonfinite full-vocabulary evaluation logits')
    weights, last = cache['weights'], cache['last']
    row = torch.arange(len(target), device=target.device)
    neff = torch.where(weights.sum(-1) > 0, 1/weights.square().sum(-1).clamp_min(1e-30), 0)
    all_weight = weights.gather(-1, target[:, None, None].expand(-1, weights.shape[1], 1)).squeeze(-1)
    counts = batch.question_mask.sum(-1)
    argmax = torch.where(snapshot.valid.any(-1), weights[row, last].argmax(-1), -1)
    return {
        'all_target_weight': None if shuffled else ((all_weight*batch.question_mask).sum(-1)/counts).tolist(),
        'last_target_weight': None if shuffled else all_weight[row, last].tolist(),
        'all_n_eff': ((neff*batch.question_mask).sum(-1)/counts).tolist(),
        'last_n_eff': neff[row, last].tolist(), 'last_argmax': argmax.tolist(),
        'fusion_to_residual_all': ((cache['fusion_ratio']*batch.question_mask).sum(-1)/counts).tolist(),
        'fusion_to_residual_last': cache['fusion_ratio'][row, last].tolist(),
    }, logits


@torch.no_grad()
def evaluate(model, rows, pad_id, output, split, controls=CONTROLS, prior_token=None, *, strategy='standard'):
    """E-0016 per-example diagnostics, including position-0 versus later facts."""
    from v2_train import distribution, write_json
    if split not in ('dev', 'calibration') or len(rows) < 2 or any(r['identity']['split'] != split for r in rows):
        raise ValueError('only released dev/calibration data with >=2 rows')
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, predictions = {}, {}
    for mode in controls:
        records = []
        for start in range(0, len(rows), 16):
            selected = rows[start:start+16]
            batch = collate(selected, pad_id, next(model.parameters()).device)
            donors = None
            if mode in ('shuffled', 'tf_shuffled'):
                donors = collate([rows[(i-1) % len(rows)] for i in range(start, start+len(selected))],
                                pad_id, next(model.parameters()).device)
            details, logits = evaluate_batch(model, batch, mode, donors, strategy=strategy)
            losses = (-logits.gather(1, batch.labels[:, None]).squeeze(1)
                      if strategy == 'marginal' and mode != 'zero' else
                      F.cross_entropy(logits, batch.labels, reduction='none'))
            pred = logits.argmax(-1).tolist()
            for i, episode in enumerate(selected):
                record = {'episode_id': episode['episode_id'], 'target': int(batch.labels[i]),
                          'prediction': pred[i], 'correct': pred[i] == int(batch.labels[i]), 'nll': float(losses[i]),
                          'question_tokens': int(batch.question_mask[i].sum()),
                          'target_fact_position': episode['query']['target_fact_index'],
                          'relation': episode['query']['relation'], 'n_facts': len(episode['facts'])}
                for name, values in details.items():
                    record[name] = None if values is None else values[i]
                records.append(record)
        def summarize(group):
            summary = {'count': len(group), 'correct_count': sum(r['correct'] for r in group),
                       'em': sum(r['correct'] for r in group)/len(group),
                       'nll': sum(r['nll'] for r in group)/len(group)}
            if prior_token is not None:
                summary['prior_em'] = sum(r['target'] == prior_token for r in group)/len(group)
            token_count = sum(r['question_tokens'] for r in group)
            summary['diagnostics'] = {
                name: sum(r[field]*r['question_tokens'] for r in group)/token_count
                for name, field in (('address.target_weight', 'all_target_weight'), ('address.n_eff', 'all_n_eff'))
                if all(r[field] is not None for r in group)}
            valid_argmax = [r for r in group if r['last_argmax'] >= 0]
            width = max(r['n_facts'] for r in group)
            counts = [sum(r['last_argmax'] == i for r in valid_argmax) for i in range(width)]
            summary['last_position'] = {'argmax_slot_counts': counts,
                'argmax_most_common_slot_rate': max(counts)/len(valid_argmax) if valid_argmax else None,
                'target_weight_mean': (sum(r['last_target_weight'] for r in group)/len(group)
                                       if all(r['last_target_weight'] is not None for r in group) else None),
                'n_eff_mean': sum(r['last_n_eff'] for r in group)/len(group),
                'fusion_to_residual': distribution([r['fusion_to_residual_last'] for r in group])}
            summary['all_question_positions'] = {'fusion_to_residual': distribution([r['fusion_to_residual_all'] for r in group])}
            return summary
        summary = summarize(records)
        groups = {'position0': [r for r in records if r['target_fact_position'] == 0],
                  'position1plus': [r for r in records if r['target_fact_position'] > 0]}
        summary['position_strata'] = {name: summarize(group) for name, group in groups.items() if group}
        summary['protocol'] = strategy
        summary['address_weight_scope'] = ('all question positions (counterfactual outside last position); primary gate uses last'
                                          if strategy != 'standard' else 'all valid question positions')
        if mode in ('shuffled', 'tf_shuffled'):
            summary['donor_rule'] = 'global ordered split: recipient i receives support (i-1) modulo split size'
        with (output/f'{split}.{mode}.predictions.jsonl').open('w') as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
        summaries[mode], predictions[mode] = summary, records
    write_json(output/f'{split}.metrics.json', summaries)
    return summaries, predictions
