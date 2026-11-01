"""M-line episode protocol and offline diagnostics; no oracle metadata in training.

The deployed path calls BranchModel unchanged. Short-lived observation hooks retain
projected support facts for offline diagnostics (matrix memory has no slots).
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from bypass_core import MemoryState, centered_cosine_scores, elu_plus_one, rmsnorm_no_affine
from bypass_model import AdapterConfig


CONTROLS = ('normal', 'zero', 'shuffled', 'permuted', 'oracle_key', 'oracle_value')
# E-0015@3 read substitutions (diagnostic only; CONTROLS keep their V2 behaviour).
ISOLATED_CONTROLS = ('iso_kv', 'iso_v', 'iso_oracle_value', 'iso_wrong_oracle')
DIAGNOSTIC_CONTROLS = ('last_only', 'oracle_last', 'oracle_last_only', 'uniform', 'last_only_shuffled',
                       'wrong_oracle') + ISOLATED_CONTROLS
# Controls that keep an increment only at the last question position or replace it there.
LAST_POSITION_CONTROLS = ('last_only', 'oracle_last', 'oracle_last_only', 'last_only_shuffled')
# Controls whose increment is recomputed from a substituted read at every question position.
RECOMPUTED_CONTROLS = ('oracle_key', 'oracle_value', 'iso_oracle_value', 'wrong_oracle', 'iso_wrong_oracle', 'uniform')
DONOR_CONTROLS = ('shuffled', 'last_only_shuffled')
# E-0015@3 magnitude sweep: '<read>@a<alpha>' scales the increment at all question
# positions (read normal or oracle_value); 'nonfinal@a<alpha>' scales only the
# non-final positions under the normal read.
SWEEP_ALPHAS = (.1, .3)
SWEEP_CONTROLS = tuple(f'{read}@a{alpha}' for read in ('normal', 'oracle_value', 'nonfinal') for alpha in SWEEP_ALPHAS)


def parse_control(mode):
    """(read mode, alpha, scope) of a control name; scope is 'all' or 'nonfinal'."""
    if mode in CONTROLS + DIAGNOSTIC_CONTROLS:
        return mode, None, None
    if mode in SWEEP_CONTROLS:
        read, alpha = mode.split('@a')
        return ('normal', float(alpha), 'nonfinal') if read == 'nonfinal' else (read, float(alpha), 'all')
    raise ValueError('unknown control')
BACKENDS = ('kernel_matrix', 'soft_slot', 'centered_cos_slot')


def config(tap='direct_hidden', backend='kernel_matrix', fusion='F_out', prior_rank=0):
    if tap not in ('direct_hidden', 'qknorm_pre_rope') or backend not in BACKENDS or fusion not in ('F_out', 'C_up', 'C_dual'):
        raise ValueError('not a V2 arm')
    return AdapterConfig(layers=(14,), rank=128, branch='memory', tap=tap,
                         backend=backend, fusion=fusion, steps=1, slot_capacity=16,
                         prior_rank=prior_rank)


def pad(rows, pad_id, device):
    if not rows or any(not row for row in rows):
        raise ValueError('empty sequence')
    ids = torch.full((len(rows), max(map(len, rows))), pad_id, dtype=torch.long, device=device)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.as_tensor(row, device=device)
        valid[i, :len(row)] = True
    return ids, valid


@dataclass
class EpisodeBatch:
    support_ids: torch.Tensor
    support_mask: torch.Tensor
    write_mask: torch.Tensor
    question_ids: torch.Tensor
    question_mask: torch.Tensor
    owners: torch.Tensor
    labels: torch.Tensor
    rows: list

    @property
    def input_tokens(self):
        return int(self.support_mask.sum() + self.question_mask.sum())


def collate(episodes, pad_id, device='cuda', owners=None):
    """Deduplicate supports by explicit owner; questions may repeat/reorder owners.

    Only delimiter positions (public boundaries) and answer labels are used by the
    default training path. target_fact_index is read by evaluate_batch and, only
    with the E-0014 R5 extra-supervision option, by train_forward.
    """
    if not episodes:
        raise ValueError('empty batch')
    identities = [e['episode_id'] for e in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError('support owners must have unique episode_id')
    si, sm = pad([e['support_input_ids'] for e in episodes], pad_id, device)
    wm = torch.zeros_like(sm)
    for i, episode in enumerate(episodes):
        positions = [f['delimiter_token_index'] for f in episode['facts']]
        if not 1 <= len(positions) <= 16 or len(set(positions)) != len(positions):
            raise ValueError('facts require 1..16 distinct delimiters')
        if positions != sorted(positions) or any(p < 0 or p >= len(episode['support_input_ids']) for p in positions):
            raise ValueError('delimiter positions must follow support order')
        wm[i, positions] = True
    owners = list(range(len(episodes))) if owners is None else list(owners)
    if not owners or any(i < 0 or i >= len(episodes) for i in owners):
        raise ValueError('invalid episode owner')
    rows = [episodes[i] for i in owners]
    qi, qm = pad([e['question_input_ids'] for e in rows], pad_id, device)
    return EpisodeBatch(si, sm, wm, qi, qm, torch.tensor(owners, device=device),
                        torch.tensor([e['query']['answer_token_id'] for e in rows], device=device), rows)


@dataclass
class Snapshot:
    memory: dict
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    valid: torch.Tensor | None = None

    def select(self, owners):
        return Snapshot({i: s.reorder(owners) for i, s in self.memory.items()},
                        *(None if x is None else x.index_select(0, owners) for x in (self.keys, self.values, self.valid)))


@contextmanager
def capture_projection(side):
    seen = {}
    handle = side.projection.register_forward_hook(lambda module, args, out: seen.update(q=out[0], k=out[1], v=out[2]))
    try:
        yield seen
    finally:
        handle.remove()


def build_support(model, batch, observe=False):
    side = model.adapters()[14].side
    if not observe:
        out = model(batch.support_ids, batch.support_mask, phase='support', write_mask=batch.write_mask)
        return Snapshot(out.cache.memory), out.accounting
    if model.training or torch.is_grad_enabled():
        raise ValueError('support observation is offline only')
    with capture_projection(side) as seen:
        out = model(batch.support_ids, batch.support_mask, phase='support', write_mask=batch.write_mask)
    # Compact public delimiters, unlike core slots which may include masked columns.
    n = int(batch.write_mask.sum(-1).max())
    b, _, r = seen['k'].shape
    keys = seen['k'].new_zeros(b, n, r)
    values = keys.clone()
    valid = torch.zeros(b, n, device=keys.device, dtype=torch.bool)
    for i in range(b):
        k = seen['k'][i, batch.write_mask[i]]
        v = seen['v'][i, batch.write_mask[i]]
        keys[i, :len(k)] = elu_plus_one(rmsnorm_no_affine(k, side.reader.eps_norm))
        values[i, :len(v)] = v
        valid[i, :len(k)] = True
    return Snapshot(out.cache.memory, keys, values, valid), out.accounting


def observe_writes(model, ids, mask, write_mask):
    """Compact per-delimiter write tensors of a support pass (offline diagnostics).

    Returns transformed keys (as written to slots), values (A_v output, as stored),
    the frozen projection input at the value tap (E-0015 Pf) and validity, each in
    support order. Unlike build_support, no memory state is returned.
    """
    if model.training or torch.is_grad_enabled():
        raise ValueError('support observation is offline only')
    side = model.adapters()[14].side
    seen = {}
    handle = side.projection.register_forward_pre_hook(lambda module, args: seen.update(features=args[2]))
    try:
        with capture_projection(side) as projected:
            model(ids, mask, phase='support', write_mask=write_mask)
    finally:
        handle.remove()
    n = int(write_mask.sum(-1).max())
    b, _, r = projected['k'].shape
    keys = projected['k'].new_zeros(b, n, r)
    values = keys.clone()
    features = seen['features'].new_zeros(b, n, seen['features'].shape[-1])
    valid = torch.zeros(b, n, device=keys.device, dtype=torch.bool)
    for i in range(b):
        count = int(write_mask[i].sum())
        keys[i, :count] = elu_plus_one(rmsnorm_no_affine(projected['k'][i, write_mask[i]], side.reader.eps_norm))
        values[i, :count] = projected['v'][i, write_mask[i]]
        features[i, :count] = seen['features'][i, write_mask[i]]
        valid[i, :count] = True
    return {'keys': keys, 'values': values, 'features': features, 'valid': valid}


def isolated_writes(model, rows, pad_id, device):
    """E-0015 isolated encoding: each fact of each row encoded alone, in support order.

    rows[i]['_isolated'] lists, per fact in support order, the single-fact support
    (E=1 format) as {'ids', 'delimiter'}. The resulting slot j of row i holds fact j.
    """
    sequences = [(i, j, item) for i, row in enumerate(rows) for j, item in enumerate(row['_isolated'])]
    if any(len(row['_isolated']) != len(row['facts']) for row in rows):
        raise ValueError('isolated encoding needs one sequence per fact')
    ids, mask = pad([item['ids'] for _, _, item in sequences], pad_id, device)
    write_mask = torch.zeros_like(mask)
    for k, (_, _, item) in enumerate(sequences):
        if not 0 <= item['delimiter'] < len(item['ids']):
            raise ValueError('isolated delimiter outside its sequence')
        write_mask[k, item['delimiter']] = True
    single = observe_writes(model, ids, mask, write_mask)
    n = max(len(row['facts']) for row in rows)
    result = {}
    for name in ('keys', 'values', 'features'):
        x = single[name]
        result[name] = x.new_zeros(len(rows), n, x.shape[-1])
    result['valid'] = torch.zeros(len(rows), n, dtype=torch.bool, device=device)
    for k, (i, j, _) in enumerate(sequences):
        for name in ('keys', 'values', 'features'):
            result[name][i, j] = single[name][k, 0]
        result['valid'][i, j] = True
    return result


def isolated_snapshot(original, isolated, mode):
    """iso_kv: isolated keys and values; iso_v / iso_oracle_value: contextual keys, isolated values."""
    if mode not in ISOLATED_CONTROLS:
        raise ValueError('not an isolated-encoding control')
    if not torch.equal(isolated['valid'], original.valid):
        raise ValueError('isolated slots must match the contextual support slots')
    keys = isolated['keys'] if mode == 'iso_kv' else original.keys
    values = isolated['values']
    states = {i: memory_from_slots(keys, values, original.valid, s.backend) for i, s in original.memory.items()}
    return Snapshot(states, keys, values, original.valid)


def answer_logits(output, mask):
    last = mask.long().sum(-1) - 1  # collate uses right padding
    return output.logits[torch.arange(len(last), device=last.device), last].float()


def slot_columns(slot_valid, target):
    """Core slot column of each row's target fact: its (target+1)-th valid slot."""
    order = slot_valid.long().cumsum(-1) - 1
    match = slot_valid & (order == target[:, None])
    if not bool((match.sum(-1) == 1).all()):
        raise ValueError('target fact index has no stored slot')
    return match.long().argmax(-1)


def address_log_weight(seen_q, memory, reader, question_mask, target):
    """log w_target of SoftmaxSlot at each row's last valid question position.

    Recomputes the S=1 query and scores from the same differentiable tensors the
    deployed read uses (no other position or slot enters).
    """
    if memory.backend != 'soft_slot':
        raise ValueError('the address loss is defined for soft_slot only')
    row = torch.arange(len(target), device=target.device)
    last = question_mask.long().sum(-1) - 1
    query = elu_plus_one(rmsnorm_no_affine(seen_q[row, last], reader.eps_norm) @ reader.Rq)
    scores = torch.einsum('br,bnr->bn', query, memory.slot_keys) / memory.rank ** .5
    scores = scores.masked_fill(~memory.slot_valid, -torch.inf)
    return scores.log_softmax(-1)[row, slot_columns(memory.slot_valid, target)]


def train_forward(model, batch, *, zero_memory=False, address_weight=0., log=None):
    """Answer CE by default; no address, value, path or oracle input.

    E-0014 options (default off): zero_memory reads an empty memory after the
    normal support pass (R4 stage 1); address_weight > 0 adds the extra-supervision
    term address_weight * mean(-log w_target) at the last question position (R5).
    """
    support, accounting = build_support(model, batch)
    memory = support.select(batch.owners).memory
    if zero_memory:
        memory = {i: s.reset() for i, s in memory.items()}
    if address_weight:
        if zero_memory or address_weight < 0:
            raise ValueError('address loss needs real memory and a positive weight')
        side = model.adapters()[14].side
        target = torch.tensor([r['query']['target_fact_index'] for r in batch.rows], device=batch.labels.device)
        with capture_projection(side) as seen:
            output = model(batch.question_ids, batch.question_mask, memory=memory, phase='query')
        address = -address_log_weight(seen['q'], memory[14], side.reader, batch.question_mask, target).mean()
    else:
        output = model(batch.question_ids, batch.question_mask, memory=memory, phase='query')
    logits = answer_logits(output, batch.question_mask)
    loss = F.cross_entropy(logits, batch.labels)
    if address_weight:
        if log is not None:
            log.update(answer_ce=float(loss.detach()), address_loss=float(address.detach()))
        loss = loss + address_weight * address
    return loss, logits, accounting, output.accounting


def memory_from_slots(keys, values, valid, backend):
    state = MemoryState.empty(len(keys), keys.shape[-1], backend, keys.device)
    for i in range(keys.shape[1]):
        state = state.write(keys[:, i], values[:, i], valid[:, i])
    return state


def control_snapshot(snapshot, mode):
    if mode not in CONTROLS:
        raise ValueError('unknown control')
    if mode == 'zero':
        return Snapshot({i: s.reset() for i, s in snapshot.memory.items()},
                        torch.zeros_like(snapshot.keys), torch.zeros_like(snapshot.values), torch.zeros_like(snapshot.valid))
    if mode == 'shuffled':
        if len(snapshot.keys) < 2:
            raise ValueError('shuffled memory requires at least two distinct owners')
        # A deterministic derangement of support owners, BEFORE selecting questions.
        indices = torch.arange(len(snapshot.keys), device=snapshot.keys.device).roll(1)
        return snapshot.select(indices)
    if mode == 'permuted':
        values = snapshot.values.clone()
        for i, valid in enumerate(snapshot.valid):
            n = int(valid.sum())
            if n < 2:
                raise ValueError('permuted association needs at least two facts')
            values[i, :n] = values[i, :n].roll(1, 0)
        states = {i: memory_from_slots(snapshot.keys, values, snapshot.valid, s.backend)
                  for i, s in snapshot.memory.items()}
        return Snapshot(states, snapshot.keys, values, snapshot.valid)
    return snapshot


def read_vectors(qbar, snapshot, reader, backend, target, mode='normal', last=None):
    """Independent explicit-slot read, used only for diagnostics/oracle evaluation.

    E-0015 modes return the read actually used per position: oracle_last(_only) the
    target value at the last question position (``last``); last_only,
    last_only_shuffled and oracle_last_only zero at the other positions; uniform the
    equal-weight mean of valid slot values everywhere; iso_oracle_value the target
    value of the (isolated) snapshot everywhere; (iso_)wrong_oracle the value of
    fact (target+1) mod E everywhere.
    """
    b, t, _ = qbar.shape
    row = torch.arange(b, device=qbar.device)
    q = elu_plus_one(rmsnorm_no_affine(qbar, reader.eps_norm) @ reader.Rq)
    if mode == 'oracle_key':
        # Substitute the actual stored positive key, not a twice-transformed key.
        q = snapshot.keys[row, target][:, None].expand_as(q)
    if backend == 'centered_cos_slot':
        scores = centered_cosine_scores(q, snapshot.keys, snapshot.valid, reader.temperature())
    else:
        scores = torch.einsum('btr,bnr->btn', q, snapshot.keys)
    valid = snapshot.valid[:, None].expand_as(scores)
    if backend in ('soft_slot', 'centered_cos_slot'):
        if backend == 'soft_slot':
            scores = scores / q.shape[-1] ** .5
        logits = scores.masked_fill(~valid, -torch.inf)
        empty = ~valid.any(-1, keepdim=True)
        logits = torch.where(empty, torch.zeros_like(logits), logits)
        weights = logits.softmax(-1).masked_fill(~valid, 0)
        # Softmax denominator is reported in stable shifted-score coordinates.
        denominator = ((logits - logits.max(-1, keepdim=True).values).exp() * valid).sum(-1)
    else:
        scores = scores.masked_fill(~valid, 0)
        denominator = scores.sum(-1) + reader.eps_read
        weights = scores / denominator[..., None]
    raw = weights @ snapshot.values
    if mode == 'oracle_value':
        raw = snapshot.values[row, target][:, None].expand_as(raw)
    elif mode == 'iso_oracle_value':
        raw = snapshot.values[row, target][:, None].expand_as(raw)
    elif mode in ('wrong_oracle', 'iso_wrong_oracle'):
        count = snapshot.valid.sum(-1)
        if not bool((count >= 2).all()):
            raise ValueError('the wrong-fact oracle needs at least two facts')
        raw = snapshot.values[row, (target + 1) % count][:, None].expand_as(raw)
    elif mode == 'uniform':
        count = snapshot.valid.sum(-1, keepdim=True)
        if not bool((count > 0).all()):
            raise ValueError('uniform read needs at least one valid slot')
        mean = (snapshot.values * snapshot.valid[..., None]).sum(1) / count
        raw = mean[:, None].expand_as(raw)
    elif mode in LAST_POSITION_CONTROLS:
        if last is None:
            raise ValueError('last-position controls need the last question position')
        at_last = torch.zeros(b, t, 1, dtype=torch.bool, device=raw.device)
        at_last[row, last] = True
        replacement = (raw if mode in ('last_only', 'last_only_shuffled')
                       else snapshot.values[row, target][:, None].expand_as(raw))
        others = raw if mode == 'oracle_last' else torch.zeros_like(raw)
        raw = torch.where(at_last, replacement, others)
    return raw, weights, scores, denominator


def read_metrics(raw, weights, scores, denominator, target_value, target, valid_question, fact_valid,
                 address_target_available=True):
    b, t, n = weights.shape
    target_indices = target[:, None, None].expand(b, t, 1)
    prob = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
    entropy = -(prob * prob.clamp_min(1e-30).log()).sum(-1)
    neff = torch.where(prob.sum(-1) > 0, 1 / prob.square().sum(-1).clamp_min(1e-30), 0)
    eligible = fact_valid[:, None].expand_as(scores)
    target_value = target_value[:, None].expand_as(raw)
    measurements = {
        'address.entropy': entropy, 'address.n_eff': neff,
        'read.normalized_error': (raw-target_value).square().sum(-1)/(target_value.square().sum(-1)+1e-6),
        'read.direction_error': 1-F.cosine_similarity(raw, target_value, dim=-1, eps=1e-6),
        'read.raw_norm': raw.norm(dim=-1), 'read.target_norm': target_value.norm(dim=-1),
    }
    if address_target_available:
        masked_scores = scores.masked_fill(~eligible, -torch.inf)
        other = masked_scores.scatter(-1, target_indices, -torch.inf).max(-1).values
        target_score = scores.gather(-1, target_indices).squeeze(-1)
        # Single-fact margin is undefined; preserve the documented zero convention.
        margin = torch.where(torch.isfinite(other), target_score - other, 0)
        measurements.update({
            'address.top1': ((masked_scores.argmax(-1) == target[:, None]) & eligible.any(-1)).float(),
            'address.margin': margin,
            'address.target_weight': weights.gather(-1, target_indices).squeeze(-1),
        })
    result = {k: float(v[valid_question].mean()) for k, v in measurements.items()}
    result['address.target_available'] = address_target_available
    den = denominator[valid_question]
    result.update({f'read.denominator.p{p}': float(torch.quantile(den, p/100)) for p in (0, 50, 95, 100)})
    return result


@torch.no_grad()
def evaluate_batch(model, batch, mode='normal', donor_batch=None):
    if model.training:
        raise ValueError('diagnostics/oracles require model.eval()')
    mode, alpha, scope = parse_control(mode)
    if model.config.steps != 1 or model.config.readout != 'increment' or model.config.layers != (14,):
        raise ValueError('offline diagnostics require the V2 single-layer single-read protocol')
    original, _ = build_support(model, batch, observe=True)
    selected_original = original.select(batch.owners)
    if donor_batch is not None:
        if mode not in DONOR_CONTROLS or len(donor_batch.rows) != len(batch.rows):
            raise ValueError('explicit donors require shuffled mode and one donor per query')
        if any(a['episode_id'] == b['episode_id'] for a, b in zip(batch.rows, donor_batch.rows)):
            raise ValueError('shuffled donor must differ from recipient')
        donor, _ = build_support(model, donor_batch, observe=True)
        snapshot = donor.select(donor_batch.owners)
    elif mode in ISOLATED_CONTROLS:
        if not torch.equal(batch.owners, torch.arange(len(batch.rows), device=batch.owners.device)):
            raise ValueError('isolated controls need one question per support owner, in order')
        # Right padding under the causal mask: the pad id never reaches valid positions.
        isolated = isolated_writes(model, batch.rows, 0, batch.support_ids.device)
        snapshot = isolated_snapshot(original, isolated, mode).select(batch.owners)
    else:
        # Without explicit donors last_only_shuffled uses shuffled's in-batch donors; the
        # other diagnostic controls substitute the read of the original memory.
        memory_mode = {'last_only_shuffled': 'shuffled'}.get(mode, 'normal' if mode in DIAGNOSTIC_CONTROLS else mode)
        snapshot = control_snapshot(original, memory_mode).select(batch.owners)
    target = torch.tensor([r['query']['target_fact_index'] for r in batch.rows], device=batch.labels.device)
    row = torch.arange(len(target), device=target.device)
    if ((target < 0) | (target >= selected_original.valid.sum(-1))).any():
        raise ValueError('invalid target fact index')
    target_value = selected_original.values[row, target]
    adapted_layer = model.adapters()[14]
    side = adapted_layer.side
    details = {}
    last = batch.question_mask.long().sum(-1) - 1
    norms = {}

    def residual_in(module, args, kwargs):
        norms['residual'] = (args[0] if args else kwargs['hidden_states']).float().norm(dim=-1)

    def post_attention(module, args):
        norms['post_attention'] = args[0].float().norm(dim=-1)

    with capture_projection(side) as seen:
        def observe(module, args, output):
            z, state, accounting = output
            raw, weights, scores, denominator = read_vectors(seen['q'], snapshot, side.reader, model.config.backend,
                                                             target, mode, last)
            at_last = torch.zeros_like(batch.question_mask)
            at_last[row, last] = True
            at_last = at_last[..., None]
            if mode in RECOMPUTED_CONTROLS:
                # Match S=1 increment's arithmetic, including cancellation rounding.
                qbar = seen['q']
                z = (qbar + side.reader.eta * (raw * side.reader.read_gain @ side.reader.Ro)) - qbar
                z = z * batch.question_mask[..., None]
            elif mode in LAST_POSITION_CONTROLS:
                # Replace only the declared positions: the last question token gets the
                # deployed (last_only, last_only_shuffled) or oracle increment, the others
                # keep the deployed increment (oracle_last) or receive none.
                if mode in ('last_only', 'last_only_shuffled'):
                    replacement = z
                else:
                    qbar = seen['q']
                    replacement = ((qbar + side.reader.eta * (raw * side.reader.read_gain @ side.reader.Ro)) - qbar)
                others = z if mode == 'oracle_last' else torch.zeros_like(z)
                z = torch.where(at_last, replacement, others) * batch.question_mask[..., None]
            factor = 1.
            if alpha is not None:
                # E-0015@3 magnitude sweep: scale the increment at the declared positions.
                factor = torch.full_like(z[..., :1], alpha)
                if scope == 'nonfinal':
                    factor = torch.where(at_last, torch.ones_like(factor), factor)
                z = z * factor
            details.update(read_metrics(raw, weights, scores, denominator, target_value, target,
                                       batch.question_mask, snapshot.valid,
                                       address_target_available=mode not in DONOR_CONTROLS))
            if mode in DONOR_CONTROLS:
                details['address.target_unavailable_reason'] = 'recipient target is undefined in donor memory'
            details['output.z_norm'] = float(z[batch.question_mask].norm(dim=-1).mean())
            for name in ('P', 'C_g', 'C_u'):
                if hasattr(side.fusion, name):
                    delta = z @ getattr(side.fusion, name)
                    details['output.'+name+'_norm'] = float(delta[batch.question_mask].norm(dim=-1).mean())
            details['read.used_norm'] = float((raw * side.reader.read_gain)[batch.question_mask].norm(dim=-1).mean())
            details['read.reconstruction_max_abs'] = float((z - ((seen['q'] + side.reader.eta*(raw * side.reader.read_gain @ side.reader.Ro)) - seen['q']) * batch.question_mask[..., None] * factor).abs().max())
            details['_clean_vectors'] = target_value.detach().cpu()
            details['_read_vectors'] = raw[row, batch.question_mask.sum(-1)-1].detach().cpu()
            # E-0014 last-question-position read and E-0012 L4 increment geometry.
            last_weights = weights[row, last]
            has_slots = snapshot.valid.any(-1)
            details['_last_argmax'] = torch.where(has_slots, last_weights.argmax(-1), -1).tolist()
            prob = last_weights / last_weights.sum(-1, keepdim=True).clamp_min(1e-30)
            details['_last_n_eff'] = torch.where(has_slots, 1 / prob.square().sum(-1).clamp_min(1e-30), 0).tolist()
            details['_last_target_weight'] = (None if mode in DONOR_CONTROLS
                                              else last_weights[row, target].tolist())
            if side.fusion.kind == 'F_out':
                norms['fusion'] = (z @ side.fusion.P).float().norm(dim=-1)
                # E-0015 Pz input: the FP32 F_out increment at the last question position.
                details['_last_fusion'] = (z[row, last] @ side.fusion.P).float().cpu()
            if side.prior is not None:
                norms['prior'] = side.prior(args[0], args[3]).float().norm(dim=-1)
            return z, state, accounting
        handles = [side.register_forward_hook(observe),
                   adapted_layer.register_forward_pre_hook(residual_in, with_kwargs=True),
                   adapted_layer.original.post_attention_layernorm.register_forward_pre_hook(post_attention)]
        try:
            out = model(batch.question_ids, batch.question_mask, memory=snapshot.memory, phase='query')
        finally:
            for handle in handles:
                handle.remove()
    # Fusion-increment / residual norm ratios (E-0012 L4: F_out FP32 z@P over the
    # layer-14 input residual; post-attention denominator also reported).
    for name in ('fusion', 'prior'):
        if name in norms:
            for denominator in ('residual', 'post_attention'):
                ratio = norms[name] / norms[denominator].clamp_min(1e-12)
                details[f'_{name}_to_{denominator}_last'] = ratio[row, last].tolist()
                details[f'_{name}_to_{denominator}_all'] = ratio[batch.question_mask].cpu()
    logits = answer_logits(out, batch.question_mask)
    if not torch.isfinite(logits).all():
        raise RuntimeError('nonfinite full-vocabulary evaluation logits')
    losses = F.cross_entropy(logits, batch.labels, reduction='none')
    correct = (logits.argmax(-1) == batch.labels).float()
    details.update(nll=float(losses.mean()), em=float(correct.mean()))
    for kind in ('relation', 'write_age'):
        groups = {}
        for i, episode in enumerate(batch.rows):
            group = episode['query']['relation'] if kind == 'relation' else str(len(episode['facts'])-1-int(target[i]))
            groups.setdefault(group, []).append(i)
        details[kind] = {g: {'nll': float(losses[ix].mean()), 'em': float(correct[ix].mean()), 'count': len(ix)} for g, ix in groups.items()}
    for field in ('M', 'c'):
        details['state.'+field+'_norm'] = float(getattr(snapshot.memory[14], field).flatten(1).norm(dim=-1).mean())
    return details, logits


def independent_probes(train_clean, train_read, train_labels, test_clean, test_read, test_labels, ridge=1.):
    """Two independently fitted linear probes, with held-out evaluation only.

    This is a diagnostic closed-form ridge classifier, never the answer decoder.
    Caller must provide disjoint train/test episode identities.
    """
    classes = torch.unique(train_labels, sorted=True)
    y = (train_labels[:, None] == classes).float()
    def fit(x):
        x = torch.cat((x.float(), torch.ones(len(x), 1, device=x.device)), -1)
        return torch.linalg.solve(x.T @ x + ridge*torch.eye(x.shape[1], device=x.device), x.T @ y), x
    clean_w, _ = fit(train_clean)
    read_w, _ = fit(train_read)
    def accuracy(x, w):
        x = torch.cat((x.float(), torch.ones(len(x), 1, device=x.device)), -1)
        return float((classes[(x @ w).argmax(-1)] == test_labels).float().mean())
    return {'clean_to_clean': accuracy(test_clean, clean_w), 'read_to_read': accuracy(test_read, read_w),
            'clean_to_read': accuracy(test_read, clean_w), 'train_examples': len(train_labels), 'test_examples': len(test_labels)}
