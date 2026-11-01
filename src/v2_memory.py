"""M-line episode protocol and offline diagnostics; no oracle metadata in training.

The deployed path calls BranchModel unchanged. Short-lived observation hooks retain
projected support facts for offline diagnostics (matrix memory has no slots).
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from bypass_core import MemoryState, elu_plus_one, rmsnorm_no_affine
from bypass_model import AdapterConfig


CONTROLS = ('normal', 'zero', 'shuffled', 'permuted', 'oracle_key', 'oracle_value')


def config(tap='direct_hidden', backend='kernel_matrix', fusion='F_out'):
    if tap not in ('direct_hidden', 'qknorm_pre_rope') or backend not in ('kernel_matrix', 'soft_slot') or fusion not in ('F_out', 'C_up', 'C_dual'):
        raise ValueError('not a V2 arm')
    return AdapterConfig(layers=(14,), rank=128, branch='memory', tap=tap,
                         backend=backend, fusion=fusion, steps=1, slot_capacity=16)


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
    training path. target_fact_index is accessed exclusively by evaluate_batch.
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


def answer_logits(output, mask):
    last = mask.long().sum(-1) - 1  # collate uses right padding
    return output.logits[torch.arange(len(last), device=last.device), last].float()


def train_forward(model, batch):
    """Only answer CE; no address, value, path or oracle input."""
    support, accounting = build_support(model, batch)
    memory = support.select(batch.owners).memory
    output = model(batch.question_ids, batch.question_mask, memory=memory, phase='query')
    logits = answer_logits(output, batch.question_mask)
    return F.cross_entropy(logits, batch.labels), logits, accounting, output.accounting


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


def read_vectors(qbar, snapshot, reader, backend, target, mode='normal'):
    """Independent explicit-slot read, used only for diagnostics/oracle evaluation."""
    b, t, _ = qbar.shape
    row = torch.arange(b, device=qbar.device)
    q = elu_plus_one(rmsnorm_no_affine(qbar, reader.eps_norm) @ reader.Rq)
    if mode == 'oracle_key':
        # Substitute the actual stored positive key, not a twice-transformed key.
        q = snapshot.keys[row, target][:, None].expand_as(q)
    scores = torch.einsum('btr,bnr->btn', q, snapshot.keys)
    valid = snapshot.valid[:, None].expand_as(scores)
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
    if mode not in CONTROLS:
        raise ValueError('unknown control')
    if model.config.steps != 1 or model.config.readout != 'increment' or model.config.layers != (14,):
        raise ValueError('offline diagnostics require the V2 single-layer single-read protocol')
    original, _ = build_support(model, batch, observe=True)
    selected_original = original.select(batch.owners)
    if donor_batch is not None:
        if mode != 'shuffled' or len(donor_batch.rows) != len(batch.rows):
            raise ValueError('explicit donors require shuffled mode and one donor per query')
        if any(a['episode_id'] == b['episode_id'] for a, b in zip(batch.rows, donor_batch.rows)):
            raise ValueError('shuffled donor must differ from recipient')
        donor, _ = build_support(model, donor_batch, observe=True)
        snapshot = donor.select(donor_batch.owners)
    else:
        snapshot = control_snapshot(original, mode).select(batch.owners)
    target = torch.tensor([r['query']['target_fact_index'] for r in batch.rows], device=batch.labels.device)
    row = torch.arange(len(target), device=target.device)
    if ((target < 0) | (target >= selected_original.valid.sum(-1))).any():
        raise ValueError('invalid target fact index')
    target_value = selected_original.values[row, target]
    side = model.adapters()[14].side
    details = {}
    with capture_projection(side) as seen:
        def observe(module, args, output):
            z, state, accounting = output
            raw, weights, scores, denominator = read_vectors(seen['q'], snapshot, side.reader, model.config.backend, target, mode)
            if mode in ('oracle_key', 'oracle_value'):
                # Match S=1 increment's arithmetic, including cancellation rounding.
                qbar = seen['q']
                z = (qbar + side.reader.eta * (raw * side.reader.read_gain @ side.reader.Ro)) - qbar
                z = z * batch.question_mask[..., None]
            details.update(read_metrics(raw, weights, scores, denominator, target_value, target,
                                       batch.question_mask, snapshot.valid,
                                       address_target_available=mode != 'shuffled'))
            if mode == 'shuffled':
                details['address.target_unavailable_reason'] = 'recipient target is undefined in donor memory'
            details['output.z_norm'] = float(z[batch.question_mask].norm(dim=-1).mean())
            for name in ('P', 'C_g', 'C_u'):
                if hasattr(side.fusion, name):
                    delta = z @ getattr(side.fusion, name)
                    details['output.'+name+'_norm'] = float(delta[batch.question_mask].norm(dim=-1).mean())
            details['read.used_norm'] = float((raw * side.reader.read_gain)[batch.question_mask].norm(dim=-1).mean())
            details['read.reconstruction_max_abs'] = float((z - ((seen['q'] + side.reader.eta*(raw * side.reader.read_gain @ side.reader.Ro)) - seen['q']) * batch.question_mask[..., None]).abs().max())
            details['_clean_vectors'] = target_value.detach().cpu()
            details['_read_vectors'] = raw[row, batch.question_mask.sum(-1)-1].detach().cpu()
            return z, state, accounting
        handle = side.register_forward_hook(observe)
        try:
            out = model(batch.question_ids, batch.question_mask, memory=snapshot.memory, phase='query')
        finally:
            handle.remove()
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
