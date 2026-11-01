"""E-0022 V3 training matrix on T2 chain-end (isolated-write M line). Execute only in an allocated GPU job.

- ``start`` (V3-S0): the D5 seed-0 adapter (ATT-0159-1, pinned adapter.pt digest) continues all
  side parameters on T2 train hop-1 questions for 6,000 steps with the D5 recipe (LR 3e-3, fresh
  AdamW, 300-step warmup then cosine, 8 episodes per step, final checkpoint only). The reviewed
  E-0016 evaluator scores T2 dev under the isolated write; the hop-1 H3 gate at S=1 decides.
- ``arm`` (V3-01..V3-10, plan §9.2): one arm from the common start, 6,000 steps on T2 train
  hops 1-3, the same order and samples for every arm. Then the main evaluation at S_eval =
  S_train, the S_eval scan, save and reload (records must be identical).
- ``intervene``: intermediate-state interventions for the S_M* arm.
- ``analyze``: the pre-registered rules and paired bootstrap intervals over the ten arms.

Arm options are new and default-off; nothing here changes an existing source file:
- steps, eta, update, readout and rank are AdapterConfig fields, so their checkpoints load with
  the stock BranchModel.load_adapter.
- V3-05 fixed_query uses LatentReader.forward's fixed_query option unchanged; the
  FixedQueryReader subclass only sets the flag, because SideBranch does not pass it.
- V3-06 (TokenwiseKernelSide), V3-08 (UntiedReader) and V3-07 (widen) are new code here.
  Every checkpoint carries variant.json, and load_arm rebuilds the variant before a strict load.
T2 hops 4/6/8 and confirmation are never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

from bypass_core import LatentReader, elu_plus_one, rmsnorm_no_affine
from bypass_model import SideBranch
from v1_data import derive_seed

RUNS = Path('/work/xg24i002/x10041/bypass-lora/runs')
CONTRACT = 'E-0023@1'  # E-0023 continues the E-0022 matrix (D-0031); E-0022 takes no more runs
EXPERIMENT = 'E-0023'
START_CONTRACTS = ('E-0022@2', 'E-0023@1')  # V3-S0 ran under E-0022@2; an E-0023 second start under E-0023@1
START_RUN = 'ATT-0159-1'  # D5 seed 0
START_ADAPTER_SHA256 = 'c4f3f9f8c24a55a89142cd251e845a39ac4c235fb22aa244afb8e7cc3425b794'
STEPS = 6000
LR = .003
BATCH = 8
MAX_HOP = 3
SEED = 20261029
BOOTSTRAP_RESAMPLES = 10000
MAC_SLOTS = 16  # E=16 written facts per episode

# plan §9.2. Absent keys keep the common start's value (steps 1, eta 0.25, accumulate, increment, r128).
ARMS = {
    'V3-01': {'steps': 1},
    'V3-02': {'steps': 2},
    'V3-03': {'steps': 4},
    'V3-04': {'steps': 4, 'eta': .0625},
    'V3-05': {'steps': 4, 'fixed_query': True},
    'V3-06': {'steps': 4, 'kernel': 'tokenwise'},
    'V3-07': {'steps': 1, 'rank': 152},
    'V3-08': {'steps': 4, 'untied': True},
    'V3-09': {'steps': 4, 'update': 'anchored'},
    'V3-10': {'steps': 4, 'readout': 'full'},
}
S_EVAL = (1, 2, 4, 8)
UNTIED_S_EVAL = (1, 2, 4)  # plan §9.4: only the defined prefixes of the four weight sets
WIDEN_SEED = ('V3', 'V3-07', 'widen', 0)


# ---- schedules -------------------------------------------------------------------------------

def v3_schedule(rows, stage, steps=STEPS):
    """E-0022 order: base = derive_seed(20261029, "concat_loop_v2", "V3", stage, 0); epoch e is a
    random.Random(base + e) shuffle of all row indices; epochs are concatenated until steps*8
    episodes, cut into consecutive batches of 8. stage 'start' (hop-1 rows) or 'train' (all)."""
    from v2_schedule import _cost
    if stage not in ('start', 'train'):
        raise ValueError('stage is start or train')
    if type(steps) is not int or steps < 1 or not rows:
        raise ValueError('positive integer steps and rows required')
    base = derive_seed(SEED, 'concat_loop_v2', 'V3', stage, 0)
    order, epochs = [], 0
    while len(order) < steps*BATCH:
        indices = list(range(len(rows)))
        random.Random(base + epochs).shuffle(indices)
        order.extend(indices)
        epochs += 1
    used = order[:steps*BATCH]
    batches = [used[i:i+BATCH] for i in range(0, len(used), BATCH)]
    return {'batches': batches, 'steps': steps, 'tokens': sum(_cost(rows[i]) for i in used),
            'warmup_steps': math.ceil(.05*steps), 'step_budget': steps, 'stage': stage, 'rows': len(rows),
            'epochs': epochs, 'order_seed': base,
            'order_seed_rule': f'derive_seed(20261029, "concat_loop_v2", "V3", "{stage}", 0); '
                               'epoch e shuffled by random.Random(base + e)',
            'order_sha256': hashlib.sha256(''.join(rows[i]['episode_id']+'\n' for i in used).encode('utf-8')).hexdigest(),
            'order_hash_encoding': 'UTF-8 episode_id followed by LF for every scheduled example'}


def trim_smoke(schedule, rows, steps):
    from v2_schedule import _cost
    plan = dict(schedule, batches=schedule['batches'][:steps], steps=steps, smoke_steps=steps,
                warmup_steps=math.ceil(.05*steps))
    plan['tokens'] = sum(_cost(rows[i]) for batch in plan['batches'] for i in batch)
    return plan


# ---- arm variants ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    fixed_query: bool = False
    kernel: str = 'memory'  # 'memory' or 'tokenwise' (V3-06)
    untied_steps: int = 0   # V3-08: number of Rq/Ro sets (0 = shared)


def arm_variant(arm):
    spec = ARMS[arm]
    return Variant(fixed_query=spec.get('fixed_query', False), kernel=spec.get('kernel', 'memory'),
                   untied_steps=spec['steps'] if spec.get('untied') else 0)


def arm_config(start_cfg, arm):
    spec = ARMS[arm]
    return replace(start_cfg, steps=spec['steps'], eta=spec.get('eta', start_cfg.eta),
                   update=spec.get('update', 'accumulate'), readout=spec.get('readout', 'increment'),
                   rank=spec.get('rank', start_cfg.rank))


class FixedQueryReader(LatentReader):
    """LatentReader with fixed_query=True: every step uses the step-1 query (MSG-0549),
    so with accumulate/increment Z = S * eta * g * a^(0) Ro."""

    def forward(self, qbar, state, steps=1, update='accumulate', readout='increment', fixed_query=False):
        return super().forward(qbar, state, steps, update, readout, True)


class UntiedReader(LatentReader):
    """V3-08: step 1 uses Rq/Ro, step s > 1 uses untied_Rq[s-2]/untied_Ro[s-2]; otherwise
    LatentReader.forward's loop expression by expression."""

    def step_matrices(self, step):
        return (self.Rq, self.Ro) if step == 0 else (self.untied_Rq[step-1], self.untied_Ro[step-1])

    def forward(self, qbar, state, steps=1, update='accumulate', readout='increment', fixed_query=False):
        if qbar.ndim != 2 or qbar.shape[-1] != self.rank:
            raise ValueError('qbar must have shape [batch, rank]')
        if fixed_query or not 1 <= steps <= 1 + len(self.untied_Rq):
            raise ValueError('untied reader: no fixed query and at most one step per weight set')
        if update not in {'accumulate', 'anchored'} or readout not in {'increment', 'full'}:
            raise ValueError('unknown update or readout')
        self.last_read_calls = 0
        z0 = qbar
        z = z0
        for step in range(steps):
            Rq, Ro = self.step_matrices(step)
            query = elu_plus_one(rmsnorm_no_affine(z, self.eps_norm) @ Rq)
            a = state.read(query, self.eps_read, self.temperature()) * self.read_gain
            self.last_read_calls += 1
            candidate = z0 if update == 'anchored' else z
            z = candidate + self.eta * (a @ Ro)
        return z - z0 if readout == 'increment' else z


def tokenwise_loop(q, k, v, reader, steps):
    """plan §9.2 same-source tokenwise kernel: a^(s) = SiLU(N(z^(s)) Rq + N(kbar_q)) * vbar_q,
    z^(s+1) = z^(s) + eta a^(s) Ro, Z = z^(S) - z^(0); kbar_q, vbar_q from the question token."""
    nk = rmsnorm_no_affine(k, reader.eps_norm)
    z0 = q
    z = z0
    for _ in range(steps):
        a = F.silu(rmsnorm_no_affine(z, reader.eps_norm) @ reader.Rq + nk) * v
        z = z + reader.eta * (a @ reader.Ro)
    return z - z0


class TokenwiseKernelSide(SideBranch):
    """V3-06: the question phase replaces the memory loop by tokenwise_loop (no memory read).
    The support phase is SideBranch's own, so the write protocol and accounting are unchanged."""

    def forward(self, u, features, state, valid, phase, write_mask=None):
        if phase != 'query':
            if phase != 'support':
                raise ValueError('the tokenwise kernel arm is defined on the M-line protocol only')
            return super().forward(u, features, state, valid, phase, write_mask)
        cfg = self.config
        q, k, v = self.projection(features['q'], features['k'], features['v'])
        z = tokenwise_loop(q, k, v, self.reader, cfg.steps) * valid.unsqueeze(-1)
        count = int(valid.sum())
        return z, state, {'valid_tokens': count, 'writes': 0, 'reads': 0, 'reader_steps': cfg.steps,
                          'tokenwise_steps': cfg.steps*count}


def install_variant(side, variant):
    """Convert one fresh layer-14 SideBranch in place (default-off: Variant() changes nothing)."""
    cfg = side.config
    if variant.kernel not in ('memory', 'tokenwise') or variant.untied_steps < 0:
        raise ValueError('unknown variant')
    if variant.fixed_query + (variant.kernel == 'tokenwise') + bool(variant.untied_steps) > 1:
        raise ValueError('one variant per arm')
    if variant.fixed_query:
        side.reader.__class__ = FixedQueryReader
    if variant.untied_steps:
        if variant.untied_steps != cfg.steps or cfg.steps < 2:
            raise ValueError('untied reader needs one weight set per training step')
        reader = side.reader
        reader.__class__ = UntiedReader
        reader.untied_Rq = nn.ParameterList([nn.Parameter(reader.Rq.detach().clone())
                                             for _ in range(cfg.steps-1)])
        reader.untied_Ro = nn.ParameterList([nn.Parameter(reader.Ro.detach().clone())
                                             for _ in range(cfg.steps-1)])
    if variant.kernel == 'tokenwise':
        if cfg.update != 'accumulate' or cfg.readout != 'increment' or cfg.read_gain != 1.:
            raise ValueError('the tokenwise kernel is defined for accumulate/increment with g = 1')
        side.__class__ = TokenwiseKernelSide
    return side


def side_variant(side):
    return Variant(fixed_query=isinstance(side.reader, FixedQueryReader),
                   kernel='tokenwise' if isinstance(side, TokenwiseKernelSide) else 'memory',
                   untied_steps=1 + len(side.reader.untied_Rq) if isinstance(side.reader, UntiedReader) else 0)


def widen(weights, rank, dims, seed_parts=WIDEN_SEED):
    """V3-07 (E-0022@2): r128 common-start weights to rank r'. The first 128 dims are the start;
    new A_q, A_k, A_v columns are randn/sqrt(d) from a derive_seed generator (drawn in that
    order on CPU); Rq' = blockdiag(Rq, I), Ro' = blockdiag(Ro, I); new P rows are zero."""
    old = weights['reader.Rq'].shape[0]
    extra = rank - old
    if extra <= 0:
        raise ValueError('widening needs a larger rank')
    generator = torch.Generator().manual_seed(derive_seed(SEED, 'concat_loop_v2', *seed_parts) % 2**63)
    result = dict(weights)
    for name, d in zip(('projection.A_q', 'projection.A_k', 'projection.A_v'), dims):
        if weights[name].shape != (d, old):
            raise ValueError(f'{name} has an unexpected shape')
        new = torch.randn(d, extra, generator=generator) / d ** .5
        result[name] = torch.cat((weights[name], new.to(weights[name].dtype)), dim=1)
    for name in ('reader.Rq', 'reader.Ro'):
        result[name] = torch.block_diag(weights[name], torch.eye(extra, dtype=weights[name].dtype))
    P = weights['fusion.P']
    result['fusion.P'] = torch.cat((P, torch.zeros(extra, P.shape[1], dtype=P.dtype)), dim=0)
    return result


def macs_per_question_token(arm, d=1024, n=MAC_SLOTS):
    """Executed multiply-accumulates per question token (E-0022@2): projections 3dr, exit P dr, and per
    step Rq r^2 (once in total for a fixed query, which LatentReader computes outside the loop), the
    slot scores and weighted values 2nr (none for the tokenwise kernel) and Ro r^2. Memory loop:
    4096r + S(2r^2 + 2nr); fixed query: 4096r + r^2 + S(r^2 + 2nr); tokenwise: 4096r + 2Sr^2."""
    spec = ARMS[arm]
    r, steps = spec.get('rank', 128), spec['steps']
    fixed, tokenwise = spec.get('fixed_query', False), spec.get('kernel') == 'tokenwise'
    query_projection = r*r if fixed else steps*r*r
    slots = 0 if tokenwise else steps*2*n*r
    return {'question_token': 4*d*r + query_projection + slots + steps*r*r,
            'support_projection_per_token': 3*d*r,
            'formula': ('4096r + r^2 + S(r^2 + 2nr)' if fixed else '4096r + 2Sr^2' if tokenwise
                        else '4096r + S(2r^2 + 2nr)') + ', n=16, executed'}


# ---- model construction ------------------------------------------------------------------------

def _start_config(directory):
    from bypass_model import AdapterConfig
    from v2_memory import config
    data = json.loads((Path(directory)/'config.json').read_text())
    data['layers'] = tuple(data['layers'])
    cfg = AdapterConfig(**data)
    if cfg != config('direct_hidden', 'soft_slot', 'F_out'):
        raise ValueError('the common start must be the fixed D5 configuration')
    return cfg


def _side_weights(directory):
    weights = torch.load(Path(directory)/'adapter.pt', map_location='cpu', weights_only=True)
    if set(weights) != {'14'}:
        raise ValueError('single layer-14 adapter expected')
    return weights['14']


@contextmanager
def arm_model(base, arm, start_directory):
    """The arm's model initialized from the common start (V3-08: Rq/Ro copied to every step;
    V3-07: widened); other arms load the start weights unchanged."""
    from v2_train import adapted
    start_cfg = _start_config(start_directory)
    weights = _side_weights(start_directory)
    cfg, variant = arm_config(start_cfg, arm), arm_variant(arm)
    with adapted(base, cfg=cfg) as model:
        side = model.adapters()[14].side
        install_variant(side, variant)
        if cfg.rank != start_cfg.rank:
            d = side.projection.A_q.shape[0]
            weights = widen(weights, cfg.rank, (d, side.projection.A_k.shape[0], side.projection.A_v.shape[0]))
        if variant.untied_steps:
            weights = dict(weights)
            for s in range(variant.untied_steps-1):
                weights[f'reader.untied_Rq.{s}'] = weights['reader.Rq'].clone()
                weights[f'reader.untied_Ro.{s}'] = weights['reader.Ro'].clone()
        side.load_state_dict(weights, strict=True)
        side.reader.eta = cfg.eta
        yield model


def export_arm(model, directory):
    side = model.adapters()[14].side
    model.export_adapter(directory)
    (Path(directory)/'variant.json').write_text(json.dumps(asdict(side_variant(side)), indent=2)+'\n')


@contextmanager
def load_arm(base, directory):
    """Rebuild an exported arm: stock config, the recorded variant, strict weight load."""
    from bypass_model import AdapterConfig
    from v2_train import adapted
    directory = Path(directory)
    data = json.loads((directory/'config.json').read_text())
    data['layers'] = tuple(data['layers'])
    cfg = AdapterConfig(**data)
    variant = Variant(**json.loads((directory/'variant.json').read_text()))
    with adapted(base, cfg=cfg) as model:
        side = model.adapters()[14].side
        install_variant(side, variant)
        side.load_state_dict(_side_weights(directory), strict=True)
        if side.reader.eta != cfg.eta:
            raise AssertionError('reader eta differs from the configuration')
        yield model


@contextmanager
def eval_steps(model, steps):
    """S_eval reads per question position for any arm (configuration restored on exit)."""
    if type(steps) is not int or steps < 1:
        raise ValueError('S_eval must be a positive integer')
    side = model.adapters()[14].side
    variant = side_variant(side)
    if variant.untied_steps and steps > variant.untied_steps:
        raise ValueError('the untied arm has no weights beyond its trained steps')
    cfg, original = model.config, side.config
    if original != cfg:
        raise ValueError('side and model configurations differ')
    try:
        model.config = replace(cfg, steps=steps)
        side.config = model.config
        yield model
    finally:
        model.config, side.config = cfg, original


# ---- replay, evaluation and interventions ---------------------------------------------------------

def slot_read(query, snapshot):
    """v2_memory.read_vectors' soft_slot expressions on a given positive query [B, T, r]."""
    scores = torch.einsum('btr,bnr->btn', query, snapshot.keys) / query.shape[-1] ** .5
    valid = snapshot.valid[:, None].expand_as(scores)
    logits = scores.masked_fill(~valid, -torch.inf)
    empty = ~valid.any(-1, keepdim=True)
    logits = torch.where(empty, torch.zeros_like(logits), logits)
    weights = logits.softmax(-1).masked_fill(~valid, 0)
    return weights @ snapshot.values, weights


def replay(side, qbar, kbar, vbar, snapshot, steps, replace_state=None, stop=None):
    """Explicit replay of the side's loop on a compact snapshot [B, T, r] (diagnostics,
    interventions). replace_state=(s, fn) substitutes z^(s) before step s+1 (1 <= s < steps).
    stop=s returns z^(s) only. Returns (output Z, per-step trajectory)."""
    reader, cfg = side.reader, side.config
    variant = side_variant(side)
    z0 = qbar
    z = z0
    if variant.fixed_query:
        fixed = elu_plus_one(rmsnorm_no_affine(z0, reader.eps_norm) @ reader.Rq)
    trajectory = []
    for s in range(steps):
        if stop == s:
            return z, trajectory
        if replace_state is not None and replace_state[0] == s:
            z = replace_state[1](z)
        Rq, Ro = reader.step_matrices(s) if variant.untied_steps else (reader.Rq, reader.Ro)
        if variant.kernel == 'tokenwise':
            pre = rmsnorm_no_affine(z, reader.eps_norm) @ Rq
            read = F.silu(pre + rmsnorm_no_affine(kbar, reader.eps_norm)) * vbar
            query, weights = pre, None
        else:
            query = fixed if variant.fixed_query else elu_plus_one(rmsnorm_no_affine(z, reader.eps_norm) @ Rq)
            raw, weights = slot_read(query, snapshot)
            read = raw * reader.read_gain
        trajectory.append({'state': z, 'query': query, 'weights': weights, 'read': read})
        candidate = z0 if cfg.update == 'anchored' else z
        z = candidate + reader.eta * (read @ Ro)
    if stop == steps:
        return z, trajectory
    return (z - z0 if cfg.readout == 'increment' else z), trajectory


def query_cosines(queries, mask):
    """Per step s (0-based): cosine of query(s) with query(0) (query_cosine_*) and with query(s-1)
    (query_cosine_prev_*, the adjacent-step cosine of E-0022@2; None at the first step), each as the
    mean over valid question positions (_all) and at the last valid position (_last)."""
    row = torch.arange(len(mask), device=mask.device)
    last = mask.long().sum(-1)-1
    count = mask.sum(-1)
    result = []
    for s, query in enumerate(queries):
        entry = {}
        for name, other in (('query_cosine', queries[0]), ('query_cosine_prev', queries[s-1] if s else None)):
            if other is None:
                entry[f'{name}_all'] = entry[f'{name}_last'] = None
                continue
            cosine = F.cosine_similarity(query, other, dim=-1, eps=1e-6)
            entry[f'{name}_all'] = (cosine*mask).sum(-1)/count
            entry[f'{name}_last'] = cosine[row, last]
        result.append(entry)
    return result


def _capture(model, batch, snapshot):
    """Native question forward on a snapshot; returns logits, projections, native Z, state, counts."""
    from v2_memory import answer_logits, capture_projection
    side = model.adapters()[14].side
    captured = {}

    def observe(module, args, output):
        captured['z'], captured['state'], captured['accounting'] = output

    with capture_projection(side) as seen:
        handle = side.register_forward_hook(observe)
        try:
            output = model(batch.question_ids, batch.question_mask, memory=snapshot.memory, phase='query')
        finally:
            handle.remove()
    logits = answer_logits(output, batch.question_mask)
    if not torch.isfinite(logits).all():
        raise RuntimeError('nonfinite full-vocabulary evaluation logits')
    return logits, dict(seen), captured


def _with_increment(model, batch, snapshot, z):
    """Question forward whose side output is replaced by z (masked to valid positions)."""
    from v2_memory import answer_logits
    side = model.adapters()[14].side
    mask = batch.question_mask

    def substitute(module, args, output):
        return z * mask[..., None], output[1], output[2]

    handle = side.register_forward_hook(substitute)
    try:
        output = model(batch.question_ids, mask, memory=snapshot.memory, phase='query')
    finally:
        handle.remove()
    logits = answer_logits(output, mask)
    if not torch.isfinite(logits).all():
        raise RuntimeError('nonfinite full-vocabulary evaluation logits')
    return logits


@torch.no_grad()
def evaluate_arm_batch(model, batch, mode, donor_batch, steps):
    """Native S-step forward on the fixed control memory with per-step replay diagnostics
    (v3_loop.evaluate_loop_batch generalized to the arm variants)."""
    from v3_loop import RECONSTRUCTION_TOLERANCE, _snapshots, hop_of, path_matrix
    if model.training:
        raise ValueError('evaluation needs eval mode')
    side = model.adapters()[14].side
    variant = side_variant(side)
    snapshot = _snapshots(model, batch, mode, donor_batch)
    mask = batch.question_mask
    with eval_steps(model, steps):
        logits, seen, captured = _capture(model, batch, snapshot)
        state, memory = captured['state'], snapshot.memory[14]
        if not all(torch.equal(getattr(state, f), getattr(memory, f))
                   for f in ('M', 'c', 'counts', 'slot_keys', 'slot_values', 'slot_valid')):
            raise AssertionError('the question phase changed the memory')
        counts = captured['accounting']
        reads = 0 if variant.kernel == 'tokenwise' else steps*int(mask.sum())
        if counts['writes'] != 0 or counts['reads'] != reads:
            raise AssertionError('the question phase must read S times per valid token (memory arms) and never write')
        qbar, kbar, vbar = seen['q'], seen['k'], seen['v']
        output, trajectory = replay(side, qbar, kbar, vbar, snapshot, steps)
    native = captured['z']
    if not (torch.isfinite(native).all() and torch.isfinite(output).all()):
        raise RuntimeError('nonfinite loop state')
    difference = output - native
    relative = difference.norm(dim=-1)/native.norm(dim=-1).clamp_min(1e-6)
    reconstruction = {'max_relative': float(relative[mask].max()), 'max_abs': float(difference.abs().amax(-1)[mask].max())}
    if not reconstruction['max_relative'] <= RECONSTRUCTION_TOLERANCE:
        raise AssertionError(f'replay differs from the native output: {reconstruction}')
    losses = F.cross_entropy(logits, batch.labels, reduction='none')
    row = torch.arange(len(batch.labels), device=mask.device)
    last = mask.long().sum(-1)-1
    count = mask.sum(-1)
    has_slots = snapshot.valid.any(-1)
    paths = path_matrix(batch.rows, mask.device)
    cosines = query_cosines([step['query'] for step in trajectory], mask)
    target_available = mode != 'shuffled' and variant.kernel != 'tokenwise'
    per_step = []
    for step, cosine in zip(trajectory, cosines):
        state_norm, read_norm = step['state'].norm(dim=-1), step['read'].norm(dim=-1)
        values = {**cosine,
                  'state_norm_all': (state_norm*mask).sum(-1)/count, 'state_norm_last': state_norm[row, last],
                  'read_norm_all': (read_norm*mask).sum(-1)/count, 'read_norm_last': read_norm[row, last]}
        weights = step['weights']
        if weights is not None:
            neff = torch.where(weights.sum(-1) > 0, 1/weights.square().sum(-1).clamp_min(1e-30), 0)
            path_all, path_last = [], []
            for j in range(MAX_HOP):
                fact = paths[:, j].clamp_min(0)
                weight = weights.gather(-1, fact[:, None, None].expand(-1, weights.shape[1], 1)).squeeze(-1)
                path_all.append((weight*mask).sum(-1)/count)
                path_last.append(weight[row, last])
            values.update({'path_weight_all': torch.stack(path_all, -1), 'path_weight_last': torch.stack(path_last, -1),
                           'n_eff_all': (neff*mask).sum(-1)/count, 'n_eff_last': neff[row, last],
                           'argmax_last': torch.where(has_slots, weights[row, last].argmax(-1), -1)})
        per_step.append({key: [None]*len(batch.rows) if value is None else value.tolist()
                         for key, value in values.items()})
    out_norm = native.norm(dim=-1)
    predictions, labels, losses = logits.argmax(-1).tolist(), batch.labels.tolist(), losses.tolist()
    records = []
    for i, episode in enumerate(batch.rows):
        hop = hop_of(episode)
        record = {'episode_id': episode['episode_id'], 'hop': hop,
                  'support_order': episode['axes']['support_order'], 'target': labels[i],
                  'prediction': predictions[i], 'correct': predictions[i] == labels[i], 'nll': losses[i],
                  'question_tokens': int(count[i]), 'steps': steps,
                  'output_norm_last': out_norm[row, last][i].item(),
                  'output_norm_all': ((out_norm*mask).sum(-1)/count)[i].item()}
        for name in ('query_cosine_all', 'query_cosine_last', 'query_cosine_prev_all', 'query_cosine_prev_last',
                     'state_norm_all', 'state_norm_last',
                     'read_norm_all', 'read_norm_last', 'n_eff_all', 'n_eff_last', 'argmax_last'):
            record[name] = [s[name][i] for s in per_step] if name in per_step[0] else None
        for name in ('path_weight_all', 'path_weight_last'):
            record[name] = [s[name][i][:hop] for s in per_step] if target_available else None
        records.append(record)
    return records, reconstruction


@torch.no_grad()
def evaluate_arm(model, rows, pad_id, steps, controls=('normal', 'zero', 'shuffled', 'permuted')):
    """Batches of 16 and the reviewed global donor rule (recipient i gets support (i-1) mod N)."""
    from v2_memory import collate
    from v3_loop import EVAL_BATCH
    model.eval()
    if len(rows) < 2:
        raise ValueError('at least two rows are required')
    device = next(model.parameters()).device
    result, reconstruction = {}, {}
    for mode in controls:
        records, worst = [], {'max_relative': 0., 'max_abs': 0.}
        for start in range(0, len(rows), EVAL_BATCH):
            selected = rows[start:start+EVAL_BATCH]
            batch = collate(selected, pad_id, device)
            donors = None
            if mode == 'shuffled':
                donors = collate([rows[(i-1) % len(rows)] for i in range(start, start+len(selected))], pad_id, device)
            batch_records, error = evaluate_arm_batch(model, batch, mode, donors, steps)
            records.extend(batch_records)
            worst = {k: max(worst[k], error[k]) for k in worst}
        result[mode], reconstruction[mode] = records, worst
    return result, reconstruction


def _mean(values):
    values = list(values)
    return sum(values)/len(values) if values else None


def summarize_arm(records, prior_token=None):
    """Per hop (and all, and hop x support order): EM, NLL, prior EM; per step: the diagnostics."""
    groups = {'all': records}
    for hop in range(1, MAX_HOP+1):
        groups[f'hop{hop}'] = [r for r in records if r['hop'] == hop]
        for order in ('natural', 'shuffled'):
            groups[f'hop{hop}.{order}'] = [r for r in records if r['hop'] == hop and r['support_order'] == order]
    summaries = {}
    for name, group in groups.items():
        if not group:
            continue
        summary = {'count': len(group), 'correct_count': sum(r['correct'] for r in group),
                   'em': sum(r['correct'] for r in group)/len(group), 'nll': _mean(r['nll'] for r in group)}
        if prior_token is not None:
            summary['prior_em'] = sum(r['target'] == prior_token for r in group)/len(group)
        if '.' not in name:
            tokens = sum(r['question_tokens'] for r in group)
            width = max(r['hop'] for r in group)
            per_step = []
            for s in range(group[0]['steps']):
                step = {f'{field}_all': sum(r[f'{field}_all'][s]*r['question_tokens'] for r in group)/tokens
                        for field in ('query_cosine', 'state_norm', 'read_norm')}
                step.update({f'{field}_last': _mean(r[f'{field}_last'][s] for r in group)
                             for field in ('query_cosine', 'state_norm', 'read_norm')})
                step['query_cosine_prev_all'] = None if s == 0 else \
                    sum(r['query_cosine_prev_all'][s]*r['question_tokens'] for r in group)/tokens
                step['query_cosine_prev_last'] = None if s == 0 else _mean(r['query_cosine_prev_last'][s] for r in group)
                if group[0]['n_eff_all'] is not None:
                    step['n_eff_all'] = sum(r['n_eff_all'][s]*r['question_tokens'] for r in group)/tokens
                    step['n_eff_last'] = _mean(r['n_eff_last'][s] for r in group)
                if group[0]['path_weight_all'] is not None:
                    for scope in ('all', 'last'):
                        values = []
                        for j in range(width):
                            members = [r for r in group if r['hop'] > j]
                            if scope == 'all':
                                total = sum(r['question_tokens'] for r in members)
                                values.append(sum(r['path_weight_all'][s][j]*r['question_tokens'] for r in members)/total)
                            else:
                                values.append(_mean(r['path_weight_last'][s][j] for r in members))
                        step[f'path_weight_{scope}'] = values
                    if all('_path' in r for r in group):
                        step['argmax_is_path_fact'] = [_mean(r['argmax_last'][s] == r['_path'][j]
                                                             for r in group if r['hop'] > j) for j in range(width)]
                per_step.append(step)
            summary['per_step'] = per_step
            summary['output_norm_last'] = _mean(r['output_norm_last'] for r in group)
        summaries[name] = summary
    return summaries


def matched_donors(rows):
    """Recipient i: the nearest previous row (cyclic) with the same hop and the same question
    token count, or None."""
    donors = []
    for i, row in enumerate(rows):
        donor = None
        for back in range(1, len(rows)):
            j = (i - back) % len(rows)
            other = rows[j]
            if other['axes']['hop'] == row['axes']['hop'] and len(other['question_input_ids']) == len(row['question_input_ids']):
                donor = j
                break
        donors.append(donor)
    return donors


def _align(donor_state, donor_mask, recipient_mask):
    """Right-aligned positions: recipient t takes donor t + (len_d - len_r), clamped to [0, len_d-1]."""
    lr, ld = recipient_mask.long().sum(-1), donor_mask.long().sum(-1)
    t = torch.arange(recipient_mask.shape[1], device=recipient_mask.device)
    index = (t[None] + (ld - lr)[:, None]).clamp(min=0)
    index = torch.minimum(index, (ld - 1)[:, None])
    return donor_state.gather(1, index[..., None].expand(-1, -1, donor_state.shape[-1]))


@torch.no_grad()
def intervene(model, rows, pad_id, steps):
    """plan §9.4 intermediate-state interventions on the normal memory at every valid question
    position: z^(s) (1 <= s < S) replaced by a norm-matched Gaussian (random), by the state at
    step s of the global donor episode (i-1) on its own memory (unrelated: another episode's
    entities), or by that of the nearest previous same-hop, same-question-length episode
    (matched donor). The remaining steps read the recipient's memory. 'none' is the replay
    without replacement on the same path."""
    from v2_memory import collate
    from v3_loop import EVAL_BATCH, _snapshots
    model.eval()
    side = model.adapters()[14].side
    device = next(model.parameters()).device
    matched = matched_donors(rows)
    records = {s: {k: [] for k in ('none', 'random', 'unrelated', 'matched')} for s in range(1, steps)}
    native_mismatch = 0
    with eval_steps(model, steps):
        for s in range(1, steps):
            generator = torch.Generator(device=device).manual_seed(
                derive_seed(SEED, 'concat_loop_v2', 'V3', 'intervention', s) % 2**63)
            for start in range(0, len(rows), EVAL_BATCH):
                indices = list(range(start, min(start+EVAL_BATCH, len(rows))))
                batch = collate([rows[i] for i in indices], pad_id, device)
                snapshot = _snapshots(model, batch, 'normal', None)
                logits, seen, _ = _capture(model, batch, snapshot)
                q, k, v = seen['q'], seen['k'], seen['v']
                mask = batch.question_mask

                def donor_state(donor_indices):
                    # Several recipients may share a donor: encode each donor once, then gather.
                    unique = list(dict.fromkeys(donor_indices))
                    donor = collate([rows[j] for j in unique], pad_id, device)
                    donor_snapshot = _snapshots(model, donor, 'normal', None)
                    _, donor_seen, _ = _capture(model, donor, donor_snapshot)
                    state, _ = replay(side, donor_seen['q'], donor_seen['k'], donor_seen['v'], donor_snapshot, steps, stop=s)
                    take = torch.tensor([unique.index(j) for j in donor_indices], device=device)
                    return _align(state.index_select(0, take), donor.question_mask.index_select(0, take), mask)

                unrelated = donor_state([(i-1) % len(rows) for i in indices])
                available = [matched[i] is not None for i in indices]
                matched_state = donor_state([matched[i] if matched[i] is not None else (i-1) % len(rows)
                                             for i in indices])

                def gaussian(z):
                    noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
                    return noise * (z.norm(dim=-1, keepdim=True)/noise.norm(dim=-1, keepdim=True).clamp_min(1e-30))

                options = {'none': None, 'random': (s, gaussian), 'unrelated': (s, lambda z: unrelated),
                           'matched': (s, lambda z: matched_state)}
                for name, option in options.items():
                    z, _ = replay(side, q, k, v, snapshot, steps, replace_state=option)
                    out = _with_increment(model, batch, snapshot, z)
                    losses = F.cross_entropy(out, batch.labels, reduction='none').tolist()
                    predictions = out.argmax(-1).tolist()
                    if name == 'none':
                        native_mismatch += sum(a != b for a, b in zip(predictions, logits.argmax(-1).tolist()))
                    for b, i in enumerate(indices):
                        if name == 'matched' and not available[b]:
                            continue
                        records[s][name].append({'episode_id': rows[i]['episode_id'], 'hop': rows[i]['axes']['hop'],
                                                 'correct': predictions[b] == int(batch.labels[b]), 'nll': losses[b],
                                                 'prediction': predictions[b],
                                                 'donor': None if name in ('none', 'random') else
                                                 rows[(i-1) % len(rows) if name == 'unrelated' else matched[i]]['episode_id']})
    summary = {}
    for s, by_name in records.items():
        matched_ids = {r['episode_id'] for r in by_name['matched']}
        summary[str(s)] = {}
        for name, group in by_name.items():
            entry = {}
            for hop in range(1, MAX_HOP+1):
                members = [r for r in group if r['hop'] == hop]
                if members:
                    entry[f'hop{hop}'] = {'count': len(members), 'em': _mean(r['correct'] for r in members),
                                          'nll': _mean(r['nll'] for r in members)}
            if name == 'none':
                for hop in range(1, MAX_HOP+1):
                    members = [r for r in group if r['hop'] == hop and r['episode_id'] in matched_ids]
                    if members:
                        entry[f'hop{hop}.matched_subset'] = {'count': len(members), 'em': _mean(r['correct'] for r in members),
                                                             'nll': _mean(r['nll'] for r in members)}
            summary[str(s)][name] = entry
    return summary, records, {'none_vs_native_prediction_mismatches': native_mismatch,
                              'rows_without_matched_donor': sum(m is None for m in matched)}


# ---- analysis (pre-registered rules, bootstrap) -------------------------------------------------

def primary_fraction(hops):
    """(EM_hop2 + EM_hop3)/2 from exact correct counts."""
    return (Fraction(hops['hop2']['correct_count'], hops['hop2']['count'])
            + Fraction(hops['hop3']['correct_count'], hops['hop3']['count'])) / 2


def hop_fraction(hops, hop):
    return Fraction(hops[f'hop{hop}']['correct_count'], hops[f'hop{hop}']['count'])


def decide(main):
    """E-0022@2 rules 2-8 on the main (S_eval = S_train, normal) per-hop counts of each arm.
    Differences are exact fractions in pp; thresholds include equality."""
    pp = {arm: primary_fraction(h)*100 for arm, h in main.items()}
    hop1 = {arm: hop_fraction(h, 1)*100 for arm, h in main.items()}
    star = 'V3-03' if pp['V3-03'] - pp['V3-02'] >= Fraction(1, 2) else 'V3-02'
    star_steps = ARMS[star]['steps']
    c_binding = pp['V3-07'] >= pp['V3-01']
    checks = {
        'a': {'difference_pp': pp[star]-pp['V3-01'], 'threshold': 3},
        'b': {'difference_pp': hop1[star]-hop1['V3-01'], 'threshold': -1},
        'c': {'difference_pp': pp[star]-pp['V3-07'], 'threshold': 3, 'binding': c_binding,
              'note': None if c_binding else 'V3-07 primary below V3-01: (c) does not bind, (a) applies (E-0022@2 rule 4)'},
        'd': {'difference_pp': pp[star]-pp['V3-06'], 'threshold': 3},
        'e': ({'difference_pp': pp['V3-03']-pp['V3-05'], 'threshold': 3} if star_steps == 4 else
              {'difference_pp': None, 'threshold': 3, 'deferred_to_V4': True}),
    }
    for name, check in checks.items():
        if check['difference_pp'] is None:
            check['passed'] = None
        else:
            check['passed'] = check['difference_pp'] >= check['threshold'] or check.get('binding') is False
    if star_steps == 4 and checks['e']['passed'] is False:
        checks['e']['note'] = 'no evidence of an iterated query yet'
    gl_m = all(c['passed'] is not False for c in checks.values())
    if gl_m:
        outcome = 'GL_M holds: draft the V4 (E3) contract after the conclusion review (rule 6); evidence supports'
    elif pp[star] - pp['V3-01'] <= 0:
        outcome = 'S_M* - V3-01 <= 0: screening evidence challenges H-0005@2 (rule 7)'
    else:
        outcome = 'descriptive only; pause H-0005 and write reopening conditions (rule 8)'
    as_float = lambda value: None if value is None else float(value)
    return {'primary_pp': {a: float(v) for a, v in pp.items()}, 'hop1_pp': {a: float(v) for a, v in hop1.items()},
            'primary_exact': {a: f'{v.numerator}/{v.denominator}' for a, v in pp.items()},
            'S_M*': star, 'S_M*_steps': star_steps,
            'S_M*_rule': 'V3-03 if V3-03 - V3-02 >= 0.5 pp else V3-02 (rule 3: S2 when they differ by < 0.5)',
            'checks': {n: dict(c, difference_pp=as_float(c['difference_pp'])) for n, c in checks.items()},
            'GL_M': gl_m, 'outcome': outcome,
            'descriptive': {arm: float(pp[arm]-pp[star]) for arm in ('V3-04', 'V3-08', 'V3-09', 'V3-10')}}


def bootstrap(correct, pairs, resamples=BOOTSTRAP_RESAMPLES, seed=SEED):
    """Hop-stratified paired bootstrap: correct[arm][hop] is a 0/1 list in a common episode order.
    One set of resampled indices per hop (torch CPU generator, seed 20261029, hops 1, 2, 3 in
    order) is shared by every pair. Returns 95% percentile intervals in pp for each hop and for
    the primary (hop2 + hop3)/2."""
    generator = torch.Generator().manual_seed(seed)
    arms = list(correct)
    index = {}
    for hop in range(1, MAX_HOP+1):
        n = len(correct[arms[0]][hop])
        if any(len(correct[a][hop]) != n for a in arms):
            raise ValueError('arms must cover the same episodes')
        index[hop] = torch.randint(n, (resamples, n), generator=generator)
    em = {a: {hop: torch.tensor(correct[a][hop], dtype=torch.float64)[index[hop]].mean(-1)
              for hop in index} for a in arms}
    result = {}
    for left, right in pairs:
        diff = {hop: (em[left][hop] - em[right][hop])*100 for hop in index}
        diff['primary'] = (diff[2] + diff[3])/2
        result[f'{left} - {right}'] = {
            (f'hop{k}' if k != 'primary' else k): {'low': float(torch.quantile(v, .025)),
                                                     'high': float(torch.quantile(v, .975))}
            for k, v in diff.items()}
    return result


# ---- runners ------------------------------------------------------------------------------------

def _tokenizer():
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    return tokenizer, tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def _data(tokenizer, eval_limit=0):
    from v3_loop import T2_DATA, T2_METADATA_SHA256, load_t2
    metadata, train_rows, train_differing = load_t2(tokenizer, 'train')
    _, dev, dev_differing = load_t2(tokenizer, 'dev')
    if eval_limit:
        # Smoke: the first rows of each hop, so every hop is present.
        dev = [r for hop in (1, 2, 3) for r in [x for x in dev if x['axes']['hop'] == hop][:eval_limit]]
    prior = Counter(r['query']['answer_token_id'] for r in train_rows).most_common(1)[0][0]
    info = {'directory': str(T2_DATA), 'metadata_sha256': T2_METADATA_SHA256,
            'dev_sha256': metadata['splits']['dev']['sha256'], 'train_sha256': metadata['splits']['train']['sha256'],
            'dev_rows': len(dev), 'train_rows': len(train_rows), 'prior_token': prior,
            'context_fact_tokens_differ_from_isolated': {'train': train_differing, 'dev': dev_differing},
            'baselines': {'shortcuts_dev': metadata['shortcuts']['dev']}}
    return train_rows, dev, prior, info


def readiness_provenance(ready_run, start_files):
    """E-0023 stage 0: a completed formal readiness run (v3_ready) whose rule R passed for exactly this start."""
    from v2r16_train import file_record
    ready_run = Path(ready_run)
    manifest = json.loads((ready_run/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True
            or manifest.get('contract') != CONTRACT or manifest.get('command') != 'ready'
            or manifest.get('eligible_as_evidence') is not True or manifest.get('smoke_only') is not False
            or manifest.get('metrics', {}).get('readiness_passed') is not True):
        raise ValueError('the readiness run must be a completed formal E-0023 stage-0 run whose rule R passed')
    if (manifest.get('start') or {}).get('checkpoint_files') != start_files:
        raise ValueError('the readiness run checked a different common start')
    return {'run': str(ready_run.resolve()), 'source_manifest': file_record(ready_run/'manifest.json'),
            'criteria': manifest['metrics'].get('readiness_criteria')}


def start_provenance(run, expected_sha256, smoke=False, ready_run=None):
    """A completed V3-S0 run whose adapter.pt has the pinned digest. It is a formal start when it is
    formal and either passed the E-0010@2 hop1 H3 gate (E-0022) or has an E-0023 readiness run whose
    rule R passed for this start (``ready_run``). A smoke job (smoke=True) may start from a completed
    smoke V3-S0 run; its output is smoke too."""
    from v2r16_train import checkpoint_records, file_record
    if not (isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(c in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('an expected lowercase hex sha256 of the common start adapter.pt is required')
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True
            or manifest.get('arm') != 'V3-S0' or manifest.get('contract') not in START_CONTRACTS):
        raise ValueError('the common start must be a completed V3-S0 run')
    files = checkpoint_records(run/'adapter')
    if files != manifest.get('checkpoint_files') or set(files) != {'adapter.pt', 'config.json', 'variant.json'}:
        raise ValueError('common start files differ from the run manifest')
    if files['adapter.pt']['sha256'] != expected_sha256:
        raise ValueError('common start adapter.pt differs from the pinned digest')
    readiness = readiness_provenance(ready_run, files) if ready_run is not None else None
    gate = manifest.get('metrics', {}).get('hop1_gate', {}).get('passed') is True
    formal = manifest.get('eligible_as_evidence') is True and (gate or readiness is not None)
    if not (formal or (smoke and manifest.get('smoke_only') is True)):
        raise ValueError('the common start must be a formal V3-S0 run that passed the hop1 H3 gate '
                         'or E-0023 readiness rule R')
    return {'run': str(run.resolve()), 'checkpoint_files': files, 'expected_adapter_sha256': expected_sha256,
            'source_manifest': file_record(run/'manifest.json'), 'optimizer_state_loaded': False,
            'formal_start': formal, 'hop1_gate_passed': gate, 'readiness': readiness}


def _reviewed(model, dev, pad_id, destination, prior, controls):
    from v2_memory import isolated_support
    from v2r16_memory import evaluate
    return evaluate(model, dev, pad_id, destination, 'dev', controls, prior, support_fn=isolated_support)


def run_start(args, output, result):
    from calibrate import load_base
    from v2_memory import isolated_support
    from v2_train import adapted, train
    from v2r16_memory import train_forward
    from v2r16_train import STANDARD, checkpoint_records
    from v3_loop import adapter_provenance, hop1_gate, reviewed_by_hop
    provenance = adapter_provenance(RUNS/START_RUN/'run', START_ADAPTER_SHA256)
    result['adapter'] = provenance
    tokenizer, pad_id = _tokenizer()
    train_rows, dev, prior, result['data'] = _data(tokenizer, args.eval_limit)
    hop1_train = [r for r in train_rows if r['axes']['hop'] == 1]
    dev_hop1 = [r for r in dev if r['axes']['hop'] == 1]
    schedule = v3_schedule(hop1_train, 'start')
    if args.smoke_steps:
        schedule = trim_smoke(schedule, hop1_train, args.smoke_steps)
    result['schedule'] = {k: v for k, v in schedule.items() if k != 'batches'}
    base = load_base()
    forward = partial(train_forward, support_fn=isolated_support)
    checkpoint = RUNS/START_RUN/'run'/'adapter'

    def curve(model, rows, pad, destination, split, controls, prior_token):
        return _reviewed(model, rows, pad, destination, prior_token, controls)

    with adapted(base, checkpoint=checkpoint) as model:
        if checkpoint_records(checkpoint) != provenance['checkpoint_files']:
            raise ValueError('adapter files changed between the digest check and loading')
        _start_config(checkpoint)
        result['config'] = asdict(model.config)
        for parameter in model.adapter_parameters():
            parameter.requires_grad_(True)
        result['requires_grad_parameter_names'] = sorted(n for n, p in model.adapters()[14].side.named_parameters()
                                                         if p.requires_grad)
        result['train'] = train(model, hop1_train, schedule, pad_id, LR, BATCH, output, dev_hop1, prior,
                                forward_fn=forward, curve_evaluator=curve)
        tick = time.monotonic()
        summary, records = _reviewed(model, dev, pad_id, output/'evaluation', prior, STANDARD)
        by_hop = reviewed_by_hop(records, dev, prior)
        result['evaluation'] = {'summary': summary, 'by_hop': by_hop, 'seconds': time.monotonic()-tick}
        result['hop1_gate'] = hop1_gate(by_hop)
        export_arm(model, output/'adapter')
    with load_arm(base, output/'adapter') as reloaded:
        if asdict(reloaded.config) != result['config']:
            raise AssertionError('checkpoint reload configuration changed')
        reload_summary, reload_records = _reviewed(reloaded, dev, pad_id, output/'reload', prior, STANDARD)
        if reload_records != records or reload_summary != summary:
            raise AssertionError('checkpoint reload changed full per-example records or summaries')
    result['checkpoint_reload'] = {'full_records_identical': True, 'summaries_identical': True}
    result['checkpoint_files'] = checkpoint_records(output/'adapter')


def _initial_checks(arm, model, dev, pad_id):
    """V3-08: exact logits against the shared S4 forward of the same weights (V3-03 from the
    common start). V3-07: the recorded step-0 gap (max |dlogit| on one dev batch, normal memory,
    against the common start at S=1) and dev hop1 EM before training (E-0022@2)."""
    from v2_memory import collate
    from v3_loop import EVAL_BATCH, _snapshots
    device = next(model.parameters()).device
    batch = collate(dev[:EVAL_BATCH], pad_id, device)
    side = model.adapters()[14].side
    model.eval()
    result = {}
    with torch.no_grad():
        snapshot = _snapshots(model, batch, 'normal', None)
        logits, _, _ = _capture(model, batch, snapshot)
        if arm == 'V3-08':
            side.reader.__class__ = LatentReader  # same weights, shared Rq/Ro: the V3-03 forward
            try:
                shared, _, _ = _capture(model, batch, snapshot)
            finally:
                side.reader.__class__ = UntiedReader
            result['untied_vs_shared_s4'] = {'rows': len(batch.rows), 'identical': bool(torch.equal(logits, shared)),
                                             'max_abs': float((logits-shared).abs().max())}
            if not result['untied_vs_shared_s4']['identical']:
                raise AssertionError('V3-08 initial function differs from the shared S4 forward')
        if arm == 'V3-07':
            hop1 = [r for r in dev if r['axes']['hop'] == 1]
            records, _ = evaluate_arm(model, hop1, pad_id, 1, ('normal',))
            result['widened_step0'] = {'dev_hop1_em': _mean(r['correct'] for r in records['normal']),
                                       'dev_hop1_rows': len(hop1)}
    return result, logits


def run_arm(args, output, result):
    from calibrate import load_base
    from v2_memory import isolated_support
    from v2_train import adapted, train
    from v2r16_memory import train_forward
    from v2r16_train import checkpoint_records
    from v3_loop import attach_paths, compare_to_reviewed
    arm = args.arm
    provenance = start_provenance(args.start_run, args.start_sha256, smoke=bool(args.smoke_steps),
                                  ready_run=args.ready_run)
    result['start'] = provenance
    start_directory = Path(args.start_run)/'adapter'
    tokenizer, pad_id = _tokenizer()
    train_rows, dev, prior, result['data'] = _data(tokenizer, args.eval_limit)
    schedule = v3_schedule(train_rows, 'train')
    if args.smoke_steps:
        schedule = trim_smoke(schedule, train_rows, args.smoke_steps)
    result['schedule'] = {k: v for k, v in schedule.items() if k != 'batches'}
    result['macs'] = macs_per_question_token(arm)
    base = load_base()
    forward = partial(train_forward, support_fn=isolated_support)
    s_train = ARMS[arm]['steps']
    scan = UNTIED_S_EVAL if ARMS[arm].get('untied') else S_EVAL
    if arm == 'V3-07':
        from v2_memory import collate
        from v3_loop import EVAL_BATCH, _snapshots
        with load_arm(base, start_directory) as start_model, torch.no_grad():
            start_model.eval()
            batch = collate(dev[:EVAL_BATCH], pad_id, next(start_model.parameters()).device)
            start_logits, _, _ = _capture(start_model, batch, _snapshots(start_model, batch, 'normal', None))
    with arm_model(base, arm, start_directory) as model:
        if checkpoint_records(start_directory) != provenance['checkpoint_files']:
            raise ValueError('common start files changed between the digest check and loading')
        result['config'] = asdict(model.config)
        result['variant'] = asdict(side_variant(model.adapters()[14].side))
        result['initial'], logits = _initial_checks(arm, model, dev, pad_id)
        if arm == 'V3-07':
            result['initial']['widened_step0']['max_abs_logit_difference'] = float((logits-start_logits).abs().max())
            result['initial']['widened_step0']['rows'] = len(logits)
            del start_logits
        for parameter in model.adapter_parameters():
            parameter.requires_grad_(True)
        side = model.adapters()[14].side
        result['requires_grad_parameter_names'] = sorted(n for n, p in side.named_parameters() if p.requires_grad)
        result['train'] = train(model, train_rows, schedule, pad_id, LR, BATCH, output, None, None,
                                forward_fn=forward, learning_curves=False)
        model.eval()
        result['evaluation'], result['reconstruction'], result['evaluation_seconds'] = {}, {}, {}
        main_records = None
        for steps in scan:
            torch.cuda.synchronize()
            tick = time.monotonic()
            controls = ('normal', 'zero', 'shuffled', 'permuted') if steps == s_train else ('normal',)
            records, reconstruction = evaluate_arm(model, dev, pad_id, steps, controls)
            torch.cuda.synchronize()
            result['evaluation_seconds'][str(steps)] = time.monotonic()-tick
            directory = output/f'eval-s{steps}'
            directory.mkdir()
            result['evaluation'][str(steps)] = {}
            for mode, mode_records in records.items():
                attach_paths(mode_records, dev)
                result['evaluation'][str(steps)][mode] = summarize_arm(mode_records, prior)
                _write_records(directory/f'dev.{mode}.jsonl', mode_records)
            result['reconstruction'][str(steps)] = reconstruction
            if steps == s_train:
                main_records = records
        if s_train == 1 and side_variant(side) == Variant():
            # Implementation check: the S=1 arm equals the reviewed E-0016 evaluator example by example.
            _, reviewed = _reviewed(model, dev, pad_id, output/'reviewed-s1', prior,
                                    ('normal', 'zero', 'shuffled', 'permuted'))
            result['s1_vs_reviewed'] = {m: compare_to_reviewed(main_records[m], reviewed[m]) for m in main_records}
            if not all(c['identical'] for c in result['s1_vs_reviewed'].values()):
                raise AssertionError('S=1 arm differs from the reviewed evaluator: implementation check failed')
        export_arm(model, output/'adapter')
    with load_arm(base, output/'adapter') as reloaded:
        if asdict(reloaded.config) != result['config'] or asdict(side_variant(reloaded.adapters()[14].side)) != result['variant']:
            raise AssertionError('checkpoint reload configuration changed')
        reload_records, _ = evaluate_arm(reloaded, dev, pad_id, s_train)
        for mode in reload_records:
            attach_paths(reload_records[mode], dev)
        if reload_records != main_records:
            raise AssertionError('checkpoint reload changed full per-example records')
    result['checkpoint_reload'] = {'full_records_identical': True, 'steps': s_train}
    result['checkpoint_files'] = checkpoint_records(output/'adapter')
    result['reads_per_question_token'] = 0 if ARMS[arm].get('kernel') == 'tokenwise' else s_train


def run_intervene(args, output, result):
    from calibrate import load_base
    from v2r16_train import checkpoint_records
    run = Path(args.arm_run)
    manifest = json.loads((run/'manifest.json').read_text())
    formal = manifest.get('eligible_as_evidence') is True
    if (manifest.get('status') != 'complete' or manifest.get('contract') != CONTRACT
            or manifest.get('arm') not in ('V3-02', 'V3-03')
            or not (formal or (args.eval_limit and manifest.get('smoke_only') is True))):
        raise ValueError('interventions need a completed formal V3-02 or V3-03 run (smoke: a smoke run)')
    files = checkpoint_records(run/'adapter')
    if files != manifest.get('checkpoint_files') or files['adapter.pt']['sha256'] != args.arm_sha256:
        raise ValueError('arm adapter differs from its manifest or the pinned digest')
    result['arm_run'] = {'run': str(run.resolve()), 'arm': manifest['arm'], 'checkpoint_files': files}
    tokenizer, pad_id = _tokenizer()
    _, dev, _, result['data'] = _data(tokenizer, args.eval_limit)
    base = load_base()
    with load_arm(base, run/'adapter') as model:
        summary, records, checks = intervene(model, dev, pad_id, ARMS[manifest['arm']]['steps'])
    result['interventions'] = summary
    result['checks'] = checks
    for s, by_name in records.items():
        for name, group in by_name.items():
            _write_records(output/f'intervention-s{s}.{name}.jsonl', group)


def analysis_reference():
    """The pinned released T2 data the ten arms must have used: digests, the training order and
    token count of the shared 'train' schedule, and the ordered dev identities (id, hop, target)."""
    from v3_eval import load_released_data
    from v3_loop import T2_DATA, T2_METADATA_SHA256
    metadata, train_rows = load_released_data(T2_DATA, T2_METADATA_SHA256, 'train')
    _, dev = load_released_data(T2_DATA, T2_METADATA_SHA256, 'dev')
    schedule = v3_schedule(train_rows, 'train')
    return {'metadata_sha256': T2_METADATA_SHA256, 'train_sha256': metadata['splits']['train']['sha256'],
            'dev_sha256': metadata['splits']['dev']['sha256'], 'order_sha256': schedule['order_sha256'],
            'tokens': schedule['tokens'], 'order_seed': schedule['order_seed'],
            'dev': [(r['episode_id'], r['axes']['hop'], r['query']['answer_token_id']) for r in dev]}


def _verified_artifact(manifest, path):
    """The file's sha256 and size must equal the producer manifest's artifact record."""
    from v2r16_train import file_record
    entries = [a for a in manifest.get('artifacts', []) if Path(a['path']) == path.resolve()]
    if len(entries) != 1 or {k: entries[0].get(k) for k in ('sha256', 'bytes')} != file_record(path):
        raise ValueError(f'{path} differs from its producer manifest')


def validate_arm_run(run, reference, start):
    """One formal arm run of E-0022@2, bound to the passed common start, the shared schedule and the
    released dev set. Returns (arm, metrics, records) with records re-derived and cross-checked."""
    from v2_memory import config
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    arm = manifest.get('arm')
    if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True
            or manifest.get('eligible_as_evidence') is not True or manifest.get('smoke_only') is not False
            or manifest.get('smoke_steps') or manifest.get('eval_limit') or manifest.get('contract') != CONTRACT
            or manifest.get('command') != 'arm' or arm not in ARMS):
        raise ValueError(f'{run} is not a completed formal {CONTRACT} arm run')
    _verified_artifact(manifest, run/'metrics.json')
    metrics = json.loads((run/'metrics.json').read_text())
    expected_config = asdict(arm_config(config('direct_hidden', 'soft_slot', 'F_out'), arm))
    if (json.loads(json.dumps(metrics.get('config'))) != json.loads(json.dumps(expected_config))
            or metrics.get('variant') != asdict(arm_variant(arm))):
        raise ValueError(f'{arm}: configuration or variant differs from the arm definition')
    started = metrics.get('start', {})
    if (started.get('expected_adapter_sha256') != start['expected_adapter_sha256']
            or started.get('formal_start') is not True or started.get('run') != start['run']
            or started.get('checkpoint_files') != start['checkpoint_files']):
        raise ValueError(f'{arm}: not trained from the passed common start')
    schedule, train = metrics.get('schedule', {}), metrics.get('train', {})
    if (schedule.get('steps') != STEPS or 'smoke_steps' in schedule or schedule.get('stage') != 'train'
            or schedule.get('order_sha256') != reference['order_sha256'] or schedule.get('tokens') != reference['tokens']
            or train.get('steps') != STEPS or train.get('tokens') != reference['tokens']):
        raise ValueError(f'{arm}: not the shared 6000-step training schedule')
    data = metrics.get('data', {})
    if (data.get('metadata_sha256') != reference['metadata_sha256'] or data.get('dev_sha256') != reference['dev_sha256']
            or data.get('train_sha256') != reference['train_sha256'] or data.get('dev_rows') != len(reference['dev'])):
        raise ValueError(f'{arm}: not the pinned released T2 data')
    reload = metrics.get('checkpoint_reload', {})
    steps = ARMS[arm]['steps']
    if reload.get('full_records_identical') is not True or reload.get('steps') != steps:
        raise ValueError(f'{arm}: reload check missing')
    path = run/f'eval-s{steps}'/'dev.normal.jsonl'
    _verified_artifact(manifest, path)
    records = [json.loads(line) for line in path.open()]
    if [(r['episode_id'], r['hop'], r['target']) for r in records] != reference['dev']:
        raise ValueError(f'{arm}: main records are not the released dev episodes in order')
    if any(type(r['prediction']) is not int or r['correct'] is not (r['prediction'] == r['target'])
           or r['steps'] != steps for r in records):
        raise ValueError(f'{arm}: inconsistent per-example correctness')
    summary = metrics['evaluation'][str(steps)]['normal']
    for hop in range(1, MAX_HOP+1):
        group = [r for r in records if r['hop'] == hop]
        derived = {'count': len(group), 'correct_count': sum(r['correct'] for r in group),
                   'em': sum(r['correct'] for r in group)/len(group)}
        if {k: summary[f'hop{hop}'][k] for k in derived} != derived:
            raise ValueError(f'{arm}: stored hop{hop} summary disagrees with its records')
    return arm, metrics, records


def analyze_runs(runs, reference, start):
    """Rules and bootstrap from the validated records of exactly one run per arm."""
    validated = {}
    for run in runs:
        arm, metrics, records = validate_arm_run(run, reference, start)
        if arm in validated:
            raise ValueError('one run per arm')
        validated[arm] = (Path(run), metrics, records)
    if set(validated) != set(ARMS):
        raise ValueError('all ten arms are required')
    main, correct, table = {}, {}, {}
    for arm, (run, metrics, records) in sorted(validated.items()):
        main[arm] = {f'hop{h}': {'count': sum(r['hop'] == h for r in records),
                                 'correct_count': sum(r['correct'] for r in records if r['hop'] == h)}
                     for h in (1, 2, 3)}
        correct[arm] = {h: [int(r['correct']) for r in records if r['hop'] == h] for h in (1, 2, 3)}
        evaluation = metrics['evaluation'][str(ARMS[arm]['steps'])]
        table[arm] = {
            'run': str(run.resolve()),
            'main': {mode: {f'hop{x}': {k: evaluation[mode][f'hop{x}'][k] for k in ('em', 'nll', 'count', 'prior_em')}
                            for x in (1, 2, 3)} for mode in evaluation},
            'scan': {s: {f'hop{x}': metrics['evaluation'][s]['normal'][f'hop{x}']['em'] for x in (1, 2, 3)}
                     for s in metrics['evaluation']},
            'stored_parameters': metrics['train']['stored_parameters'],
            'stable_step_seconds': metrics['train']['stable_step_seconds'],
            'node_hours': metrics['node_hours'], 'macs': metrics['macs'],
            'reads_per_question_token': metrics['reads_per_question_token']}
    decision = decide(main)
    star = decision['S_M*']
    pairs = [(star, other) for other in ('V3-01', 'V3-05', 'V3-06', 'V3-07', 'V3-04', 'V3-08', 'V3-09', 'V3-10')
             if other != star] + [('V3-03', 'V3-05'), ('V3-02', 'V3-01'), ('V3-03', 'V3-01')]
    pairs = list(dict.fromkeys(pairs))
    return {'decision': decision, 'arms': table,
            'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES, 'seed': SEED, 'stratified_by': 'hop', 'paired': True,
                          'intervals_pp': bootstrap(correct, pairs), 'descriptive_only': True},
            'inputs': {'start': start, 'reference': {k: v for k, v in reference.items() if k != 'dev'},
                       'dev_rows': len(reference['dev'])}}


def run_analyze(args, output, result):
    """Validates the ten formal arm runs against the passed common start and the released data,
    then applies the rules and the bootstrap to their records."""
    start = start_provenance(args.start_run, args.start_sha256, ready_run=args.ready_run)
    if start['formal_start'] is not True:
        raise ValueError('analysis needs the formal common start')
    result.update(analyze_runs(args.arm_runs, analysis_reference(), start))


def _write_records(path, records):
    with Path(path).open('w') as handle:
        for record in records:
            handle.write(json.dumps({k: v for k, v in record.items() if not k.startswith('_')}, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'arm', 'intervene', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--start-run', type=Path, help='the V3-S0 run directory (arm, analyze)')
    parser.add_argument('--start-sha256', help='pinned adapter.pt digest of the common start (arm, analyze)')
    parser.add_argument('--ready-run', type=Path, help='E-0023 stage-0 readiness run of the common start (arm, analyze)')
    parser.add_argument('--arm-run', type=Path, help='the S_M* arm run directory (intervene)')
    parser.add_argument('--arm-sha256', help='pinned adapter.pt digest of the S_M* arm (intervene)')
    parser.add_argument('--arm-runs', type=Path, nargs='*', default=(), help='the ten arm run directories (analyze)')
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--eval-limit', type=int, default=0, help='smoke only: first N dev rows per hop')
    args = parser.parse_args()
    if args.command == 'arm' and (args.arm is None or args.start_run is None or args.start_sha256 is None):
        parser.error('arm needs --arm, --start-run and --start-sha256')
    if args.command == 'analyze' and (args.start_run is None or args.start_sha256 is None or len(args.arm_runs) != len(ARMS)):
        parser.error('analyze needs --start-run, --start-sha256 and the ten --arm-runs')
    if args.command == 'intervene' and (args.arm_run is None or args.arm_sha256 is None):
        parser.error('intervene needs --arm-run and --arm-sha256')
    if args.command != 'arm' and args.arm:
        parser.error('--arm belongs to the arm command')
    if not 0 <= args.smoke_steps <= STEPS or (args.smoke_steps and args.command not in ('start', 'arm')):
        parser.error('smoke steps 0..6000 for start/arm only')
    if args.eval_limit and (args.eval_limit < 2 or args.command == 'analyze'):
        parser.error('eval limit >= 2 and not for analyze')
    if args.command in ('start', 'arm') and bool(args.eval_limit) != bool(args.smoke_steps):
        parser.error('smoke training and a dev limit go together')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    from calibrate import MODEL, REVISION
    from v2_train import write_json
    from v2r16_train import file_record
    started = time.monotonic()
    smoke = bool(args.smoke_steps or args.eval_limit)
    arm_name = {'start': 'V3-S0', 'arm': args.arm, 'intervene': 'intervention', 'analyze': 'analysis'}[args.command]
    result, passed, failure = {'experiment': CONTRACT, 'command': args.command, 'arm': arm_name,
                               'smoke_only': smoke}, False, None
    try:
        {'start': run_start, 'arm': run_arm, 'intervene': run_intervene, 'analyze': run_analyze}[args.command](
            args, output, result)
        passed = True
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        wall = time.monotonic()-started
        result.update(wall_seconds=wall, mig_instance_hours=wall/3600, node_hours=wall/14400,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        write_json(output/'metrics.json', _jsonable(result))
        artifacts = [{'path': str(p.resolve()), 'role': str(p.relative_to(output)), **file_record(p)}
                     for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']
        gate_result = result.get('hop1_gate')
        write_json(output/'manifest.json', _jsonable({
            'status': 'complete' if passed else 'failed', 'experiment': EXPERIMENT, 'contract': CONTRACT,
            'command': args.command, 'arm': arm_name, 'write_protocol': 'isolated',
            'spec_digest': os.environ.get('SR_SPEC_DIGEST'), 'model': MODEL, 'revision': REVISION,
            'smoke_steps': args.smoke_steps, 'eval_limit': args.eval_limit, 'smoke_only': smoke,
            'eligible_as_evidence': passed and not smoke, 'lr': LR, 'batch_size': BATCH,
            'schedule': result.get('schedule'), 'config': result.get('config'), 'variant': result.get('variant'),
            'start': result.get('start') or result.get('adapter'), 'data': result.get('data'),
            'checkpoint_files': result.get('checkpoint_files'),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'hardware': ({'device': torch.cuda.get_device_name(0),
                          'multiprocessors': torch.cuda.get_device_properties(0).multi_processor_count}
                         if torch.cuda.is_available() else None),
            'metrics': {'wall_seconds': wall, 'mig_instance_hours': wall/3600, 'node_hours': wall/14400,
                        **({'hop1_gate': gate_result} if gate_result else {}),
                        **({'decision': result['decision']} if 'decision' in result else {})},
            'artifacts': artifacts, 'failure': failure,
            'validator': {'name': 'v3_train', 'passed': passed}}))
    return 0


def _jsonable(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


if __name__ == '__main__':
    raise SystemExit(main())
