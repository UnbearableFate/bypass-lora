"""E-0012 L0--L4 measurements. Run only in an allocated Miyabi-G job.

Frozen features are captured before RoPE from the unadapted Base, with separate
support/question forwards. Probe fitting and model selection use train only.
The six L0 controls must pass before full-run measurements can be interpreted.
This program measures; it never trains an M-line adapter or chooses a route.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from bypass_core import elu_plus_one, rmsnorm_no_affine
from calibrate import MODEL, REVISION, load_base
from v2_data import RELATIONS
from v2_memory import Snapshot, build_support, collate, config, pad, read_vectors
from v2_probe import validate_probe_splits
from v2_schedule import DATA_COUNTS, load_data
from v2_train import adapted


TAPS = ('direct_hidden', 'qknorm_pre_rope')
QUANTILES = (0., .05, .25, .5, .75, .95, 1.)
L2_GRID = (0., 1e-4, 1e-2)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def name_end(offsets, start, end):
    """The token covering the final character, including multi-token bare names."""
    matches = [i for i, (a, b) in enumerate(offsets) if a <= end - 1 < b and b > start]
    if start >= end or len(matches) != 1:
        raise ValueError('name end must belong to exactly one tokenizer offset')
    return matches[0]


def token_positions(tok, episode):
    support = episode['support_text']
    encoded = tok(support, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != episode['support_input_ids']:
        raise ValueError('support tokenization differs from released data')
    heads, tails, delimiters = [], [], []
    cursor = 0
    for fact in episode['facts']:
        text = f"{fact['head']} {fact['relation']} {fact['tail']} ;\n"
        if support[cursor:cursor + len(text)] != text:
            raise ValueError('unexpected fact rendering')
        head_end = cursor + len(fact['head'])
        tail_start = head_end + 1 + len(fact['relation']) + 1
        tail_end = tail_start + len(fact['tail'])
        heads.append(name_end(encoded['offset_mapping'], cursor, head_end))
        tails.append(name_end(encoded['offset_mapping'], tail_start, tail_end))
        delim = name_end(encoded['offset_mapping'], tail_end + 1, tail_end + 2)
        if delim != fact['delimiter_token_index'] or not heads[-1] <= tails[-1] < delim:
            raise ValueError('delimiter/name position mismatch')
        delimiters.append(delim)
        cursor += len(text)
    if cursor != len(support):
        raise ValueError('unaccounted support text')
    question = episode['question_text']
    prefix = 'Who does '
    if not question.startswith(prefix + episode['query']['head'] + ' '):
        raise ValueError('unexpected question rendering')
    encoded = tok(question, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != episode['question_input_ids']:
        raise ValueError('question tokenization differs from released data')
    qhead = name_end(encoded['offset_mapping'], len(prefix), len(prefix) + len(episode['query']['head']))
    return {'head': heads, 'tail': tails, 'delimiter': delimiters,
            'question': [qhead, len(encoded['input_ids']) - 1]}


class _CapturedLayer(Exception):
    pass


@torch.no_grad()
def capture_base(base, ids, mask):
    """Stop after layer-14 attention/norm; neither later layers nor LM head run."""
    layer = base.model.layers[14]
    if hasattr(layer, 'original'):
        raise ValueError('feature extraction requires an unadapted Base')
    seen, handles = {}, []

    def hidden(module, args, output):
        seen.update(n=output, residual=args[0])

    handles.append(layer.input_layernorm.register_forward_hook(hidden))
    for key, module in (('q', layer.self_attn.q_norm), ('k', layer.self_attn.k_norm),
                        ('v', layer.self_attn.v_proj)):
        def feature(module, args, output, key=key):
            seen[key] = output.flatten(2)
        handles.append(module.register_forward_hook(feature))

    def finish(module, args, output):
        seen.update(h=args[0], u=output)
        raise _CapturedLayer

    handles.append(layer.post_attention_layernorm.register_forward_hook(finish))
    try:
        try:
            base.model(ids, attention_mask=mask,
                       position_ids=(mask.long().cumsum(-1) - 1).clamp_min(0), use_cache=False)
        except _CapturedLayer:
            pass
        else:
            raise AssertionError('layer-14 capture did not execute')
    finally:
        for handle in handles:
            handle.remove()
    if set(seen) != {'n', 'q', 'k', 'v', 'residual', 'h', 'u'}:
        raise AssertionError('incomplete Base capture')
    return seen


def labels_for(rows, names):
    names = {name: i for i, name in enumerate(names)}
    relations = {name: i for i, name in enumerate(RELATIONS)}
    return {
        'head': torch.tensor([[names[f['head']] for f in e['facts']] for e in rows]),
        'tail': torch.tensor([[names[f['tail']] for f in e['facts']] for e in rows]),
        'relation': torch.tensor([[relations[f['relation']] for f in e['facts']] for e in rows]),
        'qhead': torch.tensor([names[e['query']['head']] for e in rows]),
        'qrelation': torch.tensor([relations[e['query']['relation']] for e in rows]),
        'target': torch.tensor([e['query']['target_fact_index'] for e in rows]),
    }


@torch.no_grad()
def extract_split(base, tok, rows, names, batch_size, keep_question=False):
    positions = [token_positions(tok, row) for row in rows]
    result = {'labels': labels_for(rows, names), 'positions': positions,
              'episode_ids': [r['episode_id'] for r in rows]}
    n = len(rows)
    qlength = max(len(e['question_input_ids']) for e in rows)
    for start in range(0, n, batch_size):
        subset = rows[start:start + batch_size]
        count = len(subset)
        ids, mask = pad([e['support_input_ids'] for e in subset], tok.pad_token_id, 'cuda')
        captured = capture_base(base, ids, mask)
        owner = torch.arange(count, device='cuda')[:, None]
        for position, fields in (('delimiter', ('n', 'k', 'v')), ('head', ('n', 'k')), ('tail', ('n', 'v'))):
            index = torch.tensor([p[position] for p in positions[start:start + count]], device='cuda')
            for field in fields:
                key = position + '_' + field
                values = captured[field][owner, index].cpu()
                if key not in result:
                    result[key] = torch.empty((n, *values.shape[1:]), dtype=values.dtype)
                result[key][start:start + count] = values
        del captured
        ids, mask = pad([e['question_input_ids'] for e in subset], tok.pad_token_id, 'cuda')
        captured = capture_base(base, ids, mask)
        index = torch.tensor([p['question'] for p in positions[start:start + count]], device='cuda')
        for field in ('n', 'q'):
            key = 'question_' + field
            values = captured[field][owner, index].cpu()
            if key not in result:
                result[key] = torch.empty((n, *values.shape[1:]), dtype=values.dtype)
            result[key][start:start + count] = values
        if keep_question:
            for field in ('n', 'q', 'residual', 'h', 'u'):
                key = 'all_question_' + field
                if key not in result:
                    result[key] = torch.zeros((n, qlength, captured[field].shape[-1]), dtype=captured[field].dtype)
                result[key][start:start + count, :ids.shape[1]] = captured[field].cpu()
            if 'question_mask' not in result:
                result['question_mask'] = torch.zeros(n, qlength, dtype=torch.bool)
            result['question_mask'][start:start + count, :ids.shape[1]] = mask.cpu()
        del captured
    return result


def feature(cache, tap, kind, position='delimiter'):
    field = 'n' if tap == 'direct_hidden' else {'query': 'q', 'key': 'k', 'value': 'v'}[kind]
    return cache[('question_' if kind == 'query' else position + '_') + field]


@torch.no_grad()
def verify_hooks(base, tok, rows, checkpoints):
    """Exact projection-input equality in both phases, plus deployed read check."""
    batch = collate(rows, tok.pad_token_id)
    expected = {
        'support': capture_base(base, batch.support_ids, batch.support_mask),
        'query': capture_base(base, batch.question_ids, batch.question_mask),
    }
    results = []
    for tap in TAPS:
        checkpoint = next(c for c in checkpoints if c['tap'] == tap)
        with adapted(base, checkpoint=checkpoint['checkpoint']) as model:
            model.eval()
            side = model.adapters()[14].side
            seen = {}
            def observe(module, args):
                seen['inputs'] = tuple(x.detach().clone() for x in args)
            handle = side.projection.register_forward_pre_hook(observe)
            try:
                snapshot, _ = build_support(model, batch, observe=True)
                record = {'tap': tap, 'checkpoint': checkpoint['arm'], 'input_atol': 0., 'input_rtol': 0.,
                          'read_atol': 2e-5, 'read_rtol': 2e-5, 'max_abs': {}}
                for phase, mask in (('support', batch.support_mask), ('query', batch.question_mask)):
                    if phase == 'query':
                        output_seen = {}
                        hook = side.register_forward_hook(lambda module, args, output: output_seen.update(z=output[0]))
                        try:
                            model(batch.question_ids, batch.question_mask, memory=snapshot.memory, phase='query')
                        finally:
                            hook.remove()
                    fields = ('n', 'n', 'n') if tap == 'direct_hidden' else ('q', 'k', 'v')
                    for name, actual, field in zip(('q', 'k', 'v'), seen['inputs'], fields):
                        wanted = expected[phase][field]
                        error = float((actual[mask].float() - wanted[mask].float()).abs().max())
                        record['max_abs'][phase + '.' + name] = error
                        if error != 0.:
                            raise AssertionError('frozen and adapter projection inputs differ')
                    if phase == 'query':
                        qbar = expected['query'][fields[0]].float() @ side.projection.A_q
                        target = torch.tensor([r['query']['target_fact_index'] for r in rows], device='cuda')
                        raw, *_ = read_vectors(qbar, snapshot, side.reader, model.config.backend, target)
                        z = (qbar + side.reader.eta * (raw * side.reader.read_gain @ side.reader.Ro)) - qbar
                        z *= mask[..., None]
                        record['read_z_max_abs'] = float((z - output_seen['z']).abs().max())
                        torch.testing.assert_close(z, output_seen['z'], atol=2e-5, rtol=2e-5)
                results.append(record)
            finally:
                handle.remove()
    return results


def standardizer(x):
    mean = x.mean(0)
    scale = x.var(0, unbiased=False).sqrt().clamp_min(1e-5)
    return mean, scale


def selection_key(record):
    # Fixed before evaluation: maximize train-internal accuracy, then minimize CE;
    # exact ties prefer stronger L2 and earlier stopping.
    return (record['correct'] / record['n'], -record['nll'], record['l2'], -record['epoch'])


@torch.no_grad()
def classifier_score(model, x, y, batch_size=8192):
    correct, loss = 0, 0.
    for start in range(0, len(x), batch_size):
        target = y[start:start + batch_size]
        logits = model(x[start:start + batch_size])
        correct += int((logits.argmax(-1) == target).sum())
        loss += float(F.cross_entropy(logits, target, reduction='sum'))
    return {'correct': correct, 'n': len(y), 'accuracy': correct / len(y), 'nll': loss / len(y)}


def fit_softmax(x, y, validation_x, validation_y, classes, *, max_epochs=32, l2_grid=L2_GRID,
                batch_size=8192):
    """No calibration/dev argument exists at the fitting or selection boundary."""
    mean, scale = standardizer(x)
    x = (x - mean) / scale
    validation_x = (validation_x - mean) / scale
    best, best_state, trials = None, None, []
    epochs = sorted({max(1, max_epochs // 8), max(1, max_epochs // 4), max(1, max_epochs // 2), max_epochs})
    for l2 in l2_grid:
        torch.manual_seed(0)
        model = nn.Linear(x.shape[-1], classes, device=x.device)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        generator = torch.Generator(device=x.device).manual_seed(0)
        for epoch in range(1, max_epochs + 1):
            order = torch.randperm(len(x), generator=generator, device=x.device)
            for index in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[index]), y[index]) + .5 * l2 * model.weight.square().sum()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite softmax probe loss')
                loss.backward()
                optimizer.step()
            if epoch in epochs:
                score = classifier_score(model, validation_x, validation_y)
                record = {**score, 'epoch': epoch, 'l2': l2}
                trials.append(record)
                if best is None or selection_key(record) > selection_key(best):
                    best = record
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        del optimizer
    model.load_state_dict(best_state)
    return model, mean, scale, {'selected': best, 'trials': trials, 'fit_examples': len(y),
                                'standardization': 'per-channel mean/std, fitting episodes only; std floor 1e-5'}


def constant_baseline(train_y, test_y, classes):
    counts = torch.bincount(train_y.flatten(), minlength=classes)
    label = int(counts.argmax())
    correct = int((test_y == label).sum())
    return {'class': label, 'train_count': int(counts[label]), 'train_n': train_y.numel(),
            'correct': correct, 'n': test_y.numel(), 'accuracy': correct / test_y.numel()}


def measure_linear(train_x, train_y, calibration_x, calibration_y, fit_episodes, classes, *, smoke=False,
                   positive_control=False):
    fit_x = train_x[:fit_episodes].reshape(-1, train_x.shape[-1]).to('cuda', dtype=torch.float32)
    internal_x = train_x[fit_episodes:].reshape(-1, train_x.shape[-1]).to('cuda', dtype=torch.float32)
    fit_y = train_y[:fit_episodes].flatten().cuda()
    internal_y = train_y[fit_episodes:].flatten().cuda()
    epochs = 4 if smoke else (64 if positive_control else 32)
    size = 1024 if train_y.ndim == 1 else 8192
    tick = time.monotonic()
    model, mean, scale, selection = fit_softmax(fit_x, fit_y, internal_x, internal_y, classes,
        max_epochs=epochs, l2_grid=(1e-4,) if smoke else L2_GRID, batch_size=size)
    with torch.no_grad():
        test_x = calibration_x.reshape(-1, calibration_x.shape[-1]).to('cuda', dtype=torch.float32)
        test_y = calibration_y.flatten().cuda()
        score = classifier_score(model, (test_x - mean) / scale, test_y)
    return {**score, 'random_accuracy': 1 / classes,
            'constant_baseline': constant_baseline(train_y, calibration_y, classes),
            'selection': selection, 'seconds': time.monotonic() - tick}


class BilinearAddress(nn.Module):
    def __init__(self, query_dim, key_dim, rank, device):
        super().__init__()
        self.U = nn.Parameter(torch.randn(query_dim, rank, device=device) / math.sqrt(query_dim) / rank ** .25)
        self.V = nn.Parameter(torch.randn(key_dim, rank, device=device) / math.sqrt(key_dim) / rank ** .25)

    def forward(self, inputs):
        query, keys = inputs
        return torch.einsum('br,bnr->bn', query @ self.U, keys @ self.V)


@torch.no_grad()
def address_score(model, q, k, y, batch_size=512):
    correct, loss = 0, 0.
    for start in range(0, len(q), batch_size):
        target = y[start:start + batch_size]
        logits = model((q[start:start + batch_size], k[start:start + batch_size]))
        correct += int((logits.argmax(-1) == target).sum())
        loss += float(F.cross_entropy(logits, target, reduction='sum'))
    return {'correct': correct, 'n': len(y), 'accuracy': correct / len(y), 'nll': loss / len(y)}


def measure_address(train_q, train_k, train_y, test_q, test_k, test_y, fit_episodes, rank, *, smoke=False):
    q, k, y = train_q.float().cuda(), train_k.float().cuda(), train_y.cuda()
    qmean, qscale = standardizer(q[:fit_episodes])
    kmean, kscale = standardizer(k[:fit_episodes].flatten(0, 1))
    q, k = (q - qmean) / qscale, (k - kmean) / kscale
    best, best_state, trials = None, None, []
    max_epochs = 2 if smoke else 32
    epochs = sorted({max(1, max_epochs // 8), max(1, max_epochs // 4), max(1, max_epochs // 2), max_epochs})
    tick = time.monotonic()
    for l2 in ((1e-4,) if smoke else L2_GRID):
        torch.manual_seed(0)
        model = BilinearAddress(q.shape[-1], k.shape[-1], rank, q.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        generator = torch.Generator(device=q.device).manual_seed(0)
        for epoch in range(1, max_epochs + 1):
            order = torch.randperm(fit_episodes, generator=generator, device=q.device)
            for index in order.split(512):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model((q[index], k[index])), y[index])
                loss += .5 * l2 * (model.U.square().sum() + model.V.square().sum())
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite address probe loss')
                loss.backward()
                optimizer.step()
            if epoch in epochs:
                score = address_score(model, q[fit_episodes:], k[fit_episodes:], y[fit_episodes:])
                record = {**score, 'epoch': epoch, 'l2': l2}
                trials.append(record)
                if best is None or selection_key(record) > selection_key(best):
                    best = record
                    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        del optimizer
    model.load_state_dict(best_state)
    with torch.no_grad():
        tq = (test_q.float().cuda() - qmean) / qscale
        tk = (test_k.float().cuda() - kmean) / kscale
        score = address_score(model, tq, tk, test_y.cuda())
    return {**score, 'rank': rank, 'random_accuracy': 1 / train_k.shape[1],
            'constant_baseline': constant_baseline(train_y, test_y, train_k.shape[1]),
            'selection': {'selected': best, 'trials': trials, 'fit_examples': fit_episodes,
                          'standardization': 'fitting episodes only; separate query/key channel stats'},
            'seconds': time.monotonic() - tick}


@torch.no_grad()
def project_cache(side, cache, tap, batch_size=512):
    keys, values, queries = [], [], []
    qf, kf, vf = (feature(cache, tap, kind) for kind in ('query', 'key', 'value'))
    for start in range(0, len(qf), batch_size):
        q, k, v = side.projection(qf[start:start + batch_size].cuda(), kf[start:start + batch_size].cuda(),
                                  vf[start:start + batch_size].cuda())
        keys.append(elu_plus_one(rmsnorm_no_affine(k, side.reader.eps_norm)).cpu())
        values.append(v.cpu())
        queries.append(elu_plus_one(rmsnorm_no_affine(q, side.reader.eps_norm) @ side.reader.Rq).cpu())
    return {'key': torch.cat(keys), 'value': torch.cat(values), 'query': torch.cat(queries)}


def distribution(values):
    values = values.detach().float().flatten().cpu()
    if not len(values) or not torch.isfinite(values).all():
        raise ValueError('empty/nonfinite distribution')
    return {'n': len(values), 'mean': float(values.mean()),
            'quantiles': {str(q): float(v) for q, v in zip(QUANTILES, torch.quantile(values, torch.tensor(QUANTILES)))}}


def slot_summary(argmax, mask, relations, query_relations, targets):
    argmax, mask = argmax.cpu(), mask.cpu()
    owner = torch.arange(len(argmax))[:, None].expand_as(argmax)
    same = relations[owner, argmax] == query_relations[:, None]
    target = argmax == targets[:, None]
    selected = argmax[mask]
    n = len(selected)
    if n == 0:
        raise ValueError('slot summary has no valid questions')
    counts = torch.bincount(selected, minlength=relations.shape[1]).tolist()
    return {'n': n, 'slot_counts': counts, 'slot_frequencies': [x / n for x in counts],
            'same_relation_count': int(same[mask].sum()), 'same_relation_rate': float(same[mask].float().mean()),
            'target_count': int(target[mask].sum()), 'target_rate': float(target[mask].float().mean())}


@torch.no_grad()
def geometry(base, side, cache, tap, backend, batch_size=64):
    qf = cache['all_question_n' if tap == 'direct_hidden' else 'all_question_q']
    kf, vf = feature(cache, tap, 'key'), feature(cache, tap, 'value')
    labels, mask = cache['labels'], cache['question_mask']
    argmaxes, margins, cosines, ratios, post_ratios = [], [], [], [], []
    triangle = torch.triu_indices(kf.shape[1], kf.shape[1], offset=1, device='cuda')
    layer = base.model.layers[14]
    if hasattr(layer, 'original'):
        layer = layer.original
    for start in range(0, len(qf), batch_size):
        end = start + batch_size
        q, k, v = side.projection(qf[start:end].cuda(), kf[start:end].cuda(), vf[start:end].cuda())
        keys = elu_plus_one(rmsnorm_no_affine(k, side.reader.eps_norm))
        snapshot = Snapshot({}, keys, v, torch.ones(keys.shape[:2], dtype=torch.bool, device='cuda'))
        target = labels['target'][start:end].cuda()
        raw, weights, scores, _ = read_vectors(q, snapshot, side.reader, backend, target)
        ti = target[:, None, None].expand(-1, q.shape[1], 1)
        other = scores.clone().scatter(-1, ti, -torch.inf).max(-1).values
        margins.append((scores.gather(-1, ti).squeeze(-1) - other).cpu())
        argmaxes.append(weights.argmax(-1).cpu())
        normalized = F.normalize(keys, dim=-1)
        cos = normalized @ normalized.transpose(-1, -2)
        cosines.append(cos[:, triangle[0], triangle[1]].cpu())
        z = (q + side.reader.eta * (raw * side.reader.read_gain @ side.reader.Ro)) - q
        z *= mask[start:end].cuda()[..., None]
        u = cache['all_question_u'][start:end].cuda()
        h = cache['all_question_h'][start:end].cuda()
        residual = cache['all_question_residual'][start:end].cuda()
        if side.fusion.kind == 'F_out':
            delta = z @ side.fusion.P
        else:
            # Difference of the two actual BF16 FFN outputs, before residual add.
            plain = layer.mlp(u)
            changed = side.fusion(torch.zeros_like(h), u, z, layer.mlp, layer.post_attention_layernorm)
            delta = changed.float() - plain.float()
        ratios.append((delta.float().norm(dim=-1) / residual.float().norm(dim=-1).clamp_min(1e-12)).cpu())
        post_ratios.append((delta.float().norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-12)).cpu())
    argmax, margin = torch.cat(argmaxes), torch.cat(margins)
    ratio, post_ratio = torch.cat(ratios), torch.cat(post_ratios)
    last = torch.zeros_like(mask)
    last[torch.arange(len(mask)), mask.sum(-1) - 1] = True
    head = torch.zeros_like(mask)
    head[torch.arange(len(mask)), torch.tensor([p['question'][0] for p in cache['positions']])] = True
    result = {'key_pair_cosine': distribution(torch.cat(cosines)), 'question_positions': {},
              'fusion_increment_definition': 'F_out: FP32 z@P; C_up/C_dual: actual BF16 modified FFN minus Base FFN, before residual addition',
              'residual_denominator': 'layer 14 input residual before attention; post-attention denominator also reported',
              'margin_definition': 'target score minus maximum non-target score; soft_slot includes 1/sqrt(128)'}
    for name, selected in (('all', mask), ('last', last), ('head', head)):
        result[name] = {'read_argmax': slot_summary(argmax, selected, labels['relation'], labels['qrelation'], labels['target']),
                        'target_margin': distribution(margin[selected]),
                        'fusion_to_residual_norm': distribution(ratio[selected]),
                        'fusion_to_post_attention_norm': distribution(post_ratio[selected])}
    for position in range(mask.shape[1]):
        selected = torch.zeros_like(mask)
        selected[:, position] = mask[:, position]
        if selected.any():
            result['question_positions'][str(position)] = slot_summary(argmax, selected, labels['relation'], labels['qrelation'], labels['target'])
    return result


def checkpoints_from_map(path):
    entries = json.loads(Path(path).read_text())['training']
    if len(entries) != 10 or [e['arm'] for e in entries] != [f'V2-{i:02d}' for i in range(1, 11)]:
        raise ValueError('requires ten ordered E-0010 checkpoints')
    result = []
    for entry in entries:
        checkpoint = Path(entry['output']) / '.'.join(entry[k] for k in ('tap', 'backend', 'fusion')) / 'adapter'
        settings = json.loads((checkpoint / 'config.json').read_text())
        if any(settings[k] != entry[k] for k in ('tap', 'backend', 'fusion')):
            raise ValueError('run-map/checkpoint configuration mismatch')
        if settings != json.loads(json.dumps(config(entry['tap'], entry['backend'], entry['fusion']).__dict__)):
            raise ValueError('checkpoint differs from the V2 protocol')
        result.append({**entry, 'checkpoint': str(checkpoint)})
    return result


def markdown(report):
    lines = ['# E-0012 diagnostic measurements', '',
             f"Scope: {report['scope']}. L0 passed: {report.get('l0_pass', 'pending')}. No route or causal conclusion is made.", '',
             '| Level | Feature | Arm | Position / target | Correct / n | Accuracy | Random | Train-frequency constant |',
             '|---|---|---|---|---:|---:|---:|---:|']
    for row in report['probes']:
        label = row['position'] + ' / ' + row['target']
        if 'rank' in row:
            label += f" (r={row['rank']})"
        lines.append(f"| {row['level']} | {row['tap']} | {row.get('arm', 'Base')} | {label} | {row['correct']} / {row['n']} | {row['accuracy']:.4%} | {row['random_accuracy']:.4%} | {row['constant_baseline']['accuracy']:.4%} |")
    if report['l3']:
        lines += ['', '| L3 arm | Clean → clean | Read → read | Clean → read | Fit / held-out episodes |', '|---|---:|---:|---:|---:|']
        for arm, row in report['l3'].items():
            lines.append(f"| {arm} | {row['clean_to_clean']['correct']}/{row['test_examples']} | {row['read_to_read']['correct']}/{row['test_examples']} | {row['clean_to_read']['correct']}/{row['test_examples']} | {row['train_examples']} / {row['test_examples']} |")
    if report['l4']:
        lines += ['', '| L4 arm | Key cosine mean (pairs) | Last-position target argmax / n | Last-position same relation / n | Last margin mean (n) | Last fusion/residual mean (n) |', '|---|---:|---:|---:|---:|---:|']
        for arm, row in report['l4'].items():
            last = row['last']; argmax = last['read_argmax']; margin = last['target_margin']; ratio = last['fusion_to_residual_norm']; cosine = row['key_pair_cosine']
            lines.append(f"| {arm} | {cosine['mean']:.6f} ({cosine['n']}) | {argmax['target_count']} / {argmax['n']} | {argmax['same_relation_count']} / {argmax['n']} | {margin['mean']:.6g} ({margin['n']}) | {ratio['mean']:.6g} ({ratio['n']}) |")
    lines += ['', 'All distributions, quantiles, position histograms, selected hyperparameters, internal-validation scores, hook checks, and input/source hashes are in report.json.', '']
    return '\n'.join(lines)


def save_report(out, report):
    write_json(out / 'report.json', report)
    (out / 'report.md').write_text(markdown(report))


def run_linear_layers(cache, fit_episodes, out, report, smoke):
    train, test = cache['train'], cache['calibration']

    def run(level, tap, position, target, kind, source_position='delimiter', qposition=None):
        tx, cx = feature(train, tap, kind, source_position), feature(test, tap, kind, source_position)
        if qposition is not None:
            tx, cx = tx[:, qposition], cx[:, qposition]
        row = measure_linear(tx, train['labels'][target], cx, test['labels'][target], fit_episodes,
                             4 if 'relation' in target else 92, smoke=smoke, positive_control=level == 'L0')
        report['probes'].append({'level': level, 'tap': tap, 'position': position, 'target': target, **row})
        save_report(out, report)
        print(f"{level} {tap} {position}->{target}: {row['correct']}/{row['n']} ({row['seconds']:.1f}s)", flush=True)

    for tap in TAPS:
        run('L0', tap, 'tail_name_end', 'tail', 'value', 'tail')
        run('L0', tap, 'head_name_end', 'head', 'key', 'head')
        run('L0', tap, 'question_head_end', 'qhead', 'query', qposition=0)
    report['l0_pass'] = all(p['accuracy'] >= .90 for p in report['probes'] if p['level'] == 'L0')
    save_report(out, report)
    if not report['l0_pass'] and not smoke:
        raise RuntimeError('L0 positive control below 90%; do not interpret L1--L4')
    for tap in TAPS:
        for kind, target in (('value', 'tail'), ('key', 'head'), ('key', 'relation')):
            run('L1', tap, 'delimiter', target, kind)
        for index, position in enumerate(('question_head_end', 'question_last')):
            for target in ('qhead', 'qrelation'):
                run('L1', tap, position, target, 'query', qposition=index)
            for rank in ((128,) if smoke else (128, 256, 512, 1024)):
                row = measure_address(feature(train, tap, 'query')[:, index], feature(train, tap, 'key'), train['labels']['target'],
                    feature(test, tap, 'query')[:, index], feature(test, tap, 'key'), test['labels']['target'], fit_episodes, rank, smoke=smoke)
                report['probes'].append({'level': 'L1_address', 'tap': tap, 'position': position, 'target': 'fact_index', **row})
                save_report(out, report)
                print(f"L1 address {tap} {position} r{rank}: {row['correct']}/{row['n']} ({row['seconds']:.1f}s)", flush=True)


def run_projected(base, cache, checkpoints, fit_episodes, out, report, smoke):
    choices = [(f'random.{tap}', tap, None) for tap in TAPS]
    choices += [(c['arm'], c['tap'], c) for c in checkpoints]
    for arm, tap, checkpoint in choices:
        kwargs = {'cfg': config(tap)} if checkpoint is None else {'checkpoint': checkpoint['checkpoint']}
        with adapted(base, **kwargs) as model:
            model.eval()
            side = model.adapters()[14].side
            projected = {s: project_cache(side, cache[s], tap) for s in ('train', 'calibration')}
            for kind, target, position in (('value', 'tail', None), ('key', 'head', None), ('key', 'relation', None),
                                            ('query', 'qhead', 0), ('query', 'qrelation', 0),
                                            ('query', 'qhead', 1), ('query', 'qrelation', 1)):
                tx, cx = projected['train'][kind], projected['calibration'][kind]
                if position is not None:
                    tx, cx = tx[:, position], cx[:, position]
                row = measure_linear(tx, cache['train']['labels'][target], cx, cache['calibration']['labels'][target],
                                     fit_episodes, 4 if 'relation' in target else 92, smoke=smoke)
                name = 'delimiter' if position is None else ('question_head_end' if position == 0 else 'question_last')
                report['probes'].append({'level': 'L2', 'tap': tap, 'arm': arm, 'position': name, 'target': target, **row})
                save_report(out, report)
                print(f"L2 {arm} {kind} {name}->{target}: {row['correct']}/{row['n']} ({row['seconds']:.1f}s)", flush=True)
            if checkpoint is None:
                for backend in ('kernel_matrix', 'soft_slot'):
                    report['l4'][arm + '.' + backend] = geometry(base, side, cache['dev'], tap, backend)
            else:
                report['l4'][arm] = geometry(base, side, cache['dev'], tap, model.config.backend)
            save_report(out, report)
        del projected
        gc.collect()


def run_l3(checkpoints, data, out, report, smoke, batch_size):
    for checkpoint in checkpoints:
        arm = checkpoint['arm']
        output = out / 'l3' / arm
        argv = [sys.executable, str(Path(__file__).with_name('v2_probe.py')), '--checkpoint', checkpoint['checkpoint'],
                '--train', str(data / 'train.jsonl'), '--heldout', str(data / 'calibration.jsonl'),
                '--limit', '64' if smoke else '4096', '--batch-size', str(batch_size), '--output', str(output)]
        # --limit 4096 reads train[:4096] and the entire 2048-row calibration file.
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'stdout.log').open('w') as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        metrics = json.loads((output / 'manifest.json').read_text())['metrics']
        n = metrics['test_examples']
        if metrics['train_examples'] != (64 if smoke else 4096) or n != (64 if smoke else 2048):
            raise ValueError('L3 train/held-out count differs from design')
        row = {'train_examples': metrics['train_examples'], 'test_examples': n, 'ridge': 1.,
               'seconds': metrics['elapsed_s'], 'argv': argv}
        for name in ('clean_to_clean', 'read_to_read', 'clean_to_read'):
            correct = round(metrics[name] * n)
            if abs(correct / n - metrics[name]) > 1e-6:
                raise ValueError('L3 score is inconsistent with an integer count')
            row[name] = {'correct': correct, 'n': n, 'accuracy': metrics[name]}
        report['l3'][arm] = row
        save_report(out, report)
        print(f"L3 {arm}: {metrics['clean_to_clean']:.4f}/{metrics['read_to_read']:.4f}/{metrics['clean_to_read']:.4f} ({metrics['elapsed_s']:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='/work/xg24i002/x10041/bypass-lora')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--l3-batch-size', type=int, default=16)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    project, out = Path(args.project), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = project / 'runs/ATT-0039-1/data'
    run_map = project / 'research/artifacts/T-0011/training-run-map.json'
    report = {'experiment': 'E-0012@1', 'scope': 'smoke, not interpretable' if args.smoke else 'full diagnostic, seed 0',
              'probes': [], 'l3': {}, 'l4': {}, 'timings': {}, 'source_sha256': {}, 'input_sha256': {}}
    start = time.monotonic()
    success = False
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        for path in sorted(Path(__file__).parent.glob('*.py')):
            report['source_sha256'][path.name] = sha256_file(path)
        rows = {split: load_data(data / f'{split}.jsonl', split) for split in DATA_COUNTS}
        for split, values in rows.items():
            if [r['identity']['index'] for r in values] != list(range(DATA_COUNTS[split])):
                raise ValueError('train/split order is not index order')
            if any(len(e['facts']) != 16 for e in values):
                raise ValueError('requires E=16')
            report['input_sha256'][str(data / f'{split}.jsonl')] = sha256_file(data / f'{split}.jsonl')
        validate_probe_splits(rows['train'], rows['calibration'])
        validate_probe_splits(rows['train'], rows['dev'])
        for key in ('episode_id', 'content_sha256'):
            if {e[key] for e in rows['dev']} & {e[key] for e in rows['calibration']}:
                raise ValueError('calibration/dev overlap')
        checkpoints = checkpoints_from_map(run_map)
        for checkpoint in checkpoints:
            for filename in ('config.json', 'adapter.pt'):
                path = Path(checkpoint['checkpoint']) / filename
                report['input_sha256'][str(path)] = sha256_file(path)
        for path in (run_map, data / 'name_pools.json', data / 'metadata.json'):
            report['input_sha256'][str(path)] = sha256_file(path)
        names = [e['name'] for e in json.loads((data / 'name_pools.json').read_text())['pools']['seen']]
        if len(names) != 92:
            raise ValueError('requires 92 released names')
        if args.smoke:
            rows = {s: r[:{'train': 512, 'calibration': 128, 'dev': 64}[s]] for s, r in rows.items()}
            checkpoints = [checkpoints[0], checkpoints[4]]
        fit_episodes = len(rows['train']) - (64 if args.smoke else 2048)
        report['split_counts'] = {s: len(r) for s, r in rows.items()}
        report['fit_episodes'] = fit_episodes
        report['internal_validation_episodes'] = len(rows['train']) - fit_episodes
        report['probe_design'] = {'l2_grid': [1e-4] if args.smoke else list(L2_GRID),
            'selection': 'max train-internal top1; then min CE; then stronger L2; then earlier epoch',
            'softmax_lr': .03, 'address_lr': .003, 'softmax_max_epochs': 4 if args.smoke else 32,
            'L0_max_epochs': 4 if args.smoke else 64, 'address_max_epochs': 2 if args.smoke else 32,
            'fit_rule': 'train by index; final 2048 held out internally in full mode; no refit on internal validation',
            'random_initialization': 'v2_train.adapted seeds torch and CUDA at 0 after Base load; identical projection/reader initialization',
            'label_names': names, 'label_relations': list(RELATIONS)}
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        base = load_base()
        tick = time.monotonic()
        report['hook_verification'] = verify_hooks(base, tok, rows['calibration'][:4], checkpoints)
        report['timings']['hook_verification_s'] = time.monotonic() - tick
        cache = {}
        for split in DATA_COUNTS:
            tick = time.monotonic()
            cache[split] = extract_split(base, tok, rows[split], names, args.batch_size, keep_question=split == 'dev')
            report['timings']['extract_' + split + '_s'] = time.monotonic() - tick
            torch.save(cache[split], out / f'features.{split}.pt')
            save_report(out, report)
            print(f"Extracted {split}: {len(rows[split])} episodes in {report['timings']['extract_' + split + '_s']:.1f}s", flush=True)
        tick = time.monotonic()
        run_linear_layers(cache, fit_episodes, out, report, args.smoke)
        report['timings']['L0_L1_s'] = time.monotonic() - tick
        tick = time.monotonic()
        run_projected(base, cache, checkpoints, fit_episodes, out, report, args.smoke)
        report['timings']['L2_L4_s'] = time.monotonic() - tick
        del base, cache
        gc.collect()
        torch.cuda.empty_cache()
        tick = time.monotonic()
        run_l3(checkpoints, data, out, report, args.smoke, args.l3_batch_size)
        report['timings']['L3_s'] = time.monotonic() - tick
        if (len(report['probes']) != (52 if args.smoke else 120)
                or len(report['l4']) != (6 if args.smoke else 14)
                or len(report['l3']) != (2 if args.smoke else 10)):
            raise AssertionError('measurement table is incomplete')
        success = True
    finally:
        report['elapsed_s'] = time.monotonic() - start
        report['complete'] = success
        save_report(out, report)
        artifacts = [{'path': str(out / 'report.json'), 'role': 'diagnostic_measurements'},
                     {'path': str(out / 'report.md'), 'role': 'diagnostic_tables'}]
        write_json(out / 'manifest.json', {'status': 'complete' if success else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'metrics': {'elapsed_s': report['elapsed_s'], 'l0_pass': report.get('l0_pass', False),
                        'probe_rows': len(report['probes']), 'l3_arms': len(report['l3']), 'l4_arms': len(report['l4'])},
            'artifacts': artifacts,
            'validator': {'name': 'E0012_measurement_completion', 'passed': success}})
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
