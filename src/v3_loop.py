"""E-0018 V3 entry: T2 chain-end data on the isolated-write M line and the S>1 loop.

Execute only in an allocated GPU job. Nothing here changes an existing path:

- ``loop_steps(model, S)`` runs the adapter's own accumulating loop (plan §3.5) with S
  reads per question position: z^(0) = qbar, z^(s+1) = z^(s) + eta * g * a^(s) Ro,
  Z = z^(S) - z^(0), Rq/Ro shared across steps, eta = 0.25, g = 1. The question phase
  never writes, so every step reads the same fixed memory M. S = 1 is the unchanged
  model; the option changes the step count only, never parameters.
- T2 data: E-0011's released phase-V3 train/calibration/dev only (v3_eval.load_released_data
  checks the metadata digest, the shortcut gates and the split digest; sealed slices are
  refused before any file is opened). Each fact is encoded alone in the E=1 format and
  written at its delimiter into slot j (v2_memory.isolated_support, as in E-0016/E-0017).
- ``evaluate_loop`` scores normal/zero/shuffled/permuted at any S with the native forward
  and records, per step, the address weights on the path's j-th hop facts, the argmax slot,
  n_eff, the query cosine to step 1, state and read norms, from an explicit replay of the
  loop on the same memory (its error against the native increment is recorded).
- ``eval``: one D5 adapter on T2 dev. S_eval=1 also runs the reviewed E-0016 evaluator
  (v2r16_memory.evaluate with the isolated write, all six controls), which the hop1 H3
  gate uses, and requires the S=1 loop records to equal it example by example. With
  --t1-check, S=1 is also checked on T1 E=16 dev against the reviewed evaluator (exact)
  and against the source run's recorded predictions (recorded only; D-0018, D-0019).
- ``timing``: about 100 T2 train steps per S from the seed-0 D5 adapter; smoke only.
Confirmation and the OOD slices are never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from functools import partial
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

RUNS = Path('/work/xg24i002/x10041/bypass-lora/runs')
T2_DATA = RUNS/'ATT-0077-1'/'data'
T2_METADATA_SHA256 = '2ad8cc358ec5f6005e832125e3fa9995a0aa0a7c73f77c8fec6da6b7416a178f'
T1_DATA = RUNS/'ATT-0039-1'/'data'
CONTRACT = 'E-0018@1'
# The three D5 adapters of E-0016/E-0017 (seed 0, 1, 2).
D5_RUNS = {'ATT-0159-1': 0, 'ATT-0172-1': 1, 'ATT-0173-1': 2}
T1_CHECK_RUN = 'ATT-0172-1'  # E-0017 P1
LOOP_CONTROLS = ('normal', 'zero', 'shuffled', 'permuted')
S_EVAL = (1, 2, 4, 8)
TIMING_S = (1, 2, 4)
MAX_HOP = 3
# Same batching and donor rule as v2r16_memory.evaluate, so S=1 is comparable example by example.
EVAL_BATCH = 16
# Per-example fields that the S=1 loop must reproduce from the reviewed evaluator.
# The explicit replay must reproduce the native increment for its per-step diagnostics to
# describe the evaluated forward. The native reader scores each question position on its
# own ([rows, r] einsums), the replay all positions at once ([B, T, r], the reviewed
# read_vectors), so score rounding differs and the softmax passes it on to the read
# (ATT-0177-1/ATT-0178-1: about 5e-4 absolute on states of about 1e2 at S=1). The bound is
# on ||Z_replay - Z_native|| / ||Z_native|| per valid question position (0.1%); absolute
# and state-relative maxima are recorded alongside.
RECONSTRUCTION_TOLERANCE = 1e-3
REVIEWED_FIELDS = ('prediction', 'correct', 'nll', 'all_target_weight', 'last_target_weight',
                   'all_n_eff', 'last_n_eff', 'last_argmax')


@contextmanager
def loop_steps(model, steps):
    """S accumulating reads per question position with the model's own reader (plan §3.5).

    Only the step count of the single-layer SoftmaxSlot increment readout changes; the
    configuration objects are restored on exit, so S=1 and the exited state are master.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError('loop steps must be a positive integer')
    cfg = model.config
    if (cfg.steps != 1 or cfg.update != 'accumulate' or cfg.readout != 'increment' or cfg.layers != (14,)
            or cfg.backend != 'soft_slot' or cfg.branch != 'memory'):
        raise ValueError('the S>1 option extends the single-read layer-14 SoftmaxSlot increment model')
    sides = {i: layer.side for i, layer in model.adapters().items()}
    originals = {i: side.config for i, side in sides.items()}
    if any(c != cfg for c in originals.values()):
        raise ValueError('side and model configurations differ')
    try:
        model.config = replace(cfg, steps=steps)
        for side in sides.values():
            side.config = model.config
        yield model
    finally:
        model.config = cfg
        for i, side in sides.items():
            side.config = originals[i]


# ---- T2 data -----------------------------------------------------------------------------

def check_path(row):
    """The path's j-th hop fact is facts[path.fact_indices[j]]; hop 1 is the query target."""
    entities, indices = row['path']['entities'], row['path']['fact_indices']
    hop = row['axes']['hop']
    if not 1 <= hop <= MAX_HOP or len(indices) != hop or len(entities) != hop+1 or len(set(indices)) != hop:
        raise ValueError('path does not match the hop')
    if row['query']['target_fact_index'] != indices[0] or row['query']['head'] != entities[0]:
        raise ValueError('the query target is not the first path fact')
    for j, index in enumerate(indices):
        fact = row['facts'][index]
        if (fact['head'], fact['relation'], fact['tail']) != (entities[j], row['query']['relation'], entities[j+1]):
            raise ValueError('path fact does not join the path entities')
    if row['query']['answer_text'] != ' '+entities[-1]:
        raise ValueError('the answer is not the chain end')


def load_t2(tok, split, directory=T2_DATA):
    """Released in-distribution T2 rows with the isolated encoding annotated."""
    from v2_isolated import annotate_isolated
    from v3_eval import load_released_data
    if split not in ('train', 'calibration', 'dev'):
        raise ValueError('sealed slices are never opened')
    metadata, rows = load_released_data(directory, T2_METADATA_SHA256, split)
    for row in rows:
        check_path(row)
    return metadata, rows, annotate_isolated(tok, rows)


def path_matrix(rows, device):
    """[B, 3] support indices of each row's path facts (T1: the target), -1 beyond its hop."""
    index = torch.full((len(rows), MAX_HOP), -1, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        facts = row['path']['fact_indices'] if 'path' in row else [row['query']['target_fact_index']]
        index[i, :len(facts)] = torch.tensor(facts, device=device)
    return index


def hop_of(row):
    return row['axes']['hop'] if 'axes' in row else 1


# ---- loop replay and evaluation ------------------------------------------------------------

def loop_trajectory(qbar, snapshot, reader, steps):
    """Explicit replay of the accumulating loop on a compact snapshot (diagnostics only).

    Returns the increment z^(S) - z^(0) and, per step s, the state z^(s), the query, the
    softmax address weights and the used read g * a_raw^(s). The read is
    v2_memory.read_vectors, so step 1 is the reviewed S=1 diagnostic exactly.
    """
    from bypass_core import elu_plus_one, rmsnorm_no_affine
    from v2_memory import read_vectors
    target = torch.zeros(len(qbar), dtype=torch.long, device=qbar.device)
    z0 = qbar
    z = z0
    trajectory = []
    for _ in range(steps):
        raw, weights, _, _ = read_vectors(z, snapshot, reader, 'soft_slot', target, 'normal')
        query = elu_plus_one(rmsnorm_no_affine(z, reader.eps_norm) @ reader.Rq)
        used = raw * reader.read_gain
        trajectory.append({'state': z, 'query': query, 'weights': weights, 'read': used})
        z = z + reader.eta * (used @ reader.Ro)
    return z - z0, trajectory, z


def _snapshots(model, batch, mode, donor_batch):
    """The reviewed E-0016 control memories under the isolated write."""
    from v2_memory import control_snapshot, isolated_support
    from v2r16_memory import compact_snapshot
    original, _ = isolated_support(model, batch)
    original = compact_snapshot(original).select(batch.owners)
    if mode == 'shuffled':
        if donor_batch is None or len(donor_batch.rows) != len(batch.rows):
            raise ValueError('shuffled requires one explicit global donor per query')
        if any(a['episode_id'] == b['episode_id'] for a, b in zip(batch.rows, donor_batch.rows)):
            raise ValueError('donor must differ from recipient')
        donor, _ = isolated_support(model, donor_batch)
        return compact_snapshot(donor).select(donor_batch.owners)
    return control_snapshot(original, mode)


@torch.no_grad()
def evaluate_loop_batch(model, batch, mode, donor_batch, steps):
    """Native S-step question forward on the fixed control memory, plus per-step diagnostics."""
    from v2_memory import answer_logits, capture_projection
    if model.training or mode not in LOOP_CONTROLS:
        raise ValueError('evaluation needs eval mode and a loop control')
    snapshot = _snapshots(model, batch, mode, donor_batch)
    side = model.adapters()[14].side
    mask = batch.question_mask
    captured = {}

    def observe(module, args, output):
        captured['z'], captured['state'], captured['accounting'] = output

    with loop_steps(model, steps), capture_projection(side) as seen:
        handle = side.register_forward_hook(observe)
        try:
            output = model(batch.question_ids, mask, memory=snapshot.memory, phase='query')
        finally:
            handle.remove()
        qbar = seen['q']
    logits = answer_logits(output, mask)
    if not torch.isfinite(logits).all():
        raise RuntimeError('nonfinite full-vocabulary evaluation logits')
    # The question phase reads S times per valid token and never writes: M is fixed.
    state, memory = captured['state'], snapshot.memory[14]
    if not all(torch.equal(getattr(state, f), getattr(memory, f))
               for f in ('M', 'c', 'counts', 'slot_keys', 'slot_values', 'slot_valid')):
        raise AssertionError('the question phase changed the memory')
    counts = output.accounting[14]['side']
    if counts['writes'] != 0 or counts['reads'] != steps*int(mask.sum()):
        raise AssertionError('the loop must read S times per valid question token and never write')
    increment, trajectory, _ = loop_trajectory(qbar, snapshot, side.reader, steps)
    native = captured['z']
    if not (torch.isfinite(native).all() and torch.isfinite(increment).all()):
        raise RuntimeError('nonfinite loop state')
    difference = increment - native
    error = difference.abs().amax(-1)
    scale = torch.maximum(qbar.abs().amax(-1), (qbar + native).abs().amax(-1)).clamp_min(1e-30)
    # Empty memory gives Z = 0 exactly in both, hence a zero ratio.
    relative = difference.norm(dim=-1)/native.norm(dim=-1).clamp_min(1e-6)
    reconstruction = {'max_relative_to_increment': float(relative[mask].max()), 'max_abs': float(error[mask].max()),
                      'max_relative_to_state': float((error/scale)[mask].max()),
                      'state_max_abs': float(scale[mask].max())}
    if not reconstruction['max_relative_to_increment'] <= RECONSTRUCTION_TOLERANCE:
        raise AssertionError(f'loop replay differs from the native increment: {reconstruction}')
    losses = F.cross_entropy(logits, batch.labels, reduction='none')
    row = torch.arange(len(batch.labels), device=mask.device)
    last = mask.long().sum(-1)-1
    counts = mask.sum(-1)
    has_slots = snapshot.valid.any(-1)
    paths = path_matrix(batch.rows, mask.device)
    first_query = trajectory[0]['query']
    target_available = mode != 'shuffled'  # recipient path facts are undefined in donor memory
    per_step = []
    for step in trajectory:
        weights = step['weights']
        # The reviewed evaluator's expressions (v2r16_memory.evaluate_batch), per path fact.
        neff = torch.where(weights.sum(-1) > 0, 1/weights.square().sum(-1).clamp_min(1e-30), 0)
        path_all, path_last = [], []
        for j in range(MAX_HOP):
            fact = paths[:, j].clamp_min(0)
            weight = weights.gather(-1, fact[:, None, None].expand(-1, weights.shape[1], 1)).squeeze(-1)
            path_all.append((weight*mask).sum(-1)/counts)
            path_last.append(weight[row, last])
        cosine = F.cosine_similarity(step['query'], first_query, dim=-1, eps=1e-6)
        state_norm, read_norm = step['state'].norm(dim=-1), step['read'].norm(dim=-1)
        per_step.append({key: value.tolist() for key, value in {
            'path_weight_all': torch.stack(path_all, -1), 'path_weight_last': torch.stack(path_last, -1),
            'n_eff_all': (neff*mask).sum(-1)/counts, 'n_eff_last': neff[row, last],
            'argmax_last': torch.where(has_slots, weights[row, last].argmax(-1), -1),
            'query_cosine_all': (cosine*mask).sum(-1)/counts, 'query_cosine_last': cosine[row, last],
            'state_norm_all': (state_norm*mask).sum(-1)/counts, 'state_norm_last': state_norm[row, last],
            'read_norm_all': (read_norm*mask).sum(-1)/counts, 'read_norm_last': read_norm[row, last]}.items()})
    final_state = (qbar + native).norm(dim=-1)[row, last].tolist()
    native_norm = native.norm(dim=-1)
    increment_last = native_norm[row, last].tolist()
    increment_all = ((native_norm*mask).sum(-1)/counts).tolist()
    predictions, labels, losses = logits.argmax(-1).tolist(), batch.labels.tolist(), losses.tolist()
    records = []
    for i, episode in enumerate(batch.rows):
        hop = hop_of(episode)
        record = {'episode_id': episode['episode_id'], 'hop': hop,
                  'support_order': episode.get('axes', {}).get('support_order'),
                  'target': labels[i], 'prediction': predictions[i], 'correct': predictions[i] == labels[i],
                  'nll': losses[i], 'question_tokens': int(counts[i]), 'steps': steps,
                  'final_state_norm_last': final_state[i], 'increment_norm_last': increment_last[i],
                  'increment_norm_all': increment_all[i]}
        for name in ('n_eff_all', 'n_eff_last', 'argmax_last', 'query_cosine_all', 'query_cosine_last',
                     'state_norm_all', 'state_norm_last', 'read_norm_all', 'read_norm_last'):
            record[name] = [s[name][i] for s in per_step]
        for name in ('path_weight_all', 'path_weight_last'):
            record[name] = [s[name][i][:hop] for s in per_step] if target_available else None
        records.append(record)
    return records, reconstruction


@torch.no_grad()
def evaluate_loop(model, rows, pad_id, steps, controls=LOOP_CONTROLS):
    """Per-example loop records over rows in batches of 16; shuffled uses the reviewed global
    donor rule (recipient i receives support (i-1) modulo the split size)."""
    from v2_memory import collate
    model.eval()
    if len(rows) < 2:
        raise ValueError('at least two rows are required')
    device = next(model.parameters()).device
    result, reconstruction = {}, {}
    for mode in controls:
        records = []
        worst = {'max_relative_to_increment': 0., 'max_abs': 0., 'max_relative_to_state': 0., 'state_max_abs': 0.}
        for start in range(0, len(rows), EVAL_BATCH):
            selected = rows[start:start+EVAL_BATCH]
            batch = collate(selected, pad_id, device)
            donors = None
            if mode == 'shuffled':
                donors = collate([rows[(i-1) % len(rows)] for i in range(start, start+len(selected))], pad_id, device)
            batch_records, error = evaluate_loop_batch(model, batch, mode, donors, steps)
            records.extend(batch_records)
            worst = {k: max(worst[k], error[k]) for k in worst}
        result[mode], reconstruction[mode] = records, worst
    return result, reconstruction


def _mean(values):
    values = list(values)
    return sum(values)/len(values) if values else None


def summarize_loop(records, prior_token=None):
    """Per-hop (and all, and hop x support order) EM/NLL and per-step diagnostics."""
    groups = {'all': records}
    for hop in range(1, MAX_HOP+1):
        groups[f'hop{hop}'] = [r for r in records if r['hop'] == hop]
        for order in ('natural', 'shuffled'):
            groups[f'hop{hop}.{order}'] = [r for r in records if r['hop'] == hop and r['support_order'] == order]
    summaries = {}
    for name, group in groups.items():
        if not group:
            continue
        steps = group[0]['steps']
        tokens = sum(r['question_tokens'] for r in group)
        summary = {'count': len(group), 'correct_count': sum(r['correct'] for r in group),
                   'em': sum(r['correct'] for r in group)/len(group), 'nll': _mean(r['nll'] for r in group)}
        if prior_token is not None:
            summary['prior_em'] = sum(r['target'] == prior_token for r in group)/len(group)
        if '.' not in name:
            hops = {r['hop'] for r in group}
            width = max(hops)
            per_step = []
            for s in range(steps):
                valid_argmax = [r for r in group if r['argmax_last'][s] >= 0]
                step = {
                    'n_eff_all': sum(r['n_eff_all'][s]*r['question_tokens'] for r in group)/tokens,
                    'n_eff_last': _mean(r['n_eff_last'][s] for r in group),
                    'query_cosine_all': sum(r['query_cosine_all'][s]*r['question_tokens'] for r in group)/tokens,
                    'query_cosine_last': _mean(r['query_cosine_last'][s] for r in group),
                    'state_norm_last': _mean(r['state_norm_last'][s] for r in group),
                    'read_norm_last': _mean(r['read_norm_last'][s] for r in group),
                    'read_norm_all': sum(r['read_norm_all'][s]*r['question_tokens'] for r in group)/tokens,
                    'argmax_slot_counts': [sum(r['argmax_last'][s] == k for r in valid_argmax) for k in range(16)],
                }
                if all(r['path_weight_all'] is not None for r in group):
                    # Weight on the path's j-th hop fact; for 'all', only rows whose hop reaches j.
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
                    # Fraction of rows whose last-position argmax is the path's j-th hop fact.
                    step['argmax_is_path_fact'] = [
                        _mean(r['argmax_last'][s] == r['_path'][j] for r in group if r['hop'] > j)
                        if all('_path' in r for r in group) else None for j in range(width)]
                per_step.append(step)
            summary['per_step'] = per_step
            summary['final_state_norm_last'] = _mean(r['final_state_norm_last'] for r in group)
            summary['increment_norm_last'] = _mean(r['increment_norm_last'] for r in group)
            summary['increment_norm_all'] = sum(r['increment_norm_all']*r['question_tokens'] for r in group)/tokens
        summaries[name] = summary
    return summaries


def attach_paths(records, rows):
    by_id = {r['episode_id']: r for r in rows}
    for record in records:
        row = by_id[record['episode_id']]
        record['_path'] = row['path']['fact_indices'] if 'path' in row else [row['query']['target_fact_index']]
    return records


def compare_to_reviewed(loop_records, reviewed_records):
    """S=1 loop records versus the reviewed E-0016 evaluator's records of the same control.

    Counts examples whose fields differ in any way (exact comparison), per field.
    """
    if [r['episode_id'] for r in loop_records] != [r['episode_id'] for r in reviewed_records]:
        raise AssertionError('loop and reviewed records cover different episodes or orders')
    differences = Counter()
    for mine, theirs in zip(loop_records, reviewed_records):
        expected = {'prediction': theirs['prediction'], 'correct': theirs['correct'], 'nll': theirs['nll'],
                    'all_target_weight': theirs['all_target_weight'], 'last_target_weight': theirs['last_target_weight'],
                    'all_n_eff': theirs['all_n_eff'], 'last_n_eff': theirs['last_n_eff'],
                    'last_argmax': theirs['last_argmax']}
        actual = {'prediction': mine['prediction'], 'correct': mine['correct'], 'nll': mine['nll'],
                  'all_target_weight': None if mine['path_weight_all'] is None else mine['path_weight_all'][0][0],
                  'last_target_weight': None if mine['path_weight_last'] is None else mine['path_weight_last'][0][0],
                  'all_n_eff': mine['n_eff_all'][0], 'last_n_eff': mine['n_eff_last'][0],
                  'last_argmax': mine['argmax_last'][0]}
        for name in REVIEWED_FIELDS:
            if actual[name] != expected[name]:
                differences[name] += 1
    return {'episodes': len(loop_records), 'differing_examples_by_field': dict(differences),
            'identical': not differences}


def compare_to_recorded(loop_records, path):
    """S=1 loop records versus a run's recorded per-example predictions (record only)."""
    recorded = [json.loads(line) for line in Path(path).open()]
    if [r['episode_id'] for r in recorded] != [r['episode_id'] for r in loop_records]:
        raise AssertionError('recorded predictions cover different episodes or orders')
    mismatches = sum(a['prediction'] != b['prediction'] for a, b in zip(loop_records, recorded))
    nll = max(abs(a['nll']-b['nll']) for a, b in zip(loop_records, recorded))
    return {'recorded': str(path), 'episodes': len(recorded), 'same_episode_order': True,
            'prediction_mismatches': mismatches, 'nll_max_abs_difference': nll,
            'nll_identical': all(a['nll'] == b['nll'] for a, b in zip(loop_records, recorded)),
            'numerical_differences': 'recorded only (D-0018, D-0019)'}


def reviewed_by_hop(reviewed, rows, prior_token):
    """The reviewed S=1 records grouped by hop, in v2r16_memory.evaluate's summary form."""
    hops = {r['episode_id']: hop_of(r) for r in rows}
    summaries = {}
    for mode, records in reviewed.items():
        for hop in range(1, MAX_HOP+1):
            group = [r for r in records if hops[r['episode_id']] == hop]
            if not group:
                continue
            tokens = sum(r['question_tokens'] for r in group)
            summary = {'count': len(group), 'correct_count': sum(r['correct'] for r in group),
                       'em': sum(r['correct'] for r in group)/len(group), 'nll': _mean(r['nll'] for r in group),
                       'prior_em': sum(r['target'] == prior_token for r in group)/len(group),
                       'diagnostics': {}, 'last_position': {}}
            if all(r['all_target_weight'] is not None for r in group):
                summary['diagnostics']['address.target_weight'] = sum(
                    r['all_target_weight']*r['question_tokens'] for r in group)/tokens
                summary['last_position']['target_weight_mean'] = _mean(r['last_target_weight'] for r in group)
            summary['diagnostics']['address.n_eff'] = sum(r['all_n_eff']*r['question_tokens'] for r in group)/tokens
            summaries.setdefault(f'hop{hop}', {})[mode] = summary
    return summaries


def hop1_gate(by_hop):
    """E-0010@2 H3 (isolated-write protocol) on hop1 at S_eval=1: the reviewed exact gate."""
    from v2r16_isolated import gate
    return gate({m: by_hop['hop1'][m] for m in ('normal', 'zero', 'shuffled')})


# ---- adapters --------------------------------------------------------------------------------

def adapter_provenance(run, expected_sha256):
    """A completed formal D5 run (E-0016/E-0017) whose adapter.pt has the pinned digest."""
    from v2r16_train import checkpoint_records, file_record
    if not (isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(c in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('an expected lowercase hex sha256 of adapter.pt is required')
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True
            or manifest.get('arm') != 'D5' or manifest.get('eligible_as_evidence') is not True
            or manifest.get('smoke_steps') or manifest.get('write_protocol') != 'isolated'):
        raise ValueError('the adapter must come from a completed formal isolated-write D5 run')
    files = checkpoint_records(run/'adapter')
    if files != manifest.get('checkpoint_files'):
        raise ValueError('adapter files differ from the run manifest')
    if files['adapter.pt']['sha256'] != expected_sha256:
        raise ValueError('adapter.pt differs from the pinned digest')
    return {'run': str(run.resolve()), 'checkpoint_files': files, 'expected_adapter_sha256': expected_sha256,
            'source_manifest': file_record(run/'manifest.json'), 'experiment': manifest.get('experiment'),
            'seed': manifest.get('seed'), 'start_seed': manifest.get('start_seed')}


# ---- runners -----------------------------------------------------------------------------------

def _write_records(path, records):
    with Path(path).open('w') as handle:
        for record in records:
            handle.write(json.dumps({k: v for k, v in record.items() if not k.startswith('_')}, allow_nan=False)+'\n')


def run_eval(args, output, result):
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION, load_base
    from v2_isolated import annotate_isolated
    from v2_memory import isolated_support
    from v2_schedule import load_data
    from v2_train import adapted, write_json
    from v2r16_memory import evaluate as reviewed_evaluate
    from v2r16_train import STANDARD, checkpoint_records

    provenance = adapter_provenance(args.adapter_run, args.adapter_sha256)
    result['adapter'] = provenance
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tick = time.monotonic()
    metadata, train_rows, train_differing = load_t2(tokenizer, 'train')
    _, dev, dev_differing = load_t2(tokenizer, 'dev')
    if args.eval_limit:
        dev = dev[:args.eval_limit]
    prior = Counter(r['query']['answer_token_id'] for r in train_rows).most_common(1)[0][0]
    del train_rows
    result['data'] = {'directory': str(T2_DATA), 'metadata_sha256': T2_METADATA_SHA256,
                      'dev_sha256': metadata['splits']['dev']['sha256'], 'dev_rows': len(dev),
                      'context_fact_tokens_differ_from_isolated': {'train': train_differing, 'dev': dev_differing},
                      'prior_token': prior, 'seconds': time.monotonic()-tick,
                      'baselines': {'shortcuts_dev': metadata['shortcuts']['dev']}}
    t1 = None
    if args.t1_check:
        t1 = load_data(T1_DATA/'dev.jsonl', 'dev')
        if args.eval_limit:
            t1 = t1[:args.eval_limit]
        result['t1_check'] = {'context_fact_tokens_differ_from_isolated': annotate_isolated(tokenizer, t1)}
    base = load_base()
    with adapted(base, checkpoint=Path(args.adapter_run)/'adapter') as model:
        if checkpoint_records(Path(args.adapter_run)/'adapter') != provenance['checkpoint_files']:
            raise ValueError('adapter files changed between the digest check and loading')
        result['config'] = asdict(model.config)
        model.eval()
        # Reviewed S_eval=1 path: all six controls, the hop1 H3 gate.
        torch.cuda.synchronize()
        tick = time.monotonic()
        reviewed_summary, reviewed = reviewed_evaluate(model, dev, pad_id, output/'reviewed-s1', 'dev', STANDARD,
                                                       prior, support_fn=isolated_support)
        by_hop = reviewed_by_hop(reviewed, dev, prior)
        result['reviewed_s1'] = {'summary': reviewed_summary, 'by_hop': by_hop,
                                 'seconds': time.monotonic()-tick}
        result['hop1_gate'] = hop1_gate(by_hop)
        result['loop'], result['reconstruction'], result['loop_seconds'] = {}, {}, {}
        for steps in args.s_eval:
            torch.cuda.synchronize()
            tick = time.monotonic()
            records, reconstruction = evaluate_loop(model, dev, pad_id, steps)
            torch.cuda.synchronize()
            result['loop_seconds'][str(steps)] = time.monotonic()-tick
            directory = output/f'loop-s{steps}'
            directory.mkdir()
            result['loop'][str(steps)] = {}
            for mode, mode_records in records.items():
                attach_paths(mode_records, dev)
                result['loop'][str(steps)][mode] = summarize_loop(mode_records, prior)
                _write_records(directory/f'dev.{mode}.jsonl', mode_records)
            result['reconstruction'][str(steps)] = reconstruction
            if steps == 1:
                result['s1_vs_reviewed'] = {mode: compare_to_reviewed(records[mode], reviewed[mode])
                                            for mode in LOOP_CONTROLS}
                if not all(c['identical'] for c in result['s1_vs_reviewed'].values()):
                    raise AssertionError('S=1 loop differs from the reviewed evaluator: implementation check failed')
        if t1 is not None:
            tick = time.monotonic()
            _, t1_reviewed = reviewed_evaluate(model, t1, pad_id, output/'t1-reviewed-s1', 'dev', LOOP_CONTROLS,
                                               None, support_fn=isolated_support)
            t1_loop, t1_reconstruction = evaluate_loop(model, t1, pad_id, 1)
            recorded = Path(args.adapter_run)/'evaluation'/'isolated'
            result['t1_check'].update({
                'rows': len(t1), 'reconstruction': t1_reconstruction,
                'vs_reviewed_same_job': {m: compare_to_reviewed(t1_loop[m], t1_reviewed[m]) for m in LOOP_CONTROLS},
                'vs_recorded': {m: compare_to_recorded(t1_loop[m], recorded/f'dev.{m}.predictions.jsonl')
                                for m in LOOP_CONTROLS} if not args.eval_limit else None,
                'seconds': time.monotonic()-tick})
            if not all(c['identical'] for c in result['t1_check']['vs_reviewed_same_job'].values()):
                raise AssertionError('S=1 loop differs from the reviewed evaluator on T1: implementation check failed')
    write_json(output/'summary.json', {k: result[k] for k in ('hop1_gate', 'reconstruction')})


def run_timing(args, output, result):
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION, load_base
    from v2_memory import isolated_support
    from v2_schedule import make_step_schedule
    from v2_train import adapted, train
    from v2r16_memory import train_forward
    from v2r16_train import checkpoint_records

    provenance = adapter_provenance(args.adapter_run, args.adapter_sha256)
    result['adapter'] = provenance
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    _, train_rows, differing = load_t2(tokenizer, 'train')
    schedule = make_step_schedule(train_rows, args.timing_steps, phase='V3')
    result['schedule'] = {k: v for k, v in schedule.items() if k not in ('order', 'batches')}
    result['context_fact_tokens_differ_from_isolated'] = differing
    base = load_base()
    forward = partial(train_forward, support_fn=isolated_support)
    result['timing'] = {}
    for steps in args.timing_s:
        directory = output/f'train-s{steps}'
        directory.mkdir()
        with adapted(base, checkpoint=Path(args.adapter_run)/'adapter') as model:
            if checkpoint_records(Path(args.adapter_run)/'adapter') != provenance['checkpoint_files']:
                raise ValueError('adapter files changed between the digest check and loading')
            for parameter in model.adapter_parameters():
                parameter.requires_grad_(True)
            with loop_steps(model, steps):
                torch.cuda.synchronize()
                tick = time.monotonic()
                record = train(model, train_rows, schedule, pad_id, .003, 8, directory, None, None,
                               forward_fn=forward, learning_curves=False)
                torch.cuda.synchronize()
            losses = [json.loads(line)['loss'] for line in (directory/'train.jsonl').open()]
            record.update(wall_seconds=time.monotonic()-tick, first_loss=losses[0], last_loss=losses[-1],
                          tokens_per_step=record['tokens']/record['steps'])
            result['timing'][str(steps)] = record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('eval', 'timing'))
    parser.add_argument('--adapter-run', type=Path, required=True)
    parser.add_argument('--adapter-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--s-eval', default='1,2,4,8')
    parser.add_argument('--t1-check', action='store_true', help='S=1 against T1 E=16 dev (P1 adapter)')
    parser.add_argument('--eval-limit', type=int, default=0, help='smoke only: first N dev rows')
    parser.add_argument('--timing-steps', type=int, default=100)
    parser.add_argument('--timing-s', default='1,2,4')
    args = parser.parse_args()
    try:
        args.s_eval = tuple(int(x) for x in args.s_eval.split(','))
        args.timing_s = tuple(int(x) for x in args.timing_s.split(','))
    except ValueError:
        parser.error('S lists are comma-separated integers')
    if args.command == 'eval' and (not args.s_eval or args.s_eval[0] != 1 or not set(args.s_eval) <= set(S_EVAL)
                                   or list(args.s_eval) != sorted(set(args.s_eval))):
        parser.error('S_eval is an increasing subset of 1,2,4,8 starting at 1')
    if args.command == 'timing' and (not set(args.timing_s) <= set(TIMING_S) or not 1 <= args.timing_steps <= 200):
        parser.error('timing runs S in 1,2,4 for at most 200 steps')
    if args.adapter_run.parent.name not in D5_RUNS:
        parser.error('the adapter must be one of the three D5 runs')
    if args.t1_check and args.adapter_run.parent.name != T1_CHECK_RUN:
        parser.error('the T1 check compares with P1 (ATT-0172-1)')
    if args.eval_limit and args.eval_limit < 2:
        parser.error('eval limit must be >= 2')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    from calibrate import MODEL, REVISION
    from v2_train import write_json
    from v2r16_train import file_record
    started = time.monotonic()
    result, passed, failure = {'experiment': CONTRACT, 'command': args.command,
                               'smoke_only': args.command == 'timing' or bool(args.eval_limit)}, False, None
    try:
        (run_eval if args.command == 'eval' else run_timing)(args, output, result)
        passed = True
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        wall = time.monotonic()-started
        result.update(wall_seconds=wall, mig_instance_hours=wall/3600, node_hours=wall/14400,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        write_json(output/'metrics.json', result)
        artifacts = [{'path': str(p.resolve()), 'role': str(p.relative_to(output)), **file_record(p)}
                     for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']
        write_json(output/'manifest.json', {
            'status': 'complete' if passed else 'failed', 'experiment': 'E-0018', 'contract': CONTRACT,
            'command': args.command, 'arm': f'D5@{args.adapter_run.parent.name}', 'write_protocol': 'isolated',
            'spec_digest': os.environ.get('SR_SPEC_DIGEST'), 'model': MODEL, 'revision': REVISION,
            's_eval': list(args.s_eval) if args.command == 'eval' else None,
            'timing_s': list(args.timing_s) if args.command == 'timing' else None,
            'eval_limit': args.eval_limit, 'smoke_only': result['smoke_only'],
            'eligible_as_evidence': passed and not result['smoke_only'],
            'adapter': result.get('adapter'), 'data': result.get('data'),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'hardware': ({'device': torch.cuda.get_device_name(0),
                          'multiprocessors': torch.cuda.get_device_properties(0).multi_processor_count}
                         if torch.cuda.is_available() else None),
            'metrics': {'wall_seconds': wall, 'mig_instance_hours': wall/3600, 'node_hours': wall/14400,
                        **({'hop1_gate': result['hop1_gate']} if 'hop1_gate' in result else {})},
            'artifacts': artifacts, 'failure': failure,
            'validator': {'name': 'v3_loop', 'passed': passed}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
