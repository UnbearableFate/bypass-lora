"""E-0023 rule 4 repair (D-0035): successor-slot CE S4 pilot and its permuted-successor-label
control. Execute only in an allocated GPU job. T2 hops 4/6/8 and confirmation are never opened.

Both arms (``--labels correct`` / ``--labels permuted``) start from the readiness-checked common
start ATT-0196-1 (readiness ATT-0227-1) in the V3-03 configuration (S = 4) and train 2,000 steps
on the first 16,000 episodes of the stage-A order (``v3_schedule(train, 'train', steps=2000)``:
5% warmup, LR 3e-3, the shared schedule shape over 2,000 steps). Init and order are identical.

Loss = answer CE + 1 x successor CE. On a training row of hop h, at the answer position (last
valid question token), the read of step s = 2..min(4, h) is labelled with path.fact_indices[s-1];
step 1 and steps beyond h are unlabelled (hop 1: 0 labels, hop 2: 1, hop 3: 2). The successor CE
is the mean over the batch's labelled (row, step) pairs of -log_softmax(slot logits)[label], on
the SoftmaxSlot logits of the actual read (the answer-position loop is replayed with gradients
and checked against the native side output every step); a batch without labels adds 0. The
permuted control replaces each label by pi(label), pi a seeded derangement of the episode's
slots (fixed per episode), so label count and positions are unchanged.

Evaluation: T2 dev at S = 4 with normal/zero/shuffled/permuted reads and the per-step path
weights (``v3_train.evaluate_arm``), then save, reload and require identical records.
``compare`` applies D-0035 on dev hops 2-3 with exact fractions: correct beats permuted only if
(i) primary EM (EM_hop2 + EM_hop3)/2, correct minus permuted, >= +3 pp and (ii) the hop-2 mean
answer-position step-2 weight on path fact 2 is >= 1/8 for correct and < 1/8 for permuted.
Otherwise the repair route stops. Extra supervision: screening evidence, not unsupervised H-0005
evidence. Hop 1 and V3-03 are descriptive.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

import v3_train as v3
from v1_data import derive_seed

ARM = 'V3-03'  # S4 configuration (E-0023 rule 4, D-0035)
PILOT_STEPS = 2000
COEFFICIENT = 1.
LABELS = ('correct', 'permuted')
EFFECT_PP = Fraction(3)  # D-0035 (i)
ADDRESS = Fraction(1, 8)  # D-0035 (ii)
PERMUTATION_SEED = ('V3', 'succ', 'permuted-successor', 0)


# ---- labels ---------------------------------------------------------------------------------------

def derangement(episode_id, n):
    """Seeded uniform derangement of range(n) for one episode (rejection sampling)."""
    if n < 2:
        raise ValueError('a derangement needs at least two slots')
    rng = random.Random(derive_seed(v3.SEED, *PERMUTATION_SEED, episode_id))
    while True:
        order = list(range(n))
        rng.shuffle(order)
        if all(order[i] != i for i in range(n)):
            return order


def successor_labels(row, labels, steps=4):
    """[(step index 0-based, slot)] for steps s = 2..min(steps, hop); permuted maps slot -> pi(slot)."""
    if labels not in LABELS:
        raise ValueError('labels is correct or permuted')
    path = row['path']['fact_indices']
    pairs = [(s-1, path[s-1]) for s in range(2, min(steps, len(path))+1)]
    if labels == 'permuted' and pairs:
        pi = derangement(row['episode_id'], len(row['facts']))
        pairs = [(s, pi[slot]) for s, slot in pairs]
    return pairs


def label_counts(rows, schedule, labels):
    used = [rows[i] for batch in schedule['batches'] for i in batch]
    per_hop = {}
    for r in used:
        hop = len(r['path']['fact_indices'])
        per_hop[str(hop)] = per_hop.get(str(hop), 0) + len(successor_labels(r, labels))
    return {'total': sum(per_hop.values()), 'by_hop': dict(sorted(per_hop.items())), 'episodes': len(used)}


# ---- training forward -----------------------------------------------------------------------------

def slot_logits(query, snapshot):
    """v3_train.slot_read's logits (its softmax is the read weight); invalid slots are -inf."""
    scores = torch.einsum('btr,bnr->btn', query, snapshot.keys) / query.shape[-1] ** .5
    return scores.masked_fill(~snapshot.valid[:, None].expand_as(scores), -torch.inf)


def succ_forward(model, batch, *, labels, tally):
    """train_forward's standard path under the isolated write, plus the successor CE."""
    from v2_memory import answer_logits, capture_projection, isolated_support
    from v2r16_memory import compact_snapshot
    from v3_loop import RECONSTRUCTION_TOLERANCE
    side = model.adapters()[14].side
    if v3.side_variant(side) != v3.Variant():
        raise ValueError('the successor CE is defined for the shared slot-reading loop (V3-03 configuration)')
    steps = side.config.steps
    support, support_counts = isolated_support(model, batch)
    selected = support.select(batch.owners)
    captured = {}

    def observe(module, args, output):
        captured['z'] = output[0]

    with capture_projection(side) as seen:
        handle = side.register_forward_hook(observe)
        try:
            output = model(batch.question_ids, batch.question_mask, memory=selected.memory, phase='query')
        finally:
            handle.remove()
    logits = answer_logits(output, batch.question_mask)
    answer = F.cross_entropy(logits, batch.labels)
    mask = batch.question_mask
    row = torch.arange(len(batch.rows), device=mask.device)
    last = mask.long().sum(-1)-1
    snapshot = compact_snapshot(selected)
    qbar, kbar, vbar = (seen[name][row, last][:, None] for name in ('q', 'k', 'v'))
    z, trajectory = v3.replay(side, qbar, kbar, vbar, snapshot, steps)
    native = captured['z'][row, last][:, None]
    relative = float(((z-native).norm(dim=-1)/native.norm(dim=-1).clamp_min(1e-6)).max())
    if not relative <= RECONSTRUCTION_TOLERANCE:
        raise AssertionError(f'answer-position replay differs from the native output: {relative}')
    pairs = [(i, s, slot) for i, r in enumerate(batch.rows) for s, slot in successor_labels(r, labels, steps)]
    if pairs:
        index = torch.tensor(pairs, device=mask.device)
        log_probs = torch.stack([F.log_softmax(slot_logits(t['query'], snapshot)[:, 0], -1) for t in trajectory])
        successor = -log_probs[index[:, 1], index[:, 0], index[:, 2]].mean()
    else:
        successor = answer.new_zeros(())
    tally.append({'answer': float(answer.detach()), 'successor': float(successor.detach()), 'labels': len(pairs),
                  'replay_max_relative': relative})
    return answer + COEFFICIENT*successor, logits, support_counts, output.accounting


# ---- pilot run ------------------------------------------------------------------------------------

def run_pilot(args, output, result):
    from calibrate import load_base
    from v2_train import train
    from v2r16_train import checkpoint_records
    from v3_loop import attach_paths
    provenance = v3.start_provenance(args.start_run, args.start_sha256, smoke=bool(args.smoke_steps),
                                     ready_run=args.ready_run)
    result['start'] = provenance
    start_directory = Path(args.start_run)/'adapter'
    tokenizer, pad_id = v3._tokenizer()
    train_rows, dev, prior, result['data'] = v3._data(tokenizer, args.eval_limit)
    schedule = v3.v3_schedule(train_rows, 'train', steps=PILOT_STEPS)
    if args.smoke_steps:
        schedule = v3.trim_smoke(schedule, train_rows, args.smoke_steps)
    result['schedule'] = {k: v for k, v in schedule.items() if k != 'batches'}
    result['successor_labels'] = label_counts(train_rows, schedule, args.labels)
    base = load_base()
    s_train = v3.ARMS[ARM]['steps']
    tally = []
    with v3.arm_model(base, ARM, start_directory) as model:
        if checkpoint_records(start_directory) != provenance['checkpoint_files']:
            raise ValueError('common start files changed between the digest check and loading')
        result['config'] = asdict(model.config)
        result['variant'] = asdict(v3.side_variant(model.adapters()[14].side))
        for parameter in model.adapter_parameters():
            parameter.requires_grad_(True)
        side = model.adapters()[14].side
        result['requires_grad_parameter_names'] = sorted(n for n, p in side.named_parameters() if p.requires_grad)
        forward = partial(succ_forward, labels=args.labels, tally=tally)
        result['train'] = train(model, train_rows, schedule, pad_id, v3.LR, v3.BATCH, output, None, None,
                                forward_fn=forward, learning_curves=False)
        if len(tally) != schedule['steps'] or sum(t['labels'] for t in tally) != result['successor_labels']['total']:
            raise AssertionError('successor labels seen in training differ from the schedule count')
        tail = tally[-100:]
        result['successor_loss'] = {
            'labels_seen': sum(t['labels'] for t in tally),
            'last100_answer_ce': sum(t['answer'] for t in tail)/len(tail),
            'last100_successor_ce': sum(t['successor'] for t in tail if t['labels'])/max(1, sum(1 for t in tail if t['labels'])),
            'first100_successor_ce': sum(t['successor'] for t in tally[:100] if t['labels'])/max(1, sum(1 for t in tally[:100] if t['labels'])),
            'replay_max_relative': max(t['replay_max_relative'] for t in tally)}
        v3._write_records(output/'successor.jsonl', tally)
        model.eval()
        torch.cuda.synchronize()
        tick = time.monotonic()
        records, reconstruction = v3.evaluate_arm(model, dev, pad_id, s_train)
        torch.cuda.synchronize()
        result['evaluation_seconds'] = time.monotonic()-tick
        directory = output/f'eval-s{s_train}'
        directory.mkdir()
        result['evaluation'] = {}
        for mode, mode_records in records.items():
            attach_paths(mode_records, dev)
            result['evaluation'][mode] = v3.summarize_arm(mode_records, prior)
            v3._write_records(directory/f'dev.{mode}.jsonl', mode_records)
        result['reconstruction'] = reconstruction
        v3.export_arm(model, output/'adapter')
    with v3.load_arm(base, output/'adapter') as reloaded:
        if asdict(reloaded.config) != result['config']:
            raise AssertionError('checkpoint reload configuration changed')
        reload_records, _ = v3.evaluate_arm(reloaded, dev, pad_id, s_train)
        for mode in reload_records:
            attach_paths(reload_records[mode], dev)
        if reload_records != records:
            raise AssertionError('checkpoint reload changed full per-example records')
    result['checkpoint_reload'] = {'full_records_identical': True, 'steps': s_train}
    result['checkpoint_files'] = checkpoint_records(output/'adapter')
    result['pilot'] = pilot_summary(records['normal'])


def _exact_mean(values):
    values = [Fraction(v) for v in values]
    return sum(values, Fraction(0))/len(values)


def pilot_summary(normal):
    """Dev hop counts and the hop-2 step-2 weight on path fact 2 from the S=4 normal records."""
    hops = {h: [r for r in normal if r['hop'] == h] for h in (1, 2, 3)}
    if any(not group for group in hops.values()):
        raise ValueError('every hop must be evaluated')
    weight = _exact_mean(r['path_weight_last'][1][1] for r in hops[2])
    return {'hops': {f'hop{h}': {'count': len(g), 'correct_count': sum(r['correct'] for r in g)} for h, g in hops.items()},
            'hop2_step2_fact2_exact': weight}


# ---- D-0035 comparison ----------------------------------------------------------------------------

def primary(hops):
    return sum((Fraction(hops[f'hop{h}']['correct_count'], hops[f'hop{h}']['count']) for h in (2, 3)), Fraction(0))*50


def decide(correct, permuted):
    """D-0035 on two pilot summaries (exact fractions; raw values next to each boolean)."""
    gap = primary(correct['hops']) - primary(permuted['hops'])
    weight_correct, weight_permuted = Fraction(correct['hop2_step2_fact2_exact']), Fraction(permuted['hop2_step2_fact2_exact'])
    i = gap >= EFFECT_PP
    ii = weight_correct >= ADDRESS and weight_permuted < ADDRESS
    hop1 = (Fraction(correct['hops']['hop1']['correct_count'], correct['hops']['hop1']['count'])
            - Fraction(permuted['hops']['hop1']['correct_count'], permuted['hops']['hop1']['count']))*100
    return {'decision': 'D-0035',
            'i_primary_em': {'correct_pp': primary(correct['hops']), 'permuted_pp': primary(permuted['hops']),
                             'difference_pp': gap, 'difference_exact': str(gap), 'threshold_pp': EFFECT_PP, 'passed': i},
            'ii_hop2_step2_fact2': {'correct': weight_correct, 'permuted': weight_permuted, 'threshold': ADDRESS,
                                    'passed': ii},
            'hop1_difference_pp_descriptive': hop1, 'hop1_difference_exact': str(hop1),
            'correct_beats_permuted': i and ii,
            'next': ('positive pilot under extra supervision: a new scoped proposal, not an unsupervised H-0005 result'
                     if i and ii else 'the repair route stops (AD-0003 drop condition)')}


def validate_pilot(run, labels, start):
    """A complete formal pilot of the given labels from the same start; returns its summary."""
    from v2r16_train import file_record
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True
            or manifest.get('contract') != v3.CONTRACT or manifest.get('command') != 'succ'
            or manifest.get('labels') != labels or manifest.get('smoke_only') is not False
            or manifest.get('eligible_as_evidence') is not True):
        raise ValueError(f'{run} is not a complete formal {labels} pilot')
    for artifact in manifest['artifacts']:
        if file_record(Path(artifact['path'])) != {k: artifact[k] for k in ('sha256', 'bytes')}:
            raise ValueError(f'artifact changed: {artifact["path"]}')
    metrics = json.loads((run/'metrics.json').read_text())
    if metrics.get('start') != start:
        raise ValueError(f'{run} was not trained from the analysis start')
    normal = [json.loads(line) for line in (run/'eval-s4'/'dev.normal.jsonl').read_text().splitlines()]
    summary = pilot_summary(normal)
    if v3._jsonable(summary) != metrics['pilot']:
        raise ValueError(f'{run}: recomputed pilot summary differs from its metrics')
    return summary, metrics


def run_compare(args, output, result):
    start = v3.start_provenance(args.start_run, args.start_sha256, ready_run=args.ready_run)
    if start['formal_start'] is not True:
        raise ValueError('the comparison needs the formal common start')
    correct, correct_metrics = validate_pilot(args.correct_run, 'correct', start)
    permuted, permuted_metrics = validate_pilot(args.permuted_run, 'permuted', start)
    for key in ('schedule', 'config', 'variant', 'data'):
        if correct_metrics[key] != permuted_metrics[key]:
            raise ValueError(f'the pilots differ in {key}')
    if correct_metrics['successor_labels']['total'] != permuted_metrics['successor_labels']['total']:
        raise ValueError('the pilots differ in their label count')
    result.update(start=start, runs={'correct': str(Path(args.correct_run).resolve()),
                                     'permuted': str(Path(args.permuted_run).resolve())},
                  summaries={'correct': correct, 'permuted': permuted}, d0035=decide(correct, permuted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('succ', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--start-run', type=Path, required=True)
        p.add_argument('--start-sha256', required=True)
        p.add_argument('--ready-run', type=Path, required=True)
    sub.choices['succ'].add_argument('--labels', choices=LABELS, required=True)
    sub.choices['succ'].add_argument('--smoke-steps', type=int, default=0)
    sub.choices['succ'].add_argument('--eval-limit', type=int, default=0)
    sub.choices['compare'].add_argument('--correct-run', type=Path, required=True)
    sub.choices['compare'].add_argument('--permuted-run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'succ' and bool(args.smoke_steps) != bool(args.eval_limit):
        raise ValueError('a smoke run sets both --smoke-steps and --eval-limit')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    from calibrate import MODEL, REVISION
    from v2_train import write_json
    from v2r16_train import file_record
    started = time.monotonic()
    smoke = bool(getattr(args, 'smoke_steps', 0))
    result = {'experiment': v3.CONTRACT, 'command': args.command, 'smoke_only': smoke}
    if args.command == 'succ':
        result.update(labels=args.labels, arm=ARM, coefficient=COEFFICIENT, decision='D-0035')
    passed, failure = False, None
    try:
        (run_pilot if args.command == 'succ' else run_compare)(args, output, result)
        passed = True
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        wall = time.monotonic()-started
        result.update(wall_seconds=wall, node_hours=wall/14400)
        write_json(output/'metrics.json', v3._jsonable(result))
        artifacts = [{'path': str(p.resolve()), 'role': str(p.relative_to(output)), **file_record(p)}
                     for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']
        write_json(output/'manifest.json', v3._jsonable({
            'status': 'complete' if passed else 'failed', 'experiment': v3.EXPERIMENT, 'contract': v3.CONTRACT,
            'command': args.command, 'arm': f'{ARM}-succ-{args.labels}' if args.command == 'succ' else 'succ-compare',
            'labels': getattr(args, 'labels', None), 'spec_digest': os.environ.get('SR_SPEC_DIGEST'),
            'model': MODEL, 'revision': REVISION, 'smoke_only': smoke, 'eligible_as_evidence': passed and not smoke,
            'extra_supervision': True, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'metrics': {'wall_seconds': wall, 'node_hours': wall/14400,
                        **({'d0035': result['d0035']} if 'd0035' in result else {}),
                        **({'pilot': result['pilot']} if 'pilot' in result else {})},
            'artifacts': artifacts, 'failure': failure,
            'validator': {'name': f'v3_succ {args.command}', 'passed': passed}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
