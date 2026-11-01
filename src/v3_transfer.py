"""E-0023 rule 4: frozen-transfer diagnostic of the S_M* arm (no training). Execute only in an
allocated GPU job. Rows: T2 calibration hops 2-3; T2 hops 4/6/8 and confirmation are never opened.

Six reads of the frozen arm, all scored at its S_train with the chain end as the label:
- native: the arm's own loop (equal to its native forward; replay checked within 1e-3);
- forced_first: z^(1) is set to the state after reading the correct first-edge value
  (q̄ + η·(v[path fact 1]·g)·Ro); step 2 onward queries and reads natively;
- bridge: the question's subject is replaced by the gold bridge entity (path.entities[1]); the
  template and relation are unchanged, so the step-1 read should address path fact 2;
- wrong_bridge: subject = head of the first same-relation fact that is off the path and whose head
  is not a path entity (its own fact is reported as ``own_fact``);
- donor_bridge: subject = the gold bridge of the nearest earlier row (cyclic, evaluated order)
  whose bridge entity does not occur in this episode;
- first/end-edge value oracles (v3_diag, reviewed RV-0067) and the zero/shuffled memory controls
  (v3_train.evaluate_arm, global donor rule) on the same rows.
Per read and hop: EM, the answer-position (last valid question token) weight of every step on every
path fact, and the step-1 argmax share on path fact 2.

The rule-4 mapping (D-0034, pooled hops 2-3, answer position, exact fractions):
(a) end-edge oracle EM - max(EM_zero, EM_shuffled) >= 10 pp; (b) bridge step-1 weight on path
fact 2 >= 1/8 while the wrong-bridge and donor-bridge means on that fact are both < 1/8;
(c) native step-2 weight on path fact 2 < 1/8. Not (a): no repair; (a)(b)(c): successor-slot CE S4
pilot with the permuted-label control; (a) without (b): linear-start S4; (a)(b) without (c): no
repair. The forced first-edge read is descriptive only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import v3_diag as diag
import v3_train as v3

READS = ('native', 'forced_first', 'bridge', 'wrong_bridge', 'donor_bridge')
HOPS = (2, 3)
ADDRESS = Fraction(1, 8)  # D-0034: readiness rule R / rule 5 level
DECODE_PP = Fraction(10)  # D-0034: H3 margin


# ---- rewritten questions -------------------------------------------------------------------------

def _question(tok, row, subject):
    from v2_data import encode
    from v3_data import QUESTION_TEMPLATE
    text = QUESTION_TEMPLATE.format(head=subject, relation=row['query']['relation'])
    return dict(row, question_text=text, question_input_ids=encode(tok, text))


def wrong_fact(row):
    """First same-relation fact off the path whose head is not a path entity (or None)."""
    path = set(row['path']['fact_indices'])
    entities = set(row['path']['entities'])
    for i, fact in enumerate(row['facts']):
        if i not in path and fact['relation'] == row['query']['relation'] and fact['head'] not in entities:
            return i
    return None


def donor_bridge(rows, i):
    """Gold bridge of the nearest earlier row (cyclic) that does not occur in row i's episode."""
    row = rows[i]
    present = {x for f in row['facts'] for x in (f['head'], f['tail'])} | set(row['path']['entities'])
    for k in range(1, len(rows)):
        name = rows[(i-k) % len(rows)]['path']['entities'][1]
        if name not in present:
            return name
    raise ValueError('no donor bridge absent from the episode')


def read_rows(tok, rows):
    """{read: rows}; wrong_bridge keeps only rows that have a wrong fact (counted)."""
    out = {'native': rows, 'forced_first': rows,
           'bridge': [_question(tok, r, r['path']['entities'][1]) for r in rows],
           'donor_bridge': [_question(tok, r, donor_bridge(rows, i)) for i, r in enumerate(rows)]}
    wrong = []
    for r in rows:
        i = wrong_fact(r)
        if i is not None:
            wrong.append(dict(_question(tok, r, r['facts'][i]['head']), _own_fact=i))
    out['wrong_bridge'] = wrong
    return out


# ---- frozen reads ---------------------------------------------------------------------------------

@torch.no_grad()
def frozen_batch(model, batch, steps, forced=False):
    from v3_loop import RECONSTRUCTION_TOLERANCE, _snapshots, hop_of
    side = model.adapters()[14].side
    reader = side.reader
    variant = v3.side_variant(side)
    if variant.kernel == 'tokenwise':
        raise ValueError('the frozen-transfer diagnostic needs a slot-reading arm')
    snapshot = _snapshots(model, batch, 'normal', None)
    mask = batch.question_mask
    row = torch.arange(len(batch.rows), device=mask.device)
    last = mask.long().sum(-1)-1
    with v3.eval_steps(model, steps):
        logits, seen, captured = v3._capture(model, batch, snapshot)
        qbar, kbar, vbar = seen['q'], seen['k'], seen['v']
        if not forced:
            z, trajectory = v3.replay(side, qbar, kbar, vbar, snapshot, steps)
            native = captured['z']
            relative = (z-native).norm(dim=-1)/native.norm(dim=-1).clamp_min(1e-6)
            if not float(relative[mask].max()) <= RECONSTRUCTION_TOLERANCE:
                raise AssertionError('replay differs from the native output')
        else:
            if steps < 2:
                raise ValueError('forcing the first read needs S >= 2')
            first = torch.tensor([r['path']['fact_indices'][0] for r in batch.rows], device=mask.device)
            read = snapshot.values[row, first][:, None].expand_as(qbar) * reader.read_gain
            Ro = reader.step_matrices(0)[1] if variant.untied_steps else reader.Ro
            state = qbar + reader.eta * (read @ Ro)
            z, trajectory = v3.replay(side, qbar, kbar, vbar, snapshot, steps, replace_state=(1, lambda _: state))
            logits = v3._with_increment(model, batch, snapshot, z)
    losses = F.cross_entropy(logits, batch.labels, reduction='none').tolist()
    predictions, labels = logits.argmax(-1).tolist(), batch.labels.tolist()
    weights = [step['weights'][row, last] for step in trajectory]  # [B, n] per step
    records = []
    for i, r in enumerate(batch.rows):
        records.append({'episode_id': r['episode_id'], 'hop': hop_of(r), 'target': labels[i],
                        'prediction': predictions[i], 'correct': predictions[i] == labels[i], 'nll': losses[i],
                        'question_tokens': int(mask[i].sum()), 'path': r['path']['fact_indices'],
                        'own_fact': r.get('_own_fact'), 'weights_last': [w[i].tolist() for w in weights]})
    return records


@torch.no_grad()
def frozen_reads(model, rows_by_read, pad_id, steps):
    from v2_memory import collate
    from v3_loop import EVAL_BATCH
    model.eval()
    device = next(model.parameters()).device
    out = {}
    for read, rows in rows_by_read.items():
        records = []
        for start in range(0, len(rows), EVAL_BATCH):
            batch = collate(rows[start:start+EVAL_BATCH], pad_id, device)
            records.extend(frozen_batch(model, batch, steps, forced=read == 'forced_first'))
        out[read] = records
    return out


# ---- summaries and the rule-4 mapping ------------------------------------------------------------

def _mean(values):
    values = list(values)
    return sum(values)/len(values) if values else None


def summarize_reads(records, steps):
    out = {}
    for read, group in records.items():
        out[read] = {}
        for name, members in (('hops23', group), *((f'hop{h}', [r for r in group if r['hop'] == h]) for h in HOPS)):
            if not members:
                continue
            width = min(r['hop'] for r in members)
            out[read][name] = {
                'count': len(members), 'correct_count': sum(r['correct'] for r in members),
                'em': sum(r['correct'] for r in members)/len(members),
                'nll': _mean(r['nll'] for r in members),
                'path_weight_last': [[_mean(r['weights_last'][s][r['path'][j]] for r in members) for j in range(width)]
                                     for s in range(steps)],
                'step1_argmax_is_fact2': _mean(max(range(len(r['weights_last'][0])), key=r['weights_last'][0].__getitem__)
                                               == r['path'][1] for r in members),
                'own_fact_weight_last': ([_mean(r['weights_last'][s][r['own_fact']] for r in members) for s in range(steps)]
                                         if members[0]['own_fact'] is not None else None)}
    return out


def _exact_mean(records, step, fact):
    values = [Fraction(r['weights_last'][step][fact(r)]) for r in records]
    return sum(values, Fraction(0))/len(values)


def _em(records):
    return Fraction(sum(r['correct'] for r in records), len(records))


def mapping(records, oracle_records, control_records):
    """D-0034 criteria (a)-(c) and the rule-4 correspondence on the pooled hops 2-3 rows."""
    fact2 = lambda r: r['path'][1]
    zero, shuffled = _em(control_records['zero']), _em(control_records['shuffled'])
    end = _em(oracle_records['end_edge_value'])
    gap = (end - max(zero, shuffled))*100
    bridge = _exact_mean(records['bridge'], 0, fact2)
    controls = {r: _exact_mean(records[r], 0, fact2) for r in ('wrong_bridge', 'donor_bridge')}
    native = _exact_mean(records['native'], 1, fact2)
    a = gap >= DECODE_PP
    b = bridge >= ADDRESS and all(c < ADDRESS for c in controls.values())
    c = native < ADDRESS
    criteria = {
        'a_end_value_decodes': {'end_edge_em': end, 'zero_em': zero, 'shuffled_em': shuffled, 'gap_pp': gap,
                                'gap_exact': str(gap), 'threshold_pp': DECODE_PP, 'passed': a,
                                'rows': len(oracle_records['end_edge_value'])},
        'b_bridge_addresses_next_edge': {'bridge_step1_fact2': bridge, 'wrong_bridge_step1_fact2': controls['wrong_bridge'],
                                         'donor_bridge_step1_fact2': controls['donor_bridge'], 'threshold': ADDRESS,
                                         'rows': {r: len(records[r]) for r in ('bridge', 'wrong_bridge', 'donor_bridge')},
                                         'passed': b},
        'c_native_transition_fails': {'native_step2_fact2': native, 'threshold': ADDRESS, 'passed': c,
                                      'rows': len(records['native'])},
    }
    if not a:
        repair = 'none: (a) fails, descriptive only'
    elif not b:
        repair = 'linear-start S4 (AD-0004)'
    elif c:
        repair = 'successor-slot CE S4 pilot with the permuted-successor-label control (AD-0003 proposal 3)'
    else:
        repair = 'none: (a)(b) hold but (c) fails; descriptive only, go to the conclusion'
    return {'decision': 'D-0034', 'criteria': criteria, 'repair': repair}


# ---- runner ---------------------------------------------------------------------------------------

def run_transfer(args, output, result):
    from calibrate import load_base
    from v2r16_train import checkpoint_records
    from v3_loop import load_t2
    start = v3.start_provenance(args.start_run, args.start_sha256, ready_run=args.ready_run)
    if start['formal_start'] is not True:
        raise ValueError('the diagnostic needs the formal common start')
    run = Path(args.arm_run)
    arm, metrics, _ = v3.validate_arm_run(run, v3.analysis_reference(), start)
    files = checkpoint_records(run/'adapter')
    if files != metrics.get('checkpoint_files') or files['adapter.pt']['sha256'] != args.arm_sha256:
        raise ValueError('arm adapter differs from its metrics or the pinned digest')
    steps = v3.ARMS[arm]['steps']
    if steps < 2:
        raise ValueError('the frozen-transfer diagnostic is defined for S_M* (S >= 2)')
    result.update(start=start, arm_run={'run': str(run.resolve()), 'arm': arm, 'steps': steps, 'checkpoint_files': files})
    tokenizer, pad_id = v3._tokenizer()
    metadata, rows, _ = load_t2(tokenizer, 'calibration')
    rows = [r for r in rows if r['axes']['hop'] in HOPS]
    if args.eval_limit:
        rows = [r for h in HOPS for r in [x for x in rows if x['axes']['hop'] == h][:args.eval_limit]]
    by_read = read_rows(tokenizer, rows)
    result['data'] = {'split': 'calibration', 'calibration_sha256': metadata['splits']['calibration']['sha256'],
                      'rows_by_hop': dict(sorted(Counter(r['axes']['hop'] for r in rows).items())),
                      'rows_by_read': {k: len(v) for k, v in by_read.items()}}
    base = load_base()
    with v3.load_arm(base, run/'adapter') as model:
        if checkpoint_records(run/'adapter') != files:
            raise ValueError('arm adapter changed between the digest check and loading')
        tick = time.monotonic()
        records = frozen_reads(model, by_read, pad_id, steps)
        oracle_records = diag.evaluate_oracles(model, rows, pad_id, steps)
        control_records, _ = v3.evaluate_arm(model, rows, pad_id, steps, ('zero', 'shuffled'))
        result['seconds'] = time.monotonic()-tick
    result['reads'] = summarize_reads(records, steps)
    result['oracles'] = {name: diag.summarize_by_hop(group) for name, group in oracle_records.items()}
    result['controls'] = {name: diag.summarize_by_hop(group) for name, group in control_records.items()}
    result['mapping'] = mapping(records, oracle_records, control_records)
    for name, group in {**records, **oracle_records, **{f'control_{k}': v for k, v in control_records.items()}}.items():
        v3._write_records(output/f'calibration.{name}.jsonl', group)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-run', type=Path, required=True)
    parser.add_argument('--start-sha256', required=True)
    parser.add_argument('--ready-run', type=Path, required=True)
    parser.add_argument('--arm-run', type=Path, required=True)
    parser.add_argument('--arm-sha256', required=True)
    parser.add_argument('--eval-limit', type=int, default=0, help='smoke only: first N calibration rows per hop')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    from calibrate import MODEL, REVISION
    from v2_train import write_json
    from v2r16_train import file_record
    started = time.monotonic()
    smoke = bool(args.eval_limit)
    result, passed, failure = {'experiment': v3.CONTRACT, 'command': 'transfer', 'smoke_only': smoke}, False, None
    try:
        run_transfer(args, output, result)
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
            'command': 'transfer', 'arm': (result.get('arm_run') or {}).get('arm'),
            'spec_digest': os.environ.get('SR_SPEC_DIGEST'), 'model': MODEL, 'revision': REVISION,
            'eval_limit': args.eval_limit, 'smoke_only': smoke, 'eligible_as_evidence': passed and not smoke,
            'descriptive_only': True, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'metrics': {'wall_seconds': wall, 'node_hours': wall/14400,
                        **({'repair': result['mapping']['repair']} if 'mapping' in result else {})},
            'artifacts': artifacts, 'failure': failure, 'validator': {'name': 'v3_transfer', 'passed': passed}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
