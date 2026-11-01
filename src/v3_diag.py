"""E-0023 descriptive diagnostics for trained arms and the stage-A rule. Execute only in an
allocated job (the oracle evaluation needs the GPU; the rest reads saved records).

Descriptive only; none of these enters a rule:
- value oracles: the read of every step at every valid question position is replaced by the stored
  value of one fact, either the first path edge (path.fact_indices[0]; on the stock S=1 reader this
  is the reviewed E-0016 oracle_value, checked example by example) or the end edge
  (path.fact_indices[-1]). Update, readout, eta, read gain and Ro (per step for untied arms) are the
  arm's own. Not defined for the tokenwise kernel arm (no slot read). Neither is a reachable bound.
- per-step path weights: from the arm's verified main records, the answer-position (last valid
  question token) weight of step s on path fact j, by hop; the contract's statistic is s = j.
- EM by answer-name frequency: occurrences of each answer name as the answer of a released T2
  train row; evaluated rows split at the median of their answers' counts (high: count > median,
  low: count <= median).

Stage A (E-0023 rule 3) from the verified main counts of V3-01, V3-02 and V3-03:
A1 S_M* - V3-01 >= +3 pp and hop1 (S_M* - V3-01) >= -1 pp; A2 S_M* - V3-01 >= +3 pp, hop1 < -1 pp;
A3 0 < S_M* - V3-01 < +3 pp; A4 S_M* - V3-01 <= 0. Exact fractions, thresholds include equality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

import v3_train as v3

STAGE_A_ARMS = ('V3-01', 'V3-02', 'V3-03')
ORACLES = {'first_edge_value': lambda row: row['path']['fact_indices'][0],
           'end_edge_value': lambda row: row['path']['fact_indices'][-1]}


# ---- value oracles --------------------------------------------------------------------------------

def oracle_replay(side, qbar, snapshot, steps, fact):
    """The arm's loop with every read replaced by the value of slot ``fact`` [B] (all positions)."""
    reader, cfg = side.reader, side.config
    variant = v3.side_variant(side)
    if variant.kernel == 'tokenwise':
        raise ValueError('value oracles are not defined for the tokenwise kernel arm')
    row = torch.arange(len(fact), device=fact.device)
    if not bool(snapshot.valid[row, fact].all()):
        raise ValueError('oracle fact is not a valid slot')
    read = snapshot.values[row, fact][:, None].expand_as(qbar) * reader.read_gain
    z0 = qbar
    z = z0
    for s in range(steps):
        Ro = reader.step_matrices(s)[1] if variant.untied_steps else reader.Ro
        candidate = z0 if cfg.update == 'anchored' else z
        z = candidate + reader.eta * (read @ Ro)
    return z - z0 if cfg.readout == 'increment' else z


@torch.no_grad()
def oracle_batch(model, batch, steps):
    from v3_loop import _snapshots, hop_of
    side = model.adapters()[14].side
    snapshot = _snapshots(model, batch, 'normal', None)
    device = batch.labels.device
    out = {}
    with v3.eval_steps(model, steps):
        _, seen, _ = v3._capture(model, batch, snapshot)
        for name, pick in ORACLES.items():
            fact = torch.tensor([pick(r) for r in batch.rows], device=device)
            z = oracle_replay(side, seen['q'], snapshot, steps, fact)
            logits = v3._with_increment(model, batch, snapshot, z)
            losses = F.cross_entropy(logits, batch.labels, reduction='none').tolist()
            predictions, labels = logits.argmax(-1).tolist(), batch.labels.tolist()
            out[name] = [{'episode_id': r['episode_id'], 'hop': hop_of(r), 'fact': f, 'target': t,
                          'prediction': p, 'correct': p == t, 'nll': n, 'steps': steps}
                         for r, f, t, p, n in zip(batch.rows, fact.tolist(), labels, predictions, losses)]
    return out


@torch.no_grad()
def evaluate_oracles(model, rows, pad_id, steps):
    from v2_memory import collate
    from v3_loop import EVAL_BATCH
    model.eval()
    device = next(model.parameters()).device
    out = {name: [] for name in ORACLES}
    for start in range(0, len(rows), EVAL_BATCH):
        batch = collate(rows[start:start+EVAL_BATCH], pad_id, device)
        for name, records in oracle_batch(model, batch, steps).items():
            out[name].extend(records)
    return out


def summarize_by_hop(records):
    out = {}
    for hop in sorted({r['hop'] for r in records}):
        group = [r for r in records if r['hop'] == hop]
        out[f'hop{hop}'] = {'count': len(group), 'correct_count': sum(r['correct'] for r in group),
                            'em': sum(r['correct'] for r in group)/len(group),
                            'nll': sum(r['nll'] for r in group)/len(group)}
    return out


def compare_first_edge_to_reviewed(mine, reviewed):
    """Exact per-example comparison with the reviewed oracle_value records (same order)."""
    if [r['episode_id'] for r in mine] != [r['episode_id'] for r in reviewed]:
        raise AssertionError('oracle and reviewed records cover different episodes or orders')
    differing = Counter(k for a, b in zip(mine, reviewed) for k in ('prediction', 'correct', 'nll') if a[k] != b[k])
    return {'episodes': len(mine), 'differing_examples_by_field': dict(differing), 'identical': not differing}


# ---- saved-record diagnostics ---------------------------------------------------------------------

def _mean(values):
    values = list(values)
    return sum(values)/len(values)


def path_weight_table(records, steps):
    """table[hop][s][j]: mean answer-position weight of step s+1 on path fact j+1 (hop rows);
    diagonal[hop][s]: step s+1 on path fact s+1 for s < min(S, hop)."""
    out = {}
    for hop in sorted({r['hop'] for r in records}):
        group = [r for r in records if r['hop'] == hop]
        if any(r['path_weight_last'] is None or len(r['path_weight_last']) != steps
               or any(len(w) != hop for w in r['path_weight_last']) for r in group):
            raise ValueError('records lack per-step answer-position path weights')
        table = [[_mean(r['path_weight_last'][s][j] for r in group) for j in range(hop)] for s in range(steps)]
        out[f'hop{hop}'] = {'count': len(group), 'table': table,
                            'diagonal': [table[s][s] for s in range(min(steps, hop))]}
    return out


def secondary_mechanism(table, steps):
    """E-0023 rule 5 (reported, not judged): hop2 step-2 weight on fact 2 against 1/16 and 1/8;
    for S4 also hop3 step-3 weight on fact 3."""
    out = {}
    if steps >= 2 and 'hop2' in table:
        w = table['hop2']['table'][1][1]
        out['hop2_step2_fact2'] = {'value': w, 'vs_1_16': w - 1/16, 'vs_1_8': w - 1/8}
    if steps >= 4 and 'hop3' in table:
        w = table['hop3']['table'][2][2]
        out['hop3_step3_fact3'] = {'value': w, 'vs_1_16': w - 1/16, 'vs_1_8': w - 1/8}
    return out


def answer_name(row):
    return row['query']['answer_text'].strip()


def frequency_strata(records, rows, train_rows):
    counts = Counter(answer_name(r) for r in train_rows)
    count_of = {r['episode_id']: counts[answer_name(r)] for r in rows}
    if any(r['episode_id'] not in count_of for r in records):
        raise ValueError('records outside the evaluated rows')
    median = statistics.median(count_of[r['episode_id']] for r in records)
    out = {'median_train_answer_count': median, 'rule': 'high: count > median; low: count <= median', 'strata': {}}
    for name, keep in (('high', lambda c: c > median), ('low', lambda c: c <= median)):
        members = [r for r in records if keep(count_of[r['episode_id']])]
        groups = {'all': members, **{f'hop{h}': [r for r in members if r['hop'] == h] for h in (1, 2, 3)}}
        out['strata'][name] = {g: {'count': len(m), 'correct_count': sum(r['correct'] for r in m),
                                   'em': sum(r['correct'] for r in m)/len(m) if m else None,
                                   'train_count_range': [min(count_of[r['episode_id']] for r in m),
                                                         max(count_of[r['episode_id']] for r in m)] if m else None}
                               for g, m in groups.items()}
    return out


# ---- stage A --------------------------------------------------------------------------------------

def stage_a(main):
    """E-0023 rule 3 on the main counts of V3-01, V3-02, V3-03 (S_M* by E-0022 rule 3)."""
    if set(main) != set(STAGE_A_ARMS):
        raise ValueError('stage A needs exactly V3-01, V3-02 and V3-03')
    pp = {arm: v3.primary_fraction(h)*100 for arm, h in main.items()}
    hop1 = {arm: v3.hop_fraction(h, 1)*100 for arm, h in main.items()}
    star = 'V3-03' if pp['V3-03'] - pp['V3-02'] >= Fraction(1, 2) else 'V3-02'
    gain, hop1_gap = pp[star] - pp['V3-01'], hop1[star] - hop1['V3-01']
    if gain >= 3:
        case = 'A1' if hop1_gap >= -1 else 'A2'
    else:
        case = 'A3' if gain > 0 else 'A4'
    next_step = {
        'A1': 'release stage B (V3-04, V3-05 at S_M*=4 or V3-05b at S_M*=2, V3-06..V3-10); judge GL_M by E-0022 rule 4 with (e) at the S_M* depth',
        'A2': 'train V3-09 and V3-10 first; release the rest of stage B only if one of them meets (a) and (b) against V3-01',
        'A3': 'descriptive only, stage B not released; one frozen-transfer diagnostic and at most one repair arm (rule 4), then the conclusion',
        'A4': 'screening evidence challenges H-0005@2 (E-0022 rule 7); one frozen-transfer diagnostic and at most one repair arm, then the conclusion review',
    }[case]
    exact = lambda v: f'{v.numerator}/{v.denominator}'
    return {'primary_pp': {a: float(v) for a, v in pp.items()}, 'primary_exact': {a: exact(v) for a, v in pp.items()},
            'hop1_pp': {a: float(v) for a, v in hop1.items()},
            'S_M*': star, 'S_M*_steps': v3.ARMS[star]['steps'],
            'S_M*_minus_V3-01_pp': float(gain), 'S_M*_minus_V3-01_exact': exact(gain),
            'hop1_S_M*_minus_V3-01_pp': float(hop1_gap), 'hop1_S_M*_minus_V3-01_exact': exact(hop1_gap),
            'case': case, 'next': next_step}


def analyze_stage_a(runs, reference, start):
    validated = {}
    for run in runs:
        arm, metrics, records = v3.validate_arm_run(run, reference, start)
        if arm in validated or arm not in STAGE_A_ARMS:
            raise ValueError('one run each of V3-01, V3-02 and V3-03')
        validated[arm] = (Path(run), metrics, records)
    if set(validated) != set(STAGE_A_ARMS):
        raise ValueError('stage A needs V3-01, V3-02 and V3-03')
    main, correct, table = {}, {}, {}
    for arm, (run, metrics, records) in sorted(validated.items()):
        main[arm] = {f'hop{h}': {'count': sum(r['hop'] == h for r in records),
                                 'correct_count': sum(r['correct'] for r in records if r['hop'] == h)}
                     for h in (1, 2, 3)}
        correct[arm] = {h: [int(r['correct']) for r in records if r['hop'] == h] for h in (1, 2, 3)}
        evaluation = metrics['evaluation'][str(v3.ARMS[arm]['steps'])]
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
    pairs = [('V3-02', 'V3-01'), ('V3-03', 'V3-01'), ('V3-03', 'V3-02')]
    return {'stage_a': stage_a(main), 'arms': table,
            'bootstrap': {'resamples': v3.BOOTSTRAP_RESAMPLES, 'seed': v3.SEED, 'stratified_by': 'hop', 'paired': True,
                          'intervals_pp': v3.bootstrap(correct, pairs), 'descriptive_only': True},
            'inputs': {'start': start, 'reference': {k: v for k, v in reference.items() if k != 'dev'},
                       'dev_rows': len(reference['dev'])}}


# ---- runners --------------------------------------------------------------------------------------

def _start(args):
    return v3.start_provenance(args.start_run, args.start_sha256, ready_run=args.ready_run)


def run_diag(args, output, result):
    from calibrate import load_base
    from v2r16_train import checkpoint_records
    from v3_eval import load_released_data
    from v3_loop import T2_DATA, T2_METADATA_SHA256
    start = _start(args)
    if start['formal_start'] is not True:
        raise ValueError('diagnostics need the formal common start')
    reference = v3.analysis_reference()
    run = Path(args.arm_run)
    arm, metrics, records = v3.validate_arm_run(run, reference, start)
    files = checkpoint_records(run/'adapter')
    if files != metrics.get('checkpoint_files') or files['adapter.pt']['sha256'] != args.arm_sha256:
        raise ValueError('arm adapter differs from its metrics or the pinned digest')
    steps = v3.ARMS[arm]['steps']
    result['arm_run'] = {'run': str(run.resolve()), 'arm': arm, 'steps': steps, 'checkpoint_files': files}
    result['start'] = start
    tokenizer, pad_id = v3._tokenizer()
    _, dev, prior, result['data'] = v3._data(tokenizer)
    _, train_rows = load_released_data(T2_DATA, T2_METADATA_SHA256, 'train')
    result['path_weights'] = path_weight_table(records, steps) if records[0]['path_weight_last'] is not None else None
    result['secondary_mechanism'] = (secondary_mechanism(result['path_weights'], steps)
                                     if result['path_weights'] else None)
    result['frequency'] = frequency_strata(records, dev, train_rows)
    if v3.arm_variant(arm).kernel == 'tokenwise':
        result['oracles'] = None
        return
    base = load_base()
    with v3.load_arm(base, run/'adapter') as model:
        if checkpoint_records(run/'adapter') != files:
            raise ValueError('arm adapter changed between the digest check and loading')
        tick = time.monotonic()
        oracles = evaluate_oracles(model, dev, pad_id, steps)
        result['oracle_seconds'] = time.monotonic()-tick
        if [(r['episode_id'], r['prediction'], r['nll']) for r in oracles['first_edge_value'] if r['hop'] == 1] != \
                [(r['episode_id'], r['prediction'], r['nll']) for r in oracles['end_edge_value'] if r['hop'] == 1]:
            raise AssertionError('first- and end-edge oracles differ on hop-1 rows (the same fact)')
        if steps == 1 and v3.side_variant(model.adapters()[14].side) == v3.Variant() and model.config.eta == .25:
            from v2_memory import isolated_support
            from v2r16_memory import evaluate
            _, reviewed = evaluate(model, dev, pad_id, output/'reviewed-oracle', 'dev', ('oracle_value',), prior,
                                   support_fn=isolated_support)
            result['first_edge_vs_reviewed'] = compare_first_edge_to_reviewed(oracles['first_edge_value'],
                                                                              reviewed['oracle_value'])
            if not result['first_edge_vs_reviewed']['identical']:
                raise AssertionError('S=1 first-edge oracle differs from the reviewed oracle_value')
    result['oracles'] = {name: summarize_by_hop(group) for name, group in oracles.items()}
    for name, group in oracles.items():
        v3._write_records(output/f'dev.{name}.jsonl', group)


def run_stage_a(args, output, result):
    start = _start(args)
    if start['formal_start'] is not True:
        raise ValueError('stage-A analysis needs the formal common start')
    result.update(analyze_stage_a(args.arm_runs, v3.analysis_reference(), start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('diag', 'stage-a'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-run', type=Path, required=True)
    parser.add_argument('--start-sha256', required=True)
    parser.add_argument('--ready-run', type=Path, required=True)
    parser.add_argument('--arm-run', type=Path, help='diag: one formal arm run')
    parser.add_argument('--arm-sha256', help='diag: its pinned adapter.pt digest')
    parser.add_argument('--arm-runs', type=Path, nargs='*', default=(), help='stage-a: the V3-01/02/03 runs')
    args = parser.parse_args()
    if args.command == 'diag' and (args.arm_run is None or args.arm_sha256 is None or args.arm_runs):
        parser.error('diag needs --arm-run and --arm-sha256 only')
    if args.command == 'stage-a' and (len(args.arm_runs) != 3 or args.arm_run or args.arm_sha256):
        parser.error('stage-a needs the three --arm-runs only')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    from calibrate import MODEL, REVISION
    from v2_train import write_json
    from v2r16_train import file_record
    started = time.monotonic()
    result, passed, failure = {'experiment': v3.CONTRACT, 'command': args.command}, False, None
    try:
        {'diag': run_diag, 'stage-a': run_stage_a}[args.command](args, output, result)
        passed = True
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        wall = time.monotonic()-started
        result.update(wall_seconds=wall, mig_instance_hours=wall/3600, node_hours=wall/14400)
        write_json(output/'metrics.json', v3._jsonable(result))
        artifacts = [{'path': str(p.resolve()), 'role': str(p.relative_to(output)), **file_record(p)}
                     for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']
        write_json(output/'manifest.json', v3._jsonable({
            'status': 'complete' if passed else 'failed', 'experiment': v3.EXPERIMENT, 'contract': v3.CONTRACT,
            'command': args.command, 'arm': (result.get('arm_run') or {}).get('arm'),
            'spec_digest': os.environ.get('SR_SPEC_DIGEST'), 'model': MODEL, 'revision': REVISION,
            'eligible_as_evidence': passed, 'smoke_only': False, 'descriptive_only': args.command == 'diag',
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'metrics': {'wall_seconds': wall, 'node_hours': wall/14400,
                        **({'stage_a': result['stage_a']} if 'stage_a' in result else {})},
            'artifacts': artifacts, 'failure': failure, 'validator': {'name': 'v3_diag', 'passed': passed}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
