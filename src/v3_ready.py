"""E-0023 stage 0: T2 readiness rule R of the common start (ATT-0196-1) on T2 calibration.

No training. The reviewed E-0016 evaluator (isolated write, six controls, S=1) scores the start's
final checkpoint on the released calibration split, which no V3 run has evaluated before. R is
judged from those records at the answer position (the last valid question token); all four
criteria must hold, thresholds include equality and use exact fractions. Execute only in an
allocated GPU job. T2 hops 4/6/8 and confirmation are never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import time

CONTRACT = 'E-0023@1'
EXPERIMENT = 'E-0023'
START_RUN = 'ATT-0196-1'
START_ADAPTER_SHA256 = 'f70ea08e5f378012975e08bd5bac0083023fef78659174ba1a1d2a9977395417'
START_CONTRACT = 'E-0022@2'
EM_MARGIN = Fraction(1, 10)
ANSWER_WEIGHT = Fraction(1, 8)
TOP_SLOT_SHARE = Fraction(1, 4)
HOPS = (1, 2, 3)
ORIGIN = ('readiness rule fixed after the start-stage dev results were seen; '
          'checked once on the previously unused T2 calibration split')


def start_record(run, expected_sha256):
    """The completed formal V3-S0 run whose adapter.pt has the pinned digest (its old-gate result is kept)."""
    from v2r16_train import checkpoint_records, file_record
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True
            or manifest.get('arm') != 'V3-S0' or manifest.get('contract') != START_CONTRACT
            or manifest.get('eligible_as_evidence') is not True or manifest.get('smoke_steps')
            or manifest.get('eval_limit')):
        raise ValueError('the common start must be the completed formal V3-S0 run')
    files = checkpoint_records(run/'adapter')
    if files != manifest.get('checkpoint_files') or set(files) != {'adapter.pt', 'config.json', 'variant.json'}:
        raise ValueError('common start files differ from the run manifest')
    if files['adapter.pt']['sha256'] != expected_sha256:
        raise ValueError('common start adapter.pt differs from the pinned digest')
    return {'run': str(run.resolve()), 'checkpoint_files': files, 'expected_adapter_sha256': expected_sha256,
            'source_manifest': file_record(run/'manifest.json'),
            'old_gate': manifest.get('metrics', {}).get('hop1_gate')}


def _exact_mean(values):
    values = list(values)
    return sum(map(Fraction, values), Fraction(0))/len(values)


def readiness(records, rows):
    """Rule R from the reviewed per-example records ({control: [record]}) and the evaluated rows."""
    hop = {r['episode_id']: r['axes']['hop'] for r in rows}
    first_fact = {r['episode_id']: r['path']['fact_indices'][0] for r in rows}
    ids = [r['episode_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate episodes')
    for mode, group in records.items():
        if [r['episode_id'] for r in group] != ids:
            raise ValueError(f'{mode} records do not match the evaluated rows')
    for r in records['normal']:
        if r['target_fact_position'] != first_fact[r['episode_id']]:
            raise ValueError('the scored target fact is not the first path fact')
    by = {mode: {h: [r for r in group if hop[r['episode_id']] == h] for h in HOPS} for mode, group in records.items()}
    one = {m: by[m][1] for m in ('normal', 'zero', 'shuffled')}
    n = len(one['normal'])
    counts = {m: sum(r['correct'] for r in g) for m, g in one.items()}
    margin = Fraction(counts['normal']-max(counts['zero'], counts['shuffled']), n)
    nll = {m: sum(r['nll'] for r in g)/len(g) for m, g in one.items()}
    weights = {h: _exact_mean(r['last_target_weight'] for r in by['normal'][h]) for h in HOPS}
    slots = Counter(r['last_argmax'] for r in one['normal'])
    if any(s < 0 for s in slots):
        raise ValueError('normal hop-1 records without a valid last-position argmax')
    top_slot, top_count = max(slots.items(), key=lambda item: (item[1], -item[0]))
    top_share = Fraction(top_count, n)
    criteria = {
        'hop1_em_margin': {'value': margin, 'fraction': str(margin), 'threshold': str(EM_MARGIN),
                           'counts': counts, 'n': n, 'passed': margin >= EM_MARGIN},
        'hop1_nll': {'nll': nll, 'passed': nll['normal'] < min(nll['zero'], nll['shuffled'])},
        'answer_position_first_fact_weight': {
            'by_hop': {f'hop{h}': {'value': w, 'n': len(by['normal'][h]), 'passed': w >= ANSWER_WEIGHT}
                       for h, w in weights.items()},
            'threshold': str(ANSWER_WEIGHT), 'passed': all(w >= ANSWER_WEIGHT for w in weights.values())},
        'hop1_answer_position_top_slot_share': {'value': top_share, 'fraction': str(top_share), 'slot': top_slot,
                                                'threshold': str(TOP_SLOT_SHARE), 'passed': top_share <= TOP_SLOT_SHARE},
    }
    return {'passed': all(c['passed'] for c in criteria.values()), 'criteria': criteria,
            'descriptive': _descriptive(by), 'origin': ORIGIN, 'scope': 'answer position (last valid question token), S=1'}


def _descriptive(by):
    """All-position weight and last-position argmax facts per hop (not part of R)."""
    out = {}
    for h in HOPS:
        group = by['normal'][h]
        tokens = sum(r['question_tokens'] for r in group)
        slots = Counter(r['last_argmax'] for r in group)
        out[f'hop{h}'] = {
            'all_position_target_weight': sum(r['all_target_weight']*r['question_tokens'] for r in group)/tokens,
            'last_argmax_is_first_fact': sum(r['last_argmax'] == r['target_fact_position'] for r in group)/len(group),
            'last_weight_ge_half': sum(r['last_target_weight'] >= .5 for r in group)/len(group),
            'top_slot_share': max(slots.values())/len(group),
            'question_tokens': dict(sorted(Counter(r['question_tokens'] for r in group).items())),
            'em': {m: sum(r['correct'] for r in by[m][h])/len(by[m][h]) for m in by},
        }
    return out


def run_ready(args, output, result):
    from calibrate import load_base
    from v2_memory import isolated_support
    from v2r16_memory import evaluate
    from v2r16_train import STANDARD, checkpoint_records
    from v3_loop import T2_DATA, T2_METADATA_SHA256, load_t2, reviewed_by_hop
    from v3_train import _tokenizer, load_arm
    run = Path(args.start_run)
    result['start'] = start_record(run, args.start_sha256)
    tokenizer, pad_id = _tokenizer()
    metadata, train_rows, _ = load_t2(tokenizer, 'train')
    _, rows, differing = load_t2(tokenizer, 'calibration')
    if args.eval_limit:
        rows = [r for h in HOPS for r in [x for x in rows if x['axes']['hop'] == h][:args.eval_limit]]
    prior = Counter(r['query']['answer_token_id'] for r in train_rows).most_common(1)[0][0]
    result['data'] = {'directory': str(T2_DATA), 'metadata_sha256': T2_METADATA_SHA256, 'split': 'calibration',
                      'calibration_sha256': metadata['splits']['calibration']['sha256'],
                      'train_sha256': metadata['splits']['train']['sha256'], 'rows': len(rows),
                      'rows_by_hop': dict(sorted(Counter(r['axes']['hop'] for r in rows).items())),
                      'prior_token': prior, 'context_fact_tokens_differ_from_isolated': differing}
    base = load_base()
    with load_arm(base, run/'adapter') as model:
        if checkpoint_records(run/'adapter') != result['start']['checkpoint_files']:
            raise ValueError('adapter files changed between the digest check and loading')
        tick = time.monotonic()
        summary, records = evaluate(model, rows, pad_id, output/'evaluation', 'calibration', STANDARD, prior,
                                    support_fn=isolated_support)
        result['evaluation'] = {'summary': summary, 'by_hop': reviewed_by_hop(records, rows, prior),
                                'seconds': time.monotonic()-tick}
    if checkpoint_records(run/'adapter') != result['start']['checkpoint_files']:
        raise ValueError('adapter files changed during the evaluation')
    result['readiness'] = readiness(records, rows)
    by_hop = result['evaluation']['by_hop']
    for h in HOPS:  # the recomputed weight must agree with the reviewed summary
        reviewed = by_hop[f'hop{h}']['normal']['last_position']['target_weight_mean']
        if abs(float(result['readiness']['criteria']['answer_position_first_fact_weight']['by_hop'][f'hop{h}']['value'])
               - reviewed) > 1e-12:
            raise AssertionError('answer-position weight disagrees with the reviewed summary')


def main():
    import torch
    from v3_train import _jsonable
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('ready',))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-run', type=Path, default=Path('/work/xg24i002/x10041/bypass-lora/runs')/START_RUN/'run')
    parser.add_argument('--start-sha256', default=START_ADAPTER_SHA256)
    parser.add_argument('--eval-limit', type=int, default=0, help='smoke only: first N calibration rows per hop')
    args = parser.parse_args()
    if args.eval_limit and args.eval_limit < 2:
        parser.error('eval limit >= 2')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    from calibrate import MODEL, REVISION
    from v2_train import write_json
    from v2r16_train import file_record
    started = time.monotonic()
    smoke = bool(args.eval_limit)
    result, passed, failure = {'experiment': CONTRACT, 'command': args.command, 'arm': 'V3-ready',
                               'smoke_only': smoke}, False, None
    try:
        run_ready(args, output, result)
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
        ready = result.get('readiness')
        write_json(output/'manifest.json', _jsonable({
            'status': 'complete' if passed else 'failed', 'experiment': EXPERIMENT, 'contract': CONTRACT,
            'command': args.command, 'arm': 'V3-ready', 'write_protocol': 'isolated',
            'spec_digest': os.environ.get('SR_SPEC_DIGEST'), 'model': MODEL, 'revision': REVISION,
            'eval_limit': args.eval_limit, 'smoke_only': smoke, 'eligible_as_evidence': passed and not smoke,
            'start': result.get('start'), 'data': result.get('data'),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'hardware': ({'device': torch.cuda.get_device_name(0),
                          'multiprocessors': torch.cuda.get_device_properties(0).multi_processor_count}
                         if torch.cuda.is_available() else None),
            'metrics': {'wall_seconds': wall, 'mig_instance_hours': wall/3600, 'node_hours': wall/14400,
                        **({'readiness_passed': ready['passed'],
                            'readiness_criteria': {k: v['passed'] for k, v in ready['criteria'].items()}}
                           if ready else {})},
            'artifacts': artifacts, 'failure': failure,
            'validator': {'name': 'v3_ready', 'passed': passed}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
