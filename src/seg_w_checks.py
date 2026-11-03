"""E-0032 rule-1 PBS checks, score-discarding cost smoke and the E-0030 checkpoint replay.

mode=smoke: deterministic tests; the seed-0 plan (must reproduce E-0025's order
hash); segment-1 write-mask audit over the seed-0 training plan and all dev rows;
initial parameters at seeds 0-2 (W1/W2 equal RES-0281's SEG-05 digests, the X
triplet elementwise identical); a 1% uniform-step timing rehearsal of the five
arms at seed 0 with 32 dev rows per task (no dev quality retained); a cost
projection against the seed-0 (2.2) and total (5.0) node-h limits. With
--replay it also replays the retained E-0030 SEG-05 checkpoint (below).
mode=checks: deterministic tests only, at the exact formal source.
mode=replay: only the replay. The PBS wrapper runs it only after the allocated
MIG slice is verified to be the checkpoint's training slice (2g.24gb).

Replay: the seed-3 SEG-05 checkpoint of ATT-0379-2 (trained on 2g.24gb) is
evaluated on dev with this source; every prediction row (scores included) must
equal its recorded predictions.json.
"""
import argparse
import gc
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from rb1_artifacts import read, reference
from seg_data import ORDER_SHA256, segment1_write
from seg_eval import evaluate
from seg_model import (make_model, KEEP, PARAMETERS, FAMILIES, parameter_digest, E0032_ARMS, WRITE_RULE,
                       X_TAP, SEG05_INITIAL_SHA256)
from seg_train import tokenizer, prepare_plan, train, write_summary
from v1_run import prepare
from v5_train import write_json

FILES = ['src/test_seg_w.py', 'src/test_seg_w_rules.py', 'src/test_seg_model.py', 'src/test_seg_data.py',
         'src/test_seg_seeds.py', 'src/test_core.py', 'src/test_rb_core.py']
SEED0_CAP_NODE_H, TOTAL_CAP_NODE_H = 2.2, 5.0
REPLAY_RUN = Path('/work/xg24i002/x10041/bypass-lora/research/artifacts/ATT-0379-2')
REPLAY = {'arm': 'SEG-05', 'seed': 3, 'hardware': 'NVIDIA GH200 120GB MIG 2g.24gb'}


def checks(out):
    with (out/'checks.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'pytest', *FILES, '-q', '-p', 'no:cacheprovider',
            '-o', 'junit_family=legacy', '--junitxml='+str(out/'checks.xml')], stdout=log, stderr=subprocess.STDOUT)
    cases = list(ET.parse(out/'checks.xml').getroot().iter('testcase'))
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    if result.returncode != 0 or failed or skipped or not cases:
        raise RuntimeError('E-0032 PBS deterministic checks failed; see checks.log/checks.xml')
    return {'checks.cases': len(cases), 'checks.failed': failed, 'checks.skipped': skipped}


def mask_audit(tok, batches, splits):
    """Rule 1: W1 writes exactly the last token; W2 = sentence ends plus the last token."""
    from seg_data import encode_segments
    sources = {'train_seed0': [e['segment1_ids'] for b in batches for e in b],
               'dev': [encode_segments(tok, r)[0]['segment1_ids'] for task in ('piqa', 'boolq')
                       for r in splits['splits'][task]['dev']]}
    dev_tasks = [task for task in ('piqa', 'boolq') for _ in splits['splits'][task]['dev']]
    result = {}
    for name, rows in sources.items():
        counts = {arm: [] for arm in WRITE_RULE}
        for ids in rows:
            w1 = segment1_write(tok, ids, 'last'); w2 = segment1_write(tok, ids, 'sentence_last')
            if sum(w1) != 1 or not w1[-1] or not w2[-1] or any(a and not b for a, b in zip(w1, w2)):
                raise ValueError('segment-1 write mask violates the registered rule')
            counts['SEG-W1'].append(sum(w1)); counts['SEG-W2'].append(sum(w2))
        result[name] = {arm: write_summary(c) for arm, c in counts.items()}
        result[name]['segment1_tokens'] = write_summary([len(ids) for ids in rows])
        if name == 'dev':
            for task in ('piqa', 'boolq'):
                selected = [c for c, t in zip(counts['SEG-W2'], dev_tasks) if t == task]
                result[name]['SEG-W2.'+task] = write_summary(selected)
    if result['train_seed0']['SEG-W1']['max'] != 1 or result['dev']['SEG-W1']['min'] != 1:
        raise ValueError('W1 does not write exactly once per example')
    return result


def digests():
    result = {}
    for seed in (0, 1, 2):
        result[seed], anchor = {}, None
        for arm in E0032_ARMS:
            model = make_model(arm, seed=seed)
            result[seed][arm] = parameter_digest(model)
            if arm in X_TAP:
                state = [p.detach().cpu().clone() for p in model.adapter_parameters()]
                anchor = anchor or state
                if len(state) != len(anchor) or not all(torch.equal(a, b) for a, b in zip(state, anchor)):
                    raise ValueError(f'seed {seed} X-tap initial parameters differ elementwise')
            if sum(p.numel() for p in model.adapter_parameters()) != PARAMETERS[arm]:
                raise ValueError('parameter count differs from the E-0032 table')
            del model; gc.collect(); torch.cuda.empty_cache()
        if not result[seed]['SEG-W1'] == result[seed]['SEG-W2'] == SEG05_INITIAL_SHA256[seed]:
            raise ValueError(f'seed {seed} W initial digest differs from RES-0281 SEG-05')
    return result


def timing(tok, out, batches_full, splits, lock):
    size = math.ceil(len(batches_full)*.01)
    indices = np.linspace(0, len(batches_full)-1, size, dtype=int).tolist()
    batches = [batches_full[i] for i in indices]
    arms = {}
    for arm in E0032_ARMS:
        root = out/arm; root.mkdir(parents=True, exist_ok=True)
        tick = time.monotonic(); model = make_model(arm, seed=0); load = time.monotonic()-tick
        m = train(model, arm, batches, tok, lock['locked_lrs'][FAMILIES[arm]], root)
        if m['params.with_grad'] != PARAMETERS[arm]:
            raise ValueError('smoke parameter-gradient coverage differs')
        tick = time.monotonic(); _, before, audit = evaluate(model, arm, splits, tok, limit=32)
        eval_s = time.monotonic()-tick
        interventions = {}
        for kind in ('iv_off', 'iv_swap') if arm in KEEP else ('iv_off',):
            tick = time.monotonic(); evaluate(model, arm, splits, tok, limit=32, kind=kind)
            interventions[kind] = time.monotonic()-tick
        tick = time.monotonic(); model.export_adapter(root/'adapter'); save_s = time.monotonic()-tick
        del model; gc.collect(); torch.cuda.empty_cache()
        tick = time.monotonic(); reloaded = make_model(arm, root/'adapter', seed=0)
        _, after, _ = evaluate(reloaded, arm, splits, tok, limit=32); reload_s = time.monotonic()-tick
        if before != after:
            raise ValueError('smoke same-job reload identity failed')
        arms[arm] = {**m, 'load.seconds': load, 'eval32.seconds': eval_s, 'eval32.forward_seconds': audit['forward_seconds'],
                     'save.seconds': save_s, 'reload32.seconds': reload_s, 'reload.identical': True,
                     'interventions32.seconds': interventions}
        del reloaded, before, after; gc.collect(); torch.cuda.empty_cache()
    base = make_model('SEG-00')
    tick = time.monotonic(); evaluate(base, 'SEG-00', splits, tok, limit=32); base_eval = (time.monotonic()-tick)*1000/32
    del base; gc.collect(); torch.cuda.empty_cache()
    full = lambda seconds: seconds*1000/32
    projected = {}
    for arm, m in arms.items():
        eval_full = full(m['eval32.seconds'])
        # Startup, load, same-allocation Base_seg + initial, steady steps, final,
        # reload (load + eval), interventions, save. No curve evaluations.
        cost = (60+2*m['load.seconds']+base_eval+3*eval_full+m['train.steady_step_seconds']*len(batches_full)+
                sum(full(v) for v in m['interventions32.seconds'].values())+m['save.seconds'])
        projected[arm] = {'expected_seconds': cost, 'training_step_seconds': m['train.steady_step_seconds'],
                          'full_dev_seconds': eval_full}
    node_h = lambda seconds: seconds/14400  # one MIG instance at 1/4 node (constraints)
    seed0 = sum(node_h(p['expected_seconds']) for p in projected.values())
    replication = {'W': 2*node_h(projected['SEG-W1']['expected_seconds']),
                   'X': 2*sum(node_h(projected[a]['expected_seconds']) for a in X_TAP)}
    # Reserve: checks reruns, the replay, the rule-7 mechanism job and analyses.
    reserve = .12
    return {'smoke.steps': size, 'smoke.seed': 0, 'uniform_sample_indices': indices, 'arms': arms,
            'base_full_dev_seconds': base_eval,
            'projection': {'arms': projected, 'seed0_training_node_h': seed0, 'reserve_node_h': reserve,
                'seed0_total_before_smoke_usage_node_h': seed0+reserve, 'seed0_cap_node_h': SEED0_CAP_NODE_H,
                'replication_node_h': replication,
                'total_worst_case_before_smoke_usage_node_h': seed0+reserve+max(replication.values()),
                'total_cap_node_h': TOTAL_CAP_NODE_H,
                'seed0_gate': seed0+reserve <= SEED0_CAP_NODE_H,
                'total_gate': seed0+reserve+max(replication.values()) <= TOTAL_CAP_NODE_H,
                'walltime_rule': '1.5x measured part model, round up to 10 min'}}


def replay(data_dir, out):
    hardware = torch.cuda.get_device_name()
    if hardware != REPLAY['hardware']:
        raise ValueError('replay must run on the checkpoint training slice (2g.24gb)')
    manifest = read(REPLAY_RUN/'manifest.json')
    if (manifest.get('status') != 'complete' or manifest.get('arm') != REPLAY['arm'] or
            manifest.get('seed') != REPLAY['seed'] or manifest['metrics'].get('hardware') != REPLAY['hardware']):
        raise ValueError('ATT-0379-2 is not the complete 2g seed-3 SEG-05 run')
    checkpoint = read(REPLAY_RUN/'checkpoint.json')
    for name, ref in checkpoint.items():
        if reference(ref['path'])['sha256'] != ref['sha256']:
            raise ValueError('retained E-0030 checkpoint changed')
    tok = tokenizer(); splits = prepare(data_dir)
    model = make_model(REPLAY['arm'], REPLAY_RUN/'adapter', seed=REPLAY['seed'])
    tick = time.monotonic(); metrics, rows, _ = evaluate(model, REPLAY['arm'], splits, tok)
    recorded = read(REPLAY_RUN/'predictions.json')
    if [(r['task'], r['id'], r['gold']) for r in rows] != [(r['task'], r['id'], r['gold']) for r in recorded]:
        raise ValueError('replay rows are not paired with the recorded predictions')
    result = {'run': str(REPLAY_RUN), 'hardware': hardware, 'predictions_ref': reference(REPLAY_RUN/'predictions.json'),
              'checkpoint': checkpoint, 'rows': len(rows), 'rows_identical': rows == recorded,
              'prediction_flips': sum(a['prediction'] != b['prediction'] for a, b in zip(rows, recorded)),
              'score_rows_differing': sum(a['scores'] != b['scores'] for a, b in zip(rows, recorded)),
              'metrics': metrics, 'seconds': time.monotonic()-tick}
    write_json(out/'replay.json', result)
    write_json(out/'replay-predictions.json', rows)
    if not result['rows_identical']:
        raise ValueError('same-slice replay of the retained E-0030 checkpoint differs')
    return {'replay.rows_identical': True, 'replay.hardware': hardware, 'replay.seconds': result['seconds']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('smoke', 'checks', 'replay'), required=True)
    p.add_argument('--replay', action='store_true', help='smoke only: also run the same-slice replay')
    p.add_argument('--data-dir')
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--lr-lock', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, values, start = False, {}, time.monotonic()
    outputs = ('checks.xml', 'checks.log', 'write-masks.json', 'initial-digests.json', 'cost-smoke.json',
               'replay.json', 'replay-predictions.json')
    try:
        values['hardware'] = torch.cuda.get_device_name()
        if args.mode != 'replay':
            values.update(checks(out))
        if args.mode == 'smoke':
            tok = tokenizer()
            args.seed = 0
            splits, batches, metadata, _, lock = prepare_plan(args, tok)
            if metadata['plan.order_sha256'] != ORDER_SHA256:
                raise ValueError('seed-0 plan differs from E-0025')
            values['plan'] = metadata
            masks = mask_audit(tok, batches, splits)
            write_json(out/'write-masks.json', masks); values['write_masks'] = masks
            values['initial_digests'] = {str(s): d for s, d in digests().items()}
            write_json(out/'initial-digests.json', values['initial_digests'])
            smoke = timing(tok, out, batches, splits, lock)
            write_json(out/'cost-smoke.json', smoke)
            values.update({'smoke.all_arms': len(smoke['arms']), 'smoke.reload_identical': True,
                'smoke.no_dev_scores_retained': True, 'projection': smoke['projection']})
        if args.mode == 'replay' or args.replay:
            values.update(replay(args.data_dir, out))
        values['confirmation_evaluated'] = False
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0032@1',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'source_commit': os.environ['SR_CODE_COMMIT'],
            'stage': {'smoke': 'rule1_smoke', 'checks': 'deterministic_checks', 'replay': 'checkpoint_replay'}[args.mode],
            'metrics': values,
            'artifacts': [{'path': str(out/f), 'role': f} for f in outputs if (out/f).exists()],
            'validator': {'name': 'e0032_rule1_checks', 'passed': passed}})


if __name__ == '__main__':
    main()
