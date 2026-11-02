"""PBS checks and score-discarding E-0027 smoke for new seeds 1 and 2.

mode=smoke: deterministic tests; seed 0/1/2 plans (seed 0 must reproduce
E-0025's order hash; seed 1/2 hashes are printed for registration); initial
parameter digests (SEG-05/06/07 identical within a seed); a 1% uniform-step
timing rehearsal of SEG-04..07 at seed 1 with 32 dev rows per task; and a
cost projection against the 3.2 node-h cap. No dev quality is retained.
mode=checks: deterministic tests only, at the exact formal source.
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

from seg_data import ORDER_SHA256, ORDER_SHA256_BY_SEED
from seg_eval import evaluate
from seg_model import make_model, KEEP, PARAMETERS, FAMILIES, parameter_digest
from seg_train import tokenizer, prepare_plan, train
from v5_train import write_json

FILES = ['src/test_seg_seeds.py', 'src/test_seg_rules.py', 'src/test_seg_data.py', 'src/test_seg_model.py',
         'src/test_seg_artifacts.py', 'src/test_seg_analysis.py', 'src/test_core.py', 'src/test_rb_core.py']
NEW_ARMS = ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07')
CAP_NODE_H = 3.2


def checks(out):
    with (out/'checks.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'pytest', *FILES, '-q', '-p', 'no:cacheprovider',
            '-o', 'junit_family=legacy', '--junitxml='+str(out/'checks.xml')], stdout=log, stderr=subprocess.STDOUT)
    cases = list(ET.parse(out/'checks.xml').getroot().iter('testcase'))
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    if result.returncode != 0 or failed or skipped or not cases:
        raise RuntimeError('E-0027 PBS deterministic checks failed; see checks.log/checks.xml')
    # Passing seeded no-channel cases (rule 1(c)); ids carry 'seed1'/'seed2'.
    seeded = [c.get('name', '') for c in cases if c.get('name', '').startswith('test_no_channel_identity_at_new_seed')]
    seeds = sorted({s for s in (1, 2) if any(f'seed{s}' in n for n in seeded)})
    return {'checks.cases': len(cases), 'checks.failed': failed, 'checks.skipped': skipped,
            'no_channel_identity': seeds == [1, 2], 'no_channel_seeds': seeds,
            'zero_initialization_identity': True, 'synthetic_rule_boundaries': True}


def plans(args, tok):
    result, kept = {}, None
    for seed in (0, 1, 2):
        args.seed = seed
        splits, batches, metadata, _, lock = prepare_plan(args, tok, require_registered=False)
        if seed == 1:
            kept = splits, batches, lock
        result[seed] = {**metadata,
            'segment1_tokens_keep': sum(len(e['segment1_ids']) for b in batches for e in b),
            'segment2_tokens': sum(len(e['input_ids']) for b in batches for e in b),
            'loss_tokens': sum(sum(x != -100 for x in e['labels'][1:]) for b in batches for e in b),
            'registered': ORDER_SHA256_BY_SEED[seed]}
        if ORDER_SHA256_BY_SEED[seed] not in (None, metadata['plan.order_sha256']):
            raise ValueError(f'seed {seed} order differs from its registered hash')
    if result[0]['plan.order_sha256'] != ORDER_SHA256:
        raise ValueError('fixed-step plan does not reproduce E-0025 seed 0')
    if len({r['plan.order_sha256'] for r in result.values()}) != 3:
        raise ValueError('seed orders are not distinct')
    return result, kept


def digests():
    result = {}
    for seed in (0, 1, 2):
        result[seed] = {}
        for arm in NEW_ARMS:
            model = make_model(arm, seed=seed)
            result[seed][arm] = parameter_digest(model)
            del model; gc.collect(); torch.cuda.empty_cache()
        if not result[seed]['SEG-05'] == result[seed]['SEG-06'] == result[seed]['SEG-07']:
            raise ValueError(f'seed {seed} r128 initial parameters differ across SEG-05/06/07')
    if len({result[s]['SEG-05'] for s in result}) != 3 or len({result[s]['SEG-04'] for s in result}) != 3:
        raise ValueError('new seeds did not change the initialization')
    return result


def timing(tok, out, plan_info, seed1):
    splits, full_plan, lock = seed1
    size = math.ceil(len(full_plan)*.01)
    indices = np.linspace(0, len(full_plan)-1, size, dtype=int).tolist()
    batches = [full_plan[i] for i in indices]
    arms = {}
    for arm in NEW_ARMS:
        root = out/arm; root.mkdir(parents=True, exist_ok=True)
        tick = time.monotonic(); model = make_model(arm, seed=1); load = time.monotonic()-tick
        m = train(model, arm, batches, tok, lock['locked_lrs'][FAMILIES[arm]], root)
        if m['params.with_grad'] != PARAMETERS[arm]:
            raise ValueError('smoke parameter-gradient coverage differs')
        tick = time.monotonic(); _, before, audit = evaluate(model, arm, splits, tok, limit=32)
        eval_s = time.monotonic()-tick
        interventions = {}
        if arm in KEEP:
            for kind in ('iv_off', 'iv_swap', 'iv_reset'):
                tick = time.monotonic(); evaluate(model, arm, splits, tok, limit=32, kind=kind)
                interventions[kind] = time.monotonic()-tick
        tick = time.monotonic(); model.export_adapter(root/'adapter'); save_s = time.monotonic()-tick
        del model; gc.collect(); torch.cuda.empty_cache()
        tick = time.monotonic(); reloaded = make_model(arm, root/'adapter', seed=1)
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
    projected = {}
    for arm, m in arms.items():
        eval_full = m['eval32.seconds']*1000/32
        # Load, steady steps, three curve evals + final + reload, save; R_keep adds
        # the same-job Base_seg and initial evaluations.
        cost = 45+m['load.seconds']+m['train.steady_step_seconds']*len(full_plan)+5*eval_full+m['save.seconds']
        if arm in KEEP:
            cost += base_eval+eval_full
        projected[arm] = {'expected_seconds': cost, 'training_step_seconds': m['train.steady_step_seconds'],
                          'full_dev_seconds': eval_full}
    keep = arms['SEG-05']
    intervention = 45+keep['load.seconds']+base_eval+projected['SEG-05']['full_dev_seconds']+sum(
        v*1000/32 for v in keep['interventions32.seconds'].values())
    per_seed = sum(p['expected_seconds'] for p in projected.values())+intervention
    # MIG instance seconds -> node-h at 1/4 (constraints): seconds/3600*0.25.
    node_h = lambda seconds: seconds/14400
    total = 2*node_h(per_seed)+node_h(60+90)+.05
    return {'smoke.steps': size, 'smoke.seed': 1, 'uniform_sample_indices': indices, 'arms': arms,
            'projection': {'arms_seconds': {a: p['expected_seconds'] for a, p in projected.items()},
                'arms': projected, 'intervention_seconds': intervention, 'per_seed_node_h': node_h(per_seed),
                'two_seeds_node_h': 2*node_h(per_seed), 'checks_analysis_reserve_node_h': node_h(150)+.05,
                'total_before_smoke_usage_node_h': total, 'cap_node_h': CAP_NODE_H, 'cost_gate': total <= CAP_NODE_H,
                'walltime_rule': '1.5x measured part model, round up to 10 min'},
            'plans': {str(k): {kk: vv for kk, vv in v.items() if kk != 'plan.epoch_end_steps'} for k, v in plan_info.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('smoke', 'checks'), required=True)
    p.add_argument('--data-dir')
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--lr-lock', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, values, start = False, {}, time.monotonic()
    try:
        values.update(checks(out))
        if not values['no_channel_identity']:
            raise RuntimeError('seeded no-channel identity tests did not cover both new seeds')
        if args.mode == 'smoke':
            tok = tokenizer()
            plan_info, seed1 = plans(args, tok)
            values['plans'] = {str(s): {k: v for k, v in r.items()} for s, r in plan_info.items()}
            write_json(out/'seed-plans.json', values['plans'])
            values['initial_digests'] = {str(s): d for s, d in digests().items()}
            write_json(out/'initial-digests.json', values['initial_digests'])
            smoke = timing(tok, out, plan_info, seed1)
            write_json(out/'cost-smoke.json', smoke)
            values.update({'smoke.all_arms': len(smoke['arms']), 'smoke.reload_identical': True,
                'smoke.no_dev_scores_retained': True, 'projection.node_h': smoke['projection']['total_before_smoke_usage_node_h'],
                'projection.within_cap_before_actual_smoke': smoke['projection']['cost_gate']})
        values['confirmation_evaluated'] = False
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0027@1',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'source_commit': os.environ['SR_CODE_COMMIT'],
            'stage': 'seed_smoke' if args.mode == 'smoke' else 'deterministic_checks', 'metrics': values,
            'artifacts': [{'path': str(f), 'role': f.name} for f in (out/'checks.xml', out/'checks.log',
                out/'seed-plans.json', out/'initial-digests.json', out/'cost-smoke.json') if f.exists()],
            'validator': {'name': 'segmented_seed_checks', 'passed': passed}})


if __name__ == '__main__':
    main()
