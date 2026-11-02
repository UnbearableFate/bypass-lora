"""PBS deterministic checks and score-discarding E-0025 cost rehearsal.

All seven possible training arms use one common 1% representative-step smoke
(at least 32 dev rows per task for evaluation timing). No SEG dev quality
metrics or predictions are retained. The cost gate includes the r128 branch.
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

from seg_model import make_model, KEEP, PARAMETERS, FAMILIES
from seg_eval import evaluate
from seg_train import tokenizer, prepare_plan, train
from v5_train import write_json

FILES = ['src/test_seg_rules.py', 'src/test_seg_data.py', 'src/test_seg_model.py',
         'src/test_seg_artifacts.py', 'src/test_seg_analysis.py',
         'src/test_core.py', 'src/test_rb_core.py',
         'src/test_rb_model.py::test_default_paths_bitwise_match_task_master']


def checks(out):
    with (out/'checks.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'pytest', *FILES, '-q', '-p', 'no:cacheprovider',
            '-o', 'junit_family=legacy', '--junitxml='+str(out/'checks.xml')], stdout=log, stderr=subprocess.STDOUT)
    cases = list(ET.parse(out/'checks.xml').getroot().iter('testcase'))
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    if result.returncode != 0 or failed or skipped or not cases:
        raise RuntimeError('E-0025 PBS deterministic checks failed; see checks.log/checks.xml')
    return {'checks.cases': len(cases), 'checks.failed': failed, 'checks.skipped': skipped,
            'no_channel_identity': True, 'default_identity': True,
            'differentiable_state_carry': True, 'synthetic_reducer_boundaries': True}


def smoke(args, out):
    tok = tokenizer()
    splits, full_plan, original, excluded, lock = prepare_plan(args, tok)
    # Uniform deterministic coverage of the entire locked order, no length or
    # score selection; all arms use exactly these same steps.
    size = math.ceil(len(full_plan)*.01)
    indices = np.linspace(0, len(full_plan)-1, size, dtype=int).tolist()
    batches = [full_plan[i] for i in indices]
    values = {'smoke.steps': size, 'smoke.dev_per_task': 32, 'plan': original, 'arms': {},
              'quality_metrics_discarded': True, 'confirmation_evaluated': False}
    for arm in FAMILIES:
        root = out/arm; root.mkdir(parents=True, exist_ok=True)
        tick = time.monotonic(); model = make_model(arm)
        load = time.monotonic()-tick
        m = train(model, arm, batches, tok, lock['locked_lrs'][FAMILIES[arm]], root)
        if m['params.with_grad'] != PARAMETERS[arm]:
            raise ValueError('smoke parameter-gradient coverage differs')
        tick = time.monotonic(); _, before, normal_audit = evaluate(model, arm, splits, tok, limit=32)
        eval_s = time.monotonic()-tick
        interventions = {}
        if arm in KEEP:
            for kind in ('iv_off', 'iv_swap', 'iv_reset'):
                tick = time.monotonic(); _, discarded, audit = evaluate(model, arm, splits, tok, limit=32, kind=kind)
                interventions[kind] = {'seconds': time.monotonic()-tick,
                                       'forward_seconds': audit['forward_seconds'],
                                       'donor_forward_seconds': audit['donor_forward_seconds']}
                del discarded
        tick = time.monotonic(); model.export_adapter(root/'adapter')
        save_s = time.monotonic()-tick
        del model; gc.collect(); torch.cuda.empty_cache()
        tick = time.monotonic(); reloaded = make_model(arm, root/'adapter')
        _, after, _ = evaluate(reloaded, arm, splits, tok, limit=32)
        reload_s = time.monotonic()-tick
        if before != after:
            raise ValueError('smoke same-job fresh-base reload identity failed')
        # Retain timing/accounting, never the dev quality or predictions.
        values['arms'][arm] = {**m, 'load.seconds': load, 'eval32.seconds': eval_s,
            'eval32.forward_seconds': normal_audit['forward_seconds'], 'save.seconds': save_s,
            'reload32.seconds': reload_s, 'reload.identical': True, 'interventions32': interventions}
        write_json(out/'cost-smoke.json', values)
        del reloaded, before, after; gc.collect(); torch.cuda.empty_cache()
    model = make_model('SEG-00')
    tick = time.monotonic(); evaluate(model, 'SEG-00', splits, tok, limit=32)
    base_eval = (time.monotonic()-tick)*1000/32
    projected = {}
    for arm, m in values['arms'].items():
        # Fixed-cost preprocessing/loading plus target-scale representative
        # steady steps, three curve evals, final eval and full reload eval.
        eval_full = m['eval32.seconds']*1000/32
        cost = 45+m['load.seconds']+m['train.steady_step_seconds']*len(full_plan)+5*eval_full+m['save.seconds']
        if arm in KEEP:
            # Independent Base_seg plus initialized R_keep, and the frozen
            # intervention job's normal, Base, off, swap and reset conditions.
            cost += base_eval+eval_full
            cost += 45+m['load.seconds']+base_eval+eval_full+sum(v['seconds']*1000/32 for v in m['interventions32'].values())
        projected[arm] = {'expected_seconds': cost, 'expected_node_h': cost/14400,
                          'training_step_seconds': m['train.steady_step_seconds'], 'full_dev_seconds': eval_full}
    # Base_seg job, analysis, submission/test/review headroom. Add the actual
    # smoke attempt's terminal ledger usage outside this runtime estimate.
    initial = sum(projected[a]['expected_node_h'] for a in ('SEG-01', 'SEG-02', 'SEG-03', 'SEG-04'))
    conditional = sum(projected[a]['expected_node_h'] for a in ('SEG-05', 'SEG-06', 'SEG-07'))
    overhead = (base_eval+90)/14400+.05
    values['projection'] = {'arms': projected, 'initial_node_h': initial, 'conditional_node_h': conditional,
        'base_analysis_review_reserve_node_h': overhead, 'total_before_smoke_usage_node_h': initial+conditional+overhead,
        'cap_node_h': 3., 'cost_gate': initial+conditional+overhead <= 3.,
        'walltime_rule': 'per formal job 1.5x measured part model, round up to 10 min',
        'all_training_steps': len(full_plan), 'uniform_sample_indices': indices}
    write_json(out/'cost-smoke.json', values)
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--lr-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, values, start = False, {}, time.monotonic()
    try:
        values.update(checks(out))
        result = smoke(args, out)
        values.update({'smoke.all_arms': len(result['arms']), 'smoke.reload_identical': True,
            'smoke.no_dev_scores_retained': True, 'projection.node_h': result['projection']['total_before_smoke_usage_node_h'],
            'projection.within_cap_before_actual_smoke': result['projection']['cost_gate'],
            'confirmation_evaluated': False})
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0025@1',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'source_commit': os.environ['SR_CODE_COMMIT'],
            'stage': 'deterministic_and_cost_smoke', 'metrics': values,
            'artifacts': [{'path': str(f), 'role': f.name} for f in (out/'checks.xml', out/'checks.log', out/'cost-smoke.json') if f.exists()],
            'validator': {'name': 'segmented_pre_review_smoke', 'passed': passed}})


if __name__ == '__main__':
    main()
