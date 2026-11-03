"""E-0032 analysis: authenticate runs, apply rules 1-6, descriptive intervals and costs.

Run in PBS. References (E-0025 seed 0, E-0027 seeds 1-2) are read from their
retained prediction files, bound by SHA256 below. Intervals are descriptive
per-seed paired task-stratified bootstraps (10,000 draws, seed 20261101);
for gain and delta-gain the max control is recomputed per draw.
"""
import argparse
import math
import os
from pathlib import Path
import time

import numpy as np

from rb1_analysis import measured_cost
from rb1_artifacts import read, reference, finite_tree
from seg_artifacts import validate_rows
from seg_model import PARAMETERS, E0032_ARMS, WRITE_RULE, X_TAP, SEG05_INITIAL_SHA256, KEEP, T0
from seg_w_rules import analyze, VARIANTS, CONTROLS, REFERENCE, number
from v5_train import write_json

CONTRACT = 'E-0032@1'
RUNS = '/work/xg24i002/x10041/bypass-lora/runs/'
# (path, sha256); 'SEG-05/iv_swap' is the iv_swap condition of the intervention job.
REFERENCE_FILES = {
    0: {'SEG-00': ('ATT-0277-1/predictions.json', '57b91f8a3a7e7f9f4f742d96f36b32f6526cd9a203338ca58e663e34ca50e91a'),
        'SEG-04': ('ATT-0281-1/predictions.json', '3dc26446d8c4d5b928e360424bd40fa0e439a0d18fd63a7eff238e5969eac797'),
        'SEG-05': ('ATT-0290-1/predictions.json', '32338ce996fffa83950065afec3903fc3a259232ef513780dc885e2c786b99fb'),
        'SEG-06': ('ATT-0291-1/predictions.json', '500077057e5b56059790d545c20f35ac35d8d708a76beecb32d4e1afd3ea430e'),
        'SEG-07': ('ATT-0292-1/predictions.json', '0b0271443d1a704e8810ecbb4513aca0d631519a7930cb68641e343d62287e33'),
        'SEG-05/iv_swap': ('ATT-0302-1/conditions.json', 'dd6b923dacef6cb0238359c773026baace62e6636311915928c6a7d3602b8819')},
    1: {'SEG-04': ('ATT-0306-1/predictions.json', '620bcc391ac3e1a3b319d4c230f8ff694724f4d188d7b0a3f2443cea8305d985'),
        'SEG-05': ('ATT-0307-1/predictions.json', 'bc0266047657dfbbf6dd0aa630aae953f472937d4635c9c128c983419616c7c0'),
        'SEG-06': ('ATT-0308-1/predictions.json', 'ad9084c3595f23dfda8a40ee43120492939a4d38b015d7a6afabaade83aac306'),
        'SEG-07': ('ATT-0309-1/predictions.json', '2d12c7179c5511b29a92c5a86afe071c0776137cd40beceb29b990b2c8dc2116'),
        'SEG-05/iv_swap': ('ATT-0319-1/conditions.json', 'd8e2e0e0af18b7bd2251b3417be46c4ecc14578c1ce050e34e2e348518223d16')},
    2: {'SEG-04': ('ATT-0310-1/predictions.json', 'a2cdd2a6fc615928e8855bb966da933ba20503b7111c020d9beb08fcd08b1731'),
        'SEG-05': ('ATT-0311-1/predictions.json', '04ae1ac17076800d5b41aafeb9467165b7b08555674f05db4c4f2f7442a15921'),
        'SEG-06': ('ATT-0312-1/predictions.json', 'd2eab4f0eb174c4d65b096809daa8182ed22fe735a52d61c2079a23ebc00433f'),
        'SEG-07': ('ATT-0313-1/predictions.json', '62d2f579f75cc5b468a73e5c41270f7f9b7a2f6ed65c2ed58d70636d9f07c2a7'),
        'SEG-05/iv_swap': ('ATT-0320-1/conditions.json', 'fa5d96f14a35d893e09d160c2ce8ccb18f70ee42e2f55fc4b449665a7a1290ac')}}
REFERENCE_FILES[1]['SEG-00'] = REFERENCE_FILES[2]['SEG-00'] = REFERENCE_FILES[0]['SEG-00']
TASKS = ('piqa', 'boolq')
BOOTSTRAP_REPEATS, BOOTSTRAP_SEED = 10000, 20261101


def pinned(relative, sha256):
    ref = reference(RUNS+relative)
    if ref['sha256'] != sha256:
        raise ValueError(f'reference {relative} changed')
    return ref


def reference_rows(seed, order):
    rows, refs = {}, {}
    for label, (relative, sha256) in REFERENCE_FILES[seed].items():
        refs[label] = pinned(relative, sha256)
        data = read(refs[label]['path'])
        rows[label] = validate_rows(data['iv_swap']['predictions'] if label.endswith('iv_swap') else data, order)
    return rows, refs


def run_rows(root, arm, seed, source, order):
    """Authenticate one E-0032 training run; returns rows, swap rows, rule-1 checks, refs."""
    root = Path(root).resolve()
    manifest = read(root/'manifest.json'); m = manifest['metrics']; finite_tree(manifest)
    if (manifest.get('contract') != CONTRACT or manifest.get('status') != 'complete' or
            manifest.get('validator', {}).get('passed') is not True or manifest.get('arm') != arm or
            manifest.get('seed') != seed or manifest.get('stage') != 'train' or m.get('source.commit') != source):
        raise ValueError(f'{root} is not a complete E-0032 {arm} seed-{seed} run at {source}')
    refs = {name: reference(root/name) for name in ('manifest.json', 'predictions.json', 'checkpoint.json')}
    listed = {a['role']: a for a in manifest['artifacts']}
    for name in ('predictions.json', 'checkpoint.json', 'initial-predictions.json', 'initial-base-predictions.json',
                 'iv-off-dev.json'):
        if name not in listed or reference(root/name)['sha256'] != listed[name]['sha256']:
            raise ValueError(f'{root}/{name} differs from its manifest')
    rows = validate_rows(read(root/'predictions.json'), order)
    base = read(root/'initial-base-predictions.json')
    checks = {'initial_matches_base': m.get('initial.identical_to_Base_seg') is True and
                                      read(root/'initial-predictions.json') == base,
              'off_matches_base': m.get('off.identical_to_Base_seg') is True and
                                  read(root/'iv-off-dev.json')['predictions'] == base,
              'reload_identical': m.get('reload.identical') is True,
              'parameters': m.get('params.stored') == m.get('params.trainable') == m.get('params.with_grad') == PARAMETERS[arm],
              'executed_phases': set(m['runtime.accounting']) == ({'segment1', 'segment2'} if arm in KEEP
                                                                  else {'segment2'})}
    # seg_train convention: a T0 branch has no state, so it records 0 writes/reads.
    seg2 = m['train.segment2_tokens']; io = 0 if arm in T0 else seg2
    checks['segment2_accounting'] = all(c['valid_tokens'] == seg2 and c['writes'] == c['reads'] == io
                                        for c in m['runtime.accounting']['segment2'].values())
    if arm in WRITE_RULE:
        checks['initial_equals_SEG05'] = m.get('initial.adapter_sha256') == SEG05_INITIAL_SHA256[seed]
        train_writes = m['train.segment1_writes']
        checks['segment1_writes_match_mask'] = all(c['writes'] == train_writes['total'] and
            c['reads'] == m['train.segment1_tokens'] for c in m['runtime.accounting']['segment1'].values())
        if arm == 'SEG-W1':
            checks['W1_one_write_per_example'] = (train_writes['min'] == train_writes['max'] == 1 and
                                                  m['dev.segment1_writes']['min'] == m['dev.segment1_writes']['max'] == 1)
    swap = None
    if arm in KEEP:
        if 'iv-swap-dev.json' not in listed or reference(root/'iv-swap-dev.json')['sha256'] != listed['iv-swap-dev.json']['sha256']:
            raise ValueError(f'{root} lacks its in-job iv_swap')
        swap = validate_rows(read(root/'iv-swap-dev.json')['predictions'], order)
        refs['iv-swap-dev.json'] = reference(root/'iv-swap-dev.json')
    describe = {k: m.get(k) for k in ('hardware', 'exec.pbs_job_id', 'initial.adapter_sha256', 'train.seconds',
        'train.steady_step_seconds', 'train.peak_cuda_bytes', 'elapsed_s', 'train.segment1_writes',
        'dev.segment1_writes', 'dev.boolq.accuracy', 'dev.piqa.accuracy', 'dev.boolq.nll', 'dev.piqa.nll')}
    return rows, swap, checks, refs, describe


def counts(rows):
    result = {}
    for task in TASKS:
        selected = [r for r in rows if r['task'] == task]
        result[task] = (sum(r['prediction'] == r['gold'] for r in selected), len(selected))
    return result


def binary(rows):
    return {t: np.array([float(r['prediction'] == r['gold']) for r in rows if r['task'] == t]) for t in TASKS}


def intervals(rows_by_label, variants, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED, chunk=128):
    """Per-seed descriptive intervals of delta, gain, delta_gain, lora, swap (pp), BoolQ and PIQA."""
    arrays = {label: binary(rows) for label, rows in rows_by_label.items()}
    sizes = {t: len(arrays['SEG-05'][t]) for t in TASKS}
    names = [(V, q) for V in variants for q in ('delta', 'gain', 'delta_gain', 'lora', 'swap')]
    samples = {(V, q, t): np.empty(repeats) for V, q in names for t in TASKS}
    rng = np.random.default_rng(seed)
    for start in range(0, repeats, chunk):
        end = min(start+chunk, repeats)
        draws = {t: rng.integers(0, sizes[t], (end-start, sizes[t])) for t in TASKS}
        for t in TASKS:
            s = {label: arrays[label][t][draws[t]].mean(1) for label in arrays}
            ref_gain = s['SEG-05']-np.maximum(s['SEG-06'], s['SEG-07'])
            for V in variants:
                gain = s[V]-np.maximum(*(s[c] for c in CONTROLS[V]))
                values = {'delta': s[V]-s['SEG-05'], 'gain': gain, 'delta_gain': gain-ref_gain,
                          'lora': s[V]-s['SEG-04'], 'swap': s[V]-s[V+'/iv_swap']}
                for q, v in values.items():
                    samples[V, q, t][start:end] = v
    return {V: {q: {t: [float(100*x) for x in np.quantile(samples[V, q, t], [.025, .975])] for t in TASKS}
                for q in ('delta', 'gain', 'delta_gain', 'lora', 'swap')} for V in variants}


def run(args):
    spec = read(args.input)
    if spec.get('contract') != CONTRACT:
        raise ValueError('analysis input is not for E-0032@1')
    order, values = None, {'input': reference(args.input)}
    rule1 = {}
    # Rule-1 checks outside the training runs: smoke (tests, masks, init digests) and replay,
    # both bound to their stage and to the declared rule-1 source (checked against the
    # training source in the conclusion review; both are reported).
    smoke = read(spec['smoke_manifest']); replay = read(spec['replay_manifest'])
    rule1_source = spec['rule1_source']
    rule1['smoke_checks_passed'] = (smoke.get('status') == 'complete' and smoke['validator']['passed'] is True and
                                    smoke.get('stage') == 'rule1_smoke' and smoke.get('source_commit') == rule1_source and
                                    smoke['metrics'].get('checks.failed') == 0 and smoke['metrics'].get('checks.cases', 0) > 0)
    rule1['replay_rows_identical'] = (replay.get('status') == 'complete' and replay['validator']['passed'] is True and
                                      replay.get('stage') in ('checkpoint_replay', 'rule1_smoke') and
                                      replay.get('source_commit') == rule1_source and
                                      replay['metrics'].get('replay.hardware') == 'NVIDIA GH200 120GB MIG 2g.24gb' and
                                      replay['metrics'].get('replay.rows_identical') is True)
    values['rule1_sources'] = {'smoke': reference(spec['smoke_manifest']), 'replay': reference(spec['replay_manifest']),
                               'rule1_source': rule1_source, 'training_source': spec['training_source']}
    all_counts, rows_by_seed, provenance, describe = {}, {}, {}, {}
    for seed in (0, 1, 2):
        base_rows = read(RUNS+REFERENCE_FILES[seed]['SEG-00'][0])
        order = [(r['task'], r['id'], r['gold']) for r in base_rows]
        rows, refs = reference_rows(seed, order)
        provenance[str(seed)] = {'reference': refs, 'runs': {}}
        for arm, root in spec['runs'].get(str(seed), {}).items():
            if arm not in E0032_ARMS:
                raise ValueError('unknown E-0032 arm in analysis input')
            arm_rows, swap, checks, refs, info = run_rows(root, arm, seed, spec['training_source'], order)
            rows[arm] = arm_rows
            if swap is not None:
                rows[arm+'/iv_swap'] = swap
            rule1.update({f'seed{seed}.{arm}.{k}': v for k, v in checks.items()})
            provenance[str(seed)]['runs'][arm] = {'root': str(Path(root).resolve()), 'refs': refs}
            describe.setdefault(str(seed), {})[arm] = info
        x_digests = {describe.get(str(seed), {}).get(a, {}).get('initial.adapter_sha256') for a in X_TAP
                     if a in spec['runs'].get(str(seed), {})}
        if x_digests:
            rule1[f'seed{seed}.X_triplet_initial_identical'] = len(x_digests) == 1
        rows_by_seed[seed] = rows
        all_counts[seed] = {label: counts(r) for label, r in rows.items()}
    result = analyze(all_counts, rule1)
    boot = {str(s): intervals(rows_by_seed[s], [V for V in VARIANTS if V in rows_by_seed[s]]) for s in (0, 1, 2)
            if any(V in rows_by_seed[s] for V in VARIANTS)}
    accuracy = {str(s): {label: {t: number(__import__('fractions').Fraction(*c[t])) for t in TASKS}
                         for label, c in all_counts[s].items()} for s in (0, 1, 2)}
    cost = measured_cost(spec['observations']) if spec.get('observations') else None
    mechanism = None
    if spec.get('mechanism'):
        mechanism = {'ref': reference(spec['mechanism']), 'models': read(spec['mechanism'])['models']}
    report = {'contract': CONTRACT, 'stage': spec.get('stage', 'screening'), **result, 'accuracy': accuracy,
              'bootstrap': {'draws': BOOTSTRAP_REPEATS, 'seed': BOOTSTRAP_SEED, 'pp': boot,
                            'scope': 'descriptive; paired rows within seed; task-stratified; max control per draw'},
              'runs': describe, 'provenance': provenance, 'mechanism': mechanism, 'cost': cost,
              'limitations': ['dev both screens and judges', 'exact point-estimate gates',
                              'intervals exclude seed and selection uncertainty']}
    write_json(args.output/'analysis.json', report)
    values.update({'outcomes': result['outcomes'], 'selected': result['rules']['4']['selected'],
                   'implementation_ok': result['implementation_ok']})
    if cost:
        values['cost.node_h'] = cost['node_h']
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--data-dir')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    passed, values, start = False, {}, time.monotonic()
    try:
        values = run(args)
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        write_json(args.output/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': CONTRACT,
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'source_commit': os.environ['SR_CODE_COMMIT'],
            'stage': 'analysis', 'metrics': values,
            'artifacts': [{'path': str(args.output/'analysis.json'), 'role': 'analysis.json'}]
            if (args.output/'analysis.json').exists() else [],
            'validator': {'name': 'e0032_analysis', 'passed': passed}})


if __name__ == '__main__':
    main()
