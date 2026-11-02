"""Artifact-bound E-0029 screening rules 1-8; execute only in PBS."""
import argparse
from fractions import Fraction
import os
from pathlib import Path

from rb1_analysis import (ACCEPTED_BASE_MANIFEST_SHA256, check_metrics, measured_cost,
                          validate_rows)
from rb1_artifacts import finite_tree, read, reference, validate_formal_run
from rb1_rules import TASKS, paired_bootstrap
from rb_eta_select import ORDER_SHA256, SPLIT_SHA256
from rb_improve_lr import verify_lock
from rb_tap_analysis import accuracy, pp
from rb_tap_identity import PINNED_PROFILE, reference_check
from rb_train import E0029_ARMS, IG_ARMS
from v5_train import write_json


SIDE_ARMS = E0029_ARMS[:6]
ONE_PP = Fraction(1, 100)


def any_gain(deltas):
    return 'supports' if any(d >= ONE_PP for d in deltas) else 'challenges' if all(d <= 0 for d in deltas) else 'inconclusive'


def rule_outcomes(scores, implementation_passed=True):
    if not implementation_passed:
        raise ValueError('implementation checks failed; no scientific outcomes')
    required = {*E0029_ARMS, 'Base', 'RB1-07', 'RB1-09'}
    if not required <= scores.keys():
        raise ValueError('required valid references are pending; withhold dependent comparisons')
    x_available = 'X-01' in scores
    up = {'T': scores['L-56']-scores['RB1-09']}
    ig = {arm: scores[arm]-scores[ref] for arm, ref in
          (('G-T', 'RB1-09'), ('G-QN', 'Q-N'))}
    pair = {'P-T': scores['P-T']-scores['RB1-09']}
    if x_available:
        up['N'] = scores['L-48']-scores['X-01']
        ig['G-N'] = scores['G-N']-scores['X-01']
        pair['P-N'] = scores['P-N']-scores['X-01']
    qk = scores['Q-N']-scores['RB1-09']
    lora = {arm: scores[arm]-scores['RB1-07'] for arm in SIDE_ARMS}
    h23 = ('challenges' if any(d >= ONE_PP for d in up.values()) else 'supports') if x_available else 'pending_X01'
    h24 = any_gain(ig.values()) if x_available else 'pending_X01'
    h25 = any_gain(pair.values()) if x_available else 'pending_X01'
    h26 = 'supports' if qk >= ONE_PP else 'challenges' if qk <= 0 else 'inconclusive'
    return {
        '1_implementation': {'passed': True, 'evidence_level': 'diagnostic',
            'X01_reference_status': 'valid' if x_available else 'pending_X01'},
        '2_learnability': {arm: {'minus_Base': pp(scores[arm]-scores['Base']),
            'at_least_Base_plus_1pp': scores[arm]-scores['Base'] >= ONE_PP} for arm in E0029_ARMS},
        '3_H0023': {'outcome': h23, 'd_up': {tap: pp(d) for tap, d in up.items()},
            'pending_comparisons': [] if x_available else ['L-48 - X-01'],
            'activate_H0027': h23 == 'challenges', 'L56_minus_RB107': pp(scores['L-56']-scores['RB1-07'])},
        '4_H0024': {'outcome': h24, 'd_IG': {arm: pp(d) for arm, d in ig.items()},
            'pending_comparisons': [] if x_available else ['G-N - X-01'],
            'activate_H0028': h24 == 'supports'},
        '5_H0025': {'outcome': h25, 'd_pair': {arm: pp(d) for arm, d in pair.items()},
            'pending_comparisons': [] if x_available else ['P-N - X-01']},
        '6_H0026': {'outcome': h26, 'd_QK': pp(qk),
            'd_V_descriptive': pp(scores['X-01']-scores['Q-N']) if x_available else None,
            'd_V_status': 'available' if x_available else 'pending_X01',
            'GQN_minus_GT_descriptive': pp(scores['G-QN']-scores['G-T'])},
        '7_LoRA': {'differences': {arm: pp(d) for arm, d in lora.items()},
            'new_two_seed_replication': [arm for arm, d in lora.items() if d >= ONE_PP],
            'LoRA_GA_required_in_replication': [arm for arm, d in lora.items() if d >= ONE_PP and arm in IG_ARMS]},
        '8_evidence': {'scientific_rules_level': 'screening', 'implementation_level': 'diagnostic',
            'hypothesis_status_changes': False}}


def bind_reference(arm, inventory, xcheck=None, diagnostic=None):
    if arm == 'X-01' and (xcheck is None or
            xcheck.get('sources', {}).get('X-01', {}).get('manifest') != inventory['manifest']):
        raise ValueError('X-01 differs from the run that passed E-0028 rule1')
    if diagnostic is not None and diagnostic.get('training_inputs') != inventory:
        raise ValueError('reference diagnostics replayed a different checkpoint')


def runtime_metrics(manifests):
    return {arm: {key: value for key, value in manifests[arm]['metrics'].items()
        if key.startswith(('train.', 'plan.', 'runtime.', 'params.', 'init.')) or
        key in ('learning_rate', 'hardware', 'source.commit', 'environment.lock_sha256')}
        for arm in E0029_ARMS}


def validate_reads(value):
    records = value.get('records', [])
    keys = set()
    for row in records:
        key = row['task'], row['layer'], row['scope'], tuple(row['prefix_bin'])
        if key in keys or key[0] not in TASKS or key[1] not in (6, 13, 20, 27) or key[2] not in ('all_valid_input', 'answer_prediction'):
            raise ValueError('unexpected or duplicate read diagnostic stratum')
        keys.add(key)
        low, high = key[3]
        means = row['means']
        if row['count'] <= 0 or not 1 <= low <= high <= 512:
            raise ValueError('invalid prefix bin/count')
        if not 0 <= means['self_weight'] <= means['maximum_weight'] <= 1.00001 or not .999 <= means['n_eff'] <= high + .001:
            raise ValueError('invalid read weight distribution')
    if {key[0] for key in keys} != set(TASKS) or {key[1] for key in keys} != {6, 13, 20, 27}:
        raise ValueError('incomplete read diagnostic tasks/layers')
    finite_tree(value)
    return value


def analyze(spec):
    # The calibration-only lock is reconstructed before any new dev is opened.
    lock = verify_lock(spec['up_lr_lock'])
    if reference(spec['up_lr_lock'])['sha256'] != spec['up_lr_lock_sha256']:
        raise ValueError('registered LoRA LR lock hash changed')
    source = spec['training_source_commit']
    if lock['identity']['source.commit'] != source:
        raise ValueError('calibration and training code differ')
    smoke = read(spec['smoke_manifest'])
    if (smoke.get('status') != 'complete' or smoke.get('validator', {}).get('passed') is not True or
            smoke.get('metrics', {}).get('source_commit') != source or smoke['metrics'].get('pytest_exit') != 0):
        raise ValueError('implementation tests do not cover the training source')
    xcheck = read(spec['e0028_analysis']) if spec.get('e0028_analysis') else None
    if xcheck is not None and xcheck.get('contract') != 'E-0028@1':
        raise ValueError('X-01 validation is not an E-0028 analysis')
    x_available = xcheck is not None and xcheck.get('rules', {}).get('1_implementation', {}).get('passed') is True
    reference_arms = ('RB1-07', 'RB1-09', 'X-01') if x_available else ('RB1-07', 'RB1-09')
    base = Path(spec['base_run'])
    if reference(base/'manifest.json')['sha256'] != ACCEPTED_BASE_MANIFEST_SHA256:
        raise ValueError('Base differs from locked E-0021 reference')
    rows = {'Base': validate_rows(read(base/'predictions.json'), 'Base')}
    manifests, sources, performance, curves, reads, checks, initial = {}, {}, {}, {}, {}, {}, {}
    performance['Base'] = check_metrics(rows['Base'], read(base/'manifest.json')['metrics'], 'Base')
    for arm in (*reference_arms, *E0029_ARMS):
        root = Path(spec['reference_runs'][arm] if arm in reference_arms else spec['training_runs'][arm])
        expected = spec['x01_source_commit'] if arm == 'X-01' else source if arm in E0029_ARMS else None
        manifest, refs, _ = validate_formal_run(root, arm, expected)
        manifests[arm], sources[arm] = manifest, refs
        if arm == 'X-01':
            bind_reference(arm, refs, xcheck=xcheck)
        rows[arm] = validate_rows(read(root/'predictions.json'), arm, rows['Base'])
        performance[arm] = check_metrics(rows[arm], manifest['metrics'], arm)
        if arm not in E0029_ARMS:
            continue
        m = manifest['metrics']
        if (m['plan.order_sha256'] != ORDER_SHA256 or m['split_manifest.sha256'] != SPLIT_SHA256 or
                m['plan.steps'] != 13115 or m['plan.examples'] != 26230 or
                m['up_lr_lock.sha256'] != spec['up_lr_lock_sha256']):
            raise ValueError('new arm order/split/budget/LoRA lock differs')
        registration = read(root/'initialization.json')
        if registration.get('passed') is not True or registration.get('sha256') != m['init.parameters.sha256'] or registration.get('timing_only_random_IG') is not False:
            raise ValueError('new arm initialization registration failed')
        initial_record = read(root/'dev-step-0.json')
        initial_rows = validate_rows(initial_record['predictions'], arm+'/init', rows['Base'])
        check_metrics(initial_rows, initial_record['metrics'], arm+'/init')
        local = None
        if m['hardware'] != PINNED_PROFILE:
            value = read(root/'dev-base-same-allocation.json')
            if value['hardware'] != m['hardware'] or value['source_commit'] != source or value['job_id'] != m['exec.pbs_job_id'] or value['job_id'] in (None, 'unknown'):
                raise ValueError('initial Base reference lacks same-allocation provenance')
            local = validate_rows(value['predictions'], arm+'/same-allocation-Base', rows['Base'])
            check_metrics(local, value['metrics'], arm+'/same-allocation-Base')
        initial[arm] = reference_check(initial_rows, rows['Base'], m['hardware'], same_allocation=local)
        if not initial[arm]['passed']:
            raise ValueError('new arm step0 differs from required Base')
        curves[arm] = {}
        for step in (3279, 6558, 9837):
            record = read(root/f'dev-step-{step}.json')
            curve_rows = validate_rows(record['predictions'], arm+f'/step{step}', rows['Base'])
            curves[arm][str(step)] = check_metrics(curve_rows, record['metrics'], arm+f'/step{step}')
        curves[arm]['13115'] = performance[arm]
        if arm in SIDE_ARMS:
            reads[arm] = {stage: validate_reads(read(root/f'{stage}-read.json')) for stage in ('step0', 'final')}
        if arm in IG_ARMS:
            init_root = Path(spec['ig_initializations'][arm])
            if reference(init_root/'manifest.json') != m['ig.initialization']:
                raise ValueError('I-G producer differs from training input')
            init_manifest = read(init_root/'manifest.json')
            details = read(init_root/'ig-initialization.json')
            if (init_manifest['metrics']['source.commit'] != source or init_manifest['metrics']['ig.sample_sha256'] != m['ig.sample_sha256'] or
                    details.get('passed') is not True or any(layer['relative_RMS_error'] > 1e-3 or
                    layer['unscaled_orthogonality_max_error'] > 1e-4 for layer in details['layers'].values())):
                raise ValueError('I-G basis/scale/provenance check failed')
            checks[arm] = details
        sources[arm].update({name: reference(root/name) for name in ('initialization.json', 'ig-samples.json', 'initial-identity.json')})
    if len({manifests[arm]['metrics']['ig.sample_sha256'] for arm in E0029_ARMS}) != 1:
        raise ValueError('arms used different training-side initialization/diagnostic samples')
    for arm in (('RB1-09', 'X-01') if x_available else ('RB1-09',)):
        diagnostic = read(spec['reference_diagnostics'][arm])
        if diagnostic['arm'] != arm or diagnostic['sample_sha256'] != manifests['Q-N']['metrics']['ig.sample_sha256']:
            raise ValueError('reference diagnostics use different examples')
        bind_reference(arm, sources[arm], xcheck=xcheck, diagnostic=diagnostic)
        reads[arm] = {stage: validate_reads(diagnostic[stage]) for stage in ('step0', 'final')}
    scores = {arm: accuracy(value) for arm, value in rows.items()}
    rules = rule_outcomes(scores)
    rules['1_implementation'].update(initial_references=initial, initialization_checks=checks,
        smoke=reference(spec['smoke_manifest']),
        X01_rule1=reference(spec['e0028_analysis']) if xcheck is not None else None)
    arrays = {arm: {task: [int(row['gold'] == row['prediction']) for row in value if row['task'] == task]
                   for task in TASKS} for arm, value in rows.items()}
    result = {'contract': 'E-0029@1', 'rules': rules, 'performance': performance, 'curves': curves,
        'comparison_status': 'complete' if x_available else 'pending_X01',
        'runtime': runtime_metrics(manifests),
        'read_diagnostics': reads, 'sources': sources, 'LoRA_LR_lock': lock,
        'bootstrap': {'repeats': 10000, 'seed': 20261101, 'comparisons': paired_bootstrap(arrays),
            'scope': 'descriptive paired task-stratified resampling; excludes seed and selection uncertainty'},
        'measured_cost': measured_cost(spec.get('observations', [])),
        'limitations': ['seed0 screening', 'dev used for decisions', '1000 dev examples/task',
            'multiple tap comparisons in H-0024/H-0025', 'LoRA has no LoRA-GA initialization',
            'eta fixed0.25; side LR inherited1e-3', 'A2 and official validation unopened']}
    finite_tree(result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = analyze(read(args.input))
    write_json(args.output/'analysis.json', result)
    write_json(args.output/'manifest.json', {'status': 'complete' if result['comparison_status'] == 'complete' else 'partial',
        'contract': 'E-0029@1',
        'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': {'implementation_passed': True, 'rules': result['rules']},
        'artifacts': [{'path': str(args.output/'analysis.json'), 'role': 'analysis'}],
        'validator': {'name': 'e0029_locked_analysis', 'passed': True}})
