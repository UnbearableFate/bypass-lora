"""Artifact-bound E-0026 r128 width-probe analysis (pre-registered rules 1-8). Run only in PBS.

Input JSON:
  expected: {source_commit, lr_lock_sha256, base_manifest_sha256}
  base_run: accepted ATT-0024-1 Base run
  registration: width-registration.json from rb_width_check.py
  training_runs: {W-01, W-02, W-03: formal E-0026 run}
  intervention_run: W-02 frozen run (iv_off, iv_swap, iv_permute)
  reference_runs: {RB1-01, RB1-02, RB1-07: accepted E-0021 runs} (descriptive)
  observations: optional list of miyabi-mcp ledger exports for measured cost
Point decisions use exact correctness fractions; intervals are descriptive.
The module assigns no hypothesis status.
"""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path

from rb1_analysis import (ACCEPTED_BASE_MANIFEST_SHA256, check_metrics, measured_cost,
                          normal_training_agreement, summarize_rows, validate_read_summary,
                          validate_rows)
from rb1_artifacts import (ACCEPTED_LR_LOCK_SHA256, finite_tree, read, reference,
                           validate_formal_run)
from rb1_intervene import WIDTH_MODES
from rb1_rules import TASKS, counts_from_rows, paired_bootstrap, paired_interval
from rb_train import LORA_RANKS, PARAMETERS, WIDTH_ARMS
from v1_analysis import _read_run
from v5_train import write_json


CONTRACT = 'E-0026@1'
REFERENCES = {'W-01': 'RB1-01', 'W-02': 'RB1-02', 'W-03': 'RB1-07'}
ONE_PP = Fraction(1, 100)
STATE_BYTES_PER_LAYER = (128*128 + 128)*4


def accuracy(rows, task=None):
    tasks = TASKS if task is None else (task,)
    counts = counts_from_rows(rows)
    values = [Fraction(*counts[t]) for t in tasks]
    return sum(values, Fraction()) / len(values)


def pp(value):
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'percentage_points': float(value*100)}


def same_predictions(left, right):
    keys = lambda rows: [(r['task'], r['id'], r['prediction']) for r in rows]
    return keys(left) == keys(right)


def implementation_checks(rows, init_rows, conditions, registration, training):
    base = rows['Base']
    init = {arm: {'predictions_identical_to_Base': same_predictions(init_rows[arm], base),
                  'rows_identical_to_Base': init_rows[arm] == base} for arm in WIDTH_ARMS}
    params = {arm: {key: training[arm]['metrics'][key] for key in
                    ('params.stored', 'params.trainable', 'params.with_grad')} for arm in WIDTH_ARMS}
    checks = {
        'a_iv_off_equals_same_allocation_Base':
            conditions['iv_off']['predictions'] == conditions['Base_same_allocation']['predictions'],
        'b_initial_predictions_equal_Base': all(v['predictions_identical_to_Base'] for v in init.values()),
        'c_reload_identical': all(training[arm]['metrics'].get('reload.identical') is True for arm in WIDTH_ARMS),
        'd_parameters_registered': (registration.get('passed') is True and all(
            set(v.values()) == {PARAMETERS[arm]} == {registration['arms'][arm]['trainable']}
            for arm, v in params.items()) and params['W-01'] == params['W-02']
            and registration.get('lora_rank_nearest_W-02') == LORA_RANKS['W-03']),
    }
    return {'checks': checks, 'passed': all(checks.values()), 'initial_evaluation': init,
            'iv_off_equals_pinned_Base_predictions': same_predictions(conditions['iv_off']['predictions'], base),
            'parameters': params}


def decide(rows, interventions_normal, passed):
    """Rules 2-8 on exact fractions. Rule outcomes are withheld if rule 1 fails."""
    macro = {label: accuracy(value) for label, value in rows.items()}
    task = {label: {t: accuracy(value, t) for t in TASKS} for label, value in rows.items()}
    base = macro['Base']
    d_L = macro['W-02'] - macro['W-03']
    d_T = macro['W-02'] - macro['W-01']
    swap_drop = accuracy(interventions_normal) - macro['W-02/iv_swap']
    permute_drop = accuracy(interventions_normal) - macro['W-02/iv_permute']
    per_task_d_L = {t: task['W-02'][t] - task['W-03'][t] for t in TASKS}
    learned = {arm: macro[arm] >= base + ONE_PP for arm in WIDTH_ARMS}
    supports = macro['W-02'] >= base + ONE_PP and d_L >= -ONE_PP
    advantage = d_L >= ONE_PP and all(v >= -ONE_PP for v in per_task_d_L.values())
    reopen = d_T >= ONE_PP and swap_drop >= ONE_PP
    gate = (lambda value: value) if passed else (lambda value: None)
    rules = {
        'rule2_learnability': {arm: {'macro_minus_Base': pp(macro[arm] - base), 'at_least_Base_plus_1': learned[arm]}
                               for arm in WIDTH_ARMS} | {
            'lora_control_not_learned': not learned['W-03'],
            'evidence_level_if_lora_not_learned': 'diagnostic' if not learned['W-03'] else None},
        'rule3_H0018': {'W-02_minus_Base': pp(macro['W-02'] - base), 'd_L': pp(d_L),
                        'outcome': gate('supports' if supports else 'challenges'),
                        'scope': 'screening, r128 only'},
        'rule4_practical_advantage': {'d_L': pp(d_L), 'per_task_d_L': {t: pp(v) for t, v in per_task_d_L.items()},
                                      'recorded': gate(advantage)},
        'rule5_history': {'d_T': pp(d_T), 'iv_swap_drop': pp(swap_drop), 'iv_permute_drop_descriptive': pp(permute_drop),
                          'reopen_H0014': gate(reopen),
                          'no_measurable_history_contribution': gate(d_T <= 0),
                          'descriptive_only': gate(not reopen and d_T > 0)},
        'rule6_width_effects_descriptive': {f'{arm} - {ref}': pp(macro[arm] - macro[ref]) for arm, ref in REFERENCES.items()},
        'rule8_no_further_widening': gate(not supports),
    }
    return rules, macro


def analyze(spec):
    expected = spec.get('expected', {})
    if (expected.get('lr_lock_sha256') != ACCEPTED_LR_LOCK_SHA256 or
            expected.get('base_manifest_sha256') != ACCEPTED_BASE_MANIFEST_SHA256 or
            not isinstance(expected.get('source_commit'), str)):
        raise ValueError('expected pins must name the reviewed source, accepted LR lock and Base manifest')
    if set(spec.get('training_runs', {})) != set(WIDTH_ARMS) or set(spec.get('reference_runs', {})) != set(REFERENCES.values()):
        raise ValueError('E-0026 needs exactly W-01/W-02/W-03 and the RB1-01/02/07 references')
    base_root = Path(spec['base_run'])
    if reference(base_root/'manifest.json')['sha256'] != expected['base_manifest_sha256']:
        raise ValueError('Base manifest differs from accepted ATT-0024-1')
    base_rows, base_manifest = _read_run(base_root, 'Base', 'base')
    rows = {'Base': validate_rows(base_rows, 'Base')}
    performance = {'Base': check_metrics(rows['Base'], base_manifest['metrics'], 'Base')}
    sources = {'Base': {name: reference(base_root/name) for name in ('manifest.json', 'predictions.json')}}
    registration_path = Path(spec['registration'])
    registration = read(registration_path)
    if registration.get('contract') != CONTRACT or registration.get('source_commit') != expected['source_commit']:
        raise ValueError('parameter registration is not from the reviewed E-0026 source')
    sources['registration'] = reference(registration_path)
    training, init_rows, curves, runtime, common = {}, {}, {}, {}, None
    for arm in WIDTH_ARMS:
        root = Path(spec['training_runs'][arm])
        manifest, refs, _ = validate_formal_run(root, arm, expected['source_commit'])
        metrics = manifest['metrics']
        if metrics['lr_lock.sha256'] != expected['lr_lock_sha256']:
            raise ValueError(f'{arm}: LR lock differs from expected release')
        plan = {k: metrics.get(k) for k in ('plan.steps', 'plan.examples', 'plan.input_tokens', 'plan.order_sha256',
                'split_manifest.sha256', 'source.commit', 'environment.lock_sha256', 'model_revision')}
        if common is not None and plan != common:
            raise ValueError(f'{arm}: training budget/order/split/source differs between arms')
        common = plan
        training[arm] = manifest
        rows[arm] = validate_rows(read(root/'predictions.json'), arm, rows['Base'])
        refs['predictions'] = reference(root/'predictions.json')
        performance[arm] = check_metrics(rows[arm], metrics, arm)
        curves[arm] = []
        for fraction in (0, .25, .5, .75):
            step = math.ceil(metrics['plan.steps']*fraction)
            path = root/f'dev-step-{step}.json'
            value = read(path)
            if value.get('step') != step:
                raise ValueError(f'{arm}: learning-curve step mismatch')
            curve_rows = validate_rows(value['predictions'], f'{arm}@{step}', rows['Base'])
            if step == 0:
                init_rows[arm] = curve_rows
            curves[arm].append({'fraction': fraction, 'step': step, 'source': reference(path),
                                'metrics': check_metrics(curve_rows, value['metrics'], f'{arm}@{step}')})
        curves[arm].append({'fraction': 1, 'step': metrics['plan.steps'], 'metrics': performance[arm]})
        sources[arm] = refs
        accounting = metrics.get('runtime.accounting', {})
        if arm != 'W-03':
            if set(accounting.get('layers', {})) != {'6', '13', '20', '27'}:
                raise ValueError(f'{arm}: accounting must cover all four adapted layers')
            writes = 0 if arm == 'W-01' else metrics['train.input_tokens']
            for value in accounting['layers'].values():
                if (value['writes'] != writes or value['reads'] != writes or
                        value['valid_tokens'] != metrics['train.input_tokens'] or
                        value['attention'] != value['ffn'] or value['ffn'] != metrics['train.examples']):
                    raise ValueError(f'{arm}: invalid per-layer write/read/original call counts')
            if accounting.get('state_M_c_bytes') != (0 if arm == 'W-01' else 4*STATE_BYTES_PER_LAYER):
                raise ValueError(f'{arm}: persistent M+c bytes differ from r128 layout')
        runtime[arm] = {k: v for k, v in metrics.items() if k.startswith(('train.', 'plan.', 'params.', 'runtime.', 'intermediate.'))
                        or k in ('learning_rate', 'init_eval.seconds', 'eval.seconds', 'save.seconds', 'load.seconds',
                                 'hardware', 'source.commit', 'environment.lock_sha256', 'model_revision', 'tokenizer_revision')}
    for ref_arm in REFERENCES.values():
        root = Path(spec['reference_runs'][ref_arm])
        manifest, refs, _ = validate_formal_run(root, ref_arm)
        if manifest['metrics']['plan.order_sha256'] != common['plan.order_sha256'] or \
                manifest['metrics']['plan.input_tokens'] != common['plan.input_tokens']:
            raise ValueError(f'{ref_arm}: reference order/budget differs from the width arms')
        rows[ref_arm] = validate_rows(read(root/'predictions.json'), ref_arm, rows['Base'])
        performance[ref_arm] = check_metrics(rows[ref_arm], manifest['metrics'], ref_arm)
        sources[ref_arm] = {**refs, 'predictions': reference(root/'predictions.json')}

    root = Path(spec['intervention_run'])
    manifest, provenance, conditions = (read(root/name) for name in ('manifest.json', 'provenance.json', 'conditions.json'))
    if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True or
            manifest.get('contract') != CONTRACT or manifest.get('arm') != 'W-02' or
            manifest.get('stage') != 'intervention' or manifest.get('seed') != 0):
        raise ValueError('W-02: incomplete or wrong intervention run')
    if provenance.get('training_manifest') != {k: sources['W-02']['manifest'][k] for k in ('path', 'sha256')}:
        raise ValueError('W-02: intervention checkpoint is not the supplied training run')
    if set(conditions) != {'normal', 'Base_same_allocation', *WIDTH_MODES}:
        raise ValueError('W-02: intervention conditions differ from E-0026')
    read_tables = {}
    for mode, condition in conditions.items():
        label = f'W-02/{mode}'
        condition_rows = validate_rows(condition['predictions'], label, rows['Base'])
        performance[label] = check_metrics(condition_rows, condition['metrics'], label)
        conditions[mode]['predictions'] = condition_rows
        if mode != 'Base_same_allocation':
            validate_read_summary(condition.get('read_diagnostics', []), 'W-02', mode)
            rows[label] = condition_rows
            read_tables[label] = condition.get('read_diagnostics', [])
    sources['W-02/interventions'] = {name: reference(root/name) for name in ('manifest.json', 'conditions.json', 'provenance.json')}
    agreement = normal_training_agreement(conditions['normal']['predictions'], rows['W-02'],
        training['W-02']['metrics'].get('hardware', 'unknown'), provenance.get('hardware', 'unknown'))

    implementation = implementation_checks(rows, init_rows, conditions, registration, training)
    rules, macro = decide(rows, rows['W-02/normal'], implementation['passed'])
    arrays = {label: {task: [int(r['prediction'] == r['gold']) for r in value if r['task'] == task] for task in TASKS}
              for label, value in rows.items()}
    comparisons = paired_bootstrap(arrays)
    metrics = ('macro_accuracy', 'piqa_accuracy', 'boolq_accuracy')
    named = [('W-02', 'W-03'), ('W-02', 'W-01'), ('W-01', 'W-03'), *(((arm, 'Base') for arm in WIDTH_ARMS)),
             *REFERENCES.items(), *((f'W-02/{mode}', 'W-02/normal') for mode in WIDTH_MODES)]
    intervals = {f'{left} - {right}': {metric: paired_interval(comparisons, left, right, metric) for metric in metrics}
                 for left, right in named}
    report = {'contract': CONTRACT, 'expected': expected, 'sources': sources, 'common_training': common,
              'performance': performance, 'learning_curves': curves, 'training_runtime': runtime,
              'implementation': implementation, 'rules': rules,
              'macro_fractions': {label: pp(value) for label, value in macro.items()},
              'intervals': intervals,
              'bootstrap': {'repeats': 10000, 'seed': 20261101, 'labels': list(arrays),
                            'scope': 'descriptive paired task-stratified dev intervals; shared indices across all labels'},
              'read_diagnostics': read_tables, 'intervention_normal_vs_training': agreement,
              'decision_authority': 'Pre-registered screening facts only; hypothesis status changes after coordinator conclusion review.'}
    if spec.get('observations'):
        report['cost'] = measured_cost(spec['observations'])
    finite_tree(report)
    return report


def markdown(report):
    lines = ['# E-0026 r128 width probe analysis', '', report['decision_authority'], '',
             '| Arm | PIQA % | BoolQ % | Macro % | PIQA NLL | BoolQ NLL |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label, row in report['performance'].items():
        if label == 'W-02/Base_same_allocation':
            continue
        lines.append(f'| {label} | {row["piqa"]["accuracy"]*100:.2f} | {row["boolq"]["accuracy"]*100:.2f} | '
                     f'{row["macro_accuracy"]*100:.2f} | {row["piqa"]["nll"]:.6f} | {row["boolq"]["nll"]:.6f} |')
    lines += ['', f'Implementation (rule 1): `{json.dumps(report["implementation"]["checks"])}`; '
              f'passed={report["implementation"]["passed"]}.', '', '## Rules', '']
    for name, rule in report['rules'].items():
        lines.append(f'- {name}: `{json.dumps(rule, ensure_ascii=False, sort_keys=True)}`')
    lines += ['', '## Paired differences (pp, descriptive 95% intervals)', '',
              '| Comparison | PIQA | BoolQ | Macro [95%] |', '| --- | ---: | ---: | ---: |']
    for name, value in report['intervals'].items():
        macro = value['macro_accuracy']; lo, hi = macro['ci95']
        lines.append(f'| {name} | {value["piqa_accuracy"]["difference"]*100:.2f} | {value["boolq_accuracy"]["difference"]*100:.2f} | '
                     f'{macro["difference"]*100:.2f} [{lo*100:.2f}, {hi*100:.2f}] |')
    lines += ['', '## Dev curves (macro %)', '', '| Arm | 0 | 25% | 50% | 75% | 100% |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, points in report['learning_curves'].items():
        lines.append(f'| {arm} | ' + ' | '.join(f'{p["metrics"]["macro_accuracy"]*100:.2f}' for p in points) + ' |')
    lines += ['', '## Training cost', '', '| Arm | Params stored/trainable/grad | Train s | Steady step s | Peak GiB | M+c bytes |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for arm, value in report['training_runtime'].items():
        lines.append(f'| {arm} | {value["params.stored"]}/{value["params.trainable"]}/{value["params.with_grad"]} | '
                     f'{value["train.seconds"]:.1f} | {value["train.steady_step_seconds"]:.4f} | '
                     f'{value["train.peak_cuda_bytes"]/2**30:.3f} | {value["runtime.accounting"].get("state_M_c_bytes", 0)} |')
    if 'cost' in report:
        lines += ['', f'Supplied measured usage: {report["cost"]["node_h"]:.6f} node hours.']
    lines += ['', 'Single seed; dev is used for the decision; intervals exclude seed and selection uncertainty. '
              'r128 reuses the r32 LR lock. T0 is a current-token nonlinear gate, not history-free attention.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, metrics, artifacts = False, {}, []
    try:
        report = analyze(read(args.input))
        report['input'] = reference(args.input)
        write_json(out/'analysis.json', report)
        (out/'analysis.md').write_text(markdown(report))
        metrics = {'implementation_passed': report['implementation']['passed'],
                   'rule3': report['rules']['rule3_H0018']['outcome'],
                   'rule4': report['rules']['rule4_practical_advantage']['recorded'],
                   'rule5_reopen_H0014': report['rules']['rule5_history']['reopen_H0014'],
                   'bootstrap_repeats': 10000}
        artifacts = [{'path': str(out/name), 'role': name} for name in ('analysis.json', 'analysis.md')]
        passed = True
    finally:
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': CONTRACT,
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics, 'artifacts': artifacts,
            'validator': {'name': 'rb_width_analysis', 'passed': passed}})


if __name__ == '__main__':
    main()
