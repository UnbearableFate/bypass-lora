"""E-0027 three-seed BoolQ replication: authenticate seeds 1-2, reuse seed 0, apply rules.

Run in PBS. Seed 0 is read from E-0025's locked analysis (ATT-0303-1), bound
by SHA256, and is not recomputed. Intervals are descriptive per-seed paired
task-stratified bootstraps (10,000 draws, seed 20261101).
"""
import argparse
from fractions import Fraction
import os
from pathlib import Path
import re
import time

import numpy as np

from rb1_analysis import measured_cost
from rb1_artifacts import read, reference
from rb1_rules import counts_from_rows, paired_bootstrap
from rb_history import donor_indices
from seg_analysis import summary
from seg_artifacts import validate, validate_rows
from seg_seed_rules import analyze as apply_rules, ARMS, NEW_SEEDS, number
from v5_train import write_json

CONTRACT = 'E-0027@1'
SEED0_ANALYSIS_SHA256 = 'a0533c5b28e143a57e498e40d48e5af85a2b3e77802cb86d171fc1d9c7514f6f'
BASE_SOURCE = '7d8b3464b6145504d155268f4c667fa3ecc53b00'
CAP_NODE_H = 3.2
KINDS = ('Base_seg', 'normal', 'iv_off', 'iv_swap', 'iv_reset')
TASKS = ('piqa', 'boolq')


def pinned_source(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{40}', value) is None:
        raise ValueError(f'{label} must be an immutable 40-hex commit')
    return value


def seed0(ref):
    if ref.get('sha256') != SEED0_ANALYSIS_SHA256 or reference(ref['path']) != ref:
        raise ValueError('seed-0 input is not the locked E-0025 analysis')
    report = read(ref['path'])
    if (report.get('contract') != 'E-0025@1' or report.get('stage') != 'final' or
            report['rules']['1'].get('passed') is not True):
        raise ValueError('seed-0 analysis is not the passed final E-0025 analysis')
    labels = (*ARMS, 'SEG-05/normal', 'SEG-05/iv_swap')
    return report, {label: tuple(report['counts'][label]['boolq']) for label in labels}


def max_control_intervals(binary, repeats=10000, seed=20261101, chunk_size=128):
    """Keep minus the better of Reset/T0, max recomputed per draw (same draws as paired_bootstrap)."""
    arrays = {a: {t: np.asarray(binary[a][t], dtype=float) for t in TASKS} for a in ('SEG-05', 'SEG-06', 'SEG-07')}
    sizes = {t: len(arrays['SEG-05'][t]) for t in TASKS}
    samples = {t: np.empty(repeats) for t in (*TASKS, 'macro')}
    rng = np.random.default_rng(seed)
    for start in range(0, repeats, chunk_size):
        end = min(start+chunk_size, repeats)
        draws = {t: rng.integers(0, sizes[t], (end-start, sizes[t])) for t in TASKS}
        scores = {a: {t: arrays[a][t][draws[t]].mean(1) for t in TASKS} for a in arrays}
        for t in TASKS:
            samples[t][start:end] = scores['SEG-05'][t]-np.maximum(scores['SEG-06'][t], scores['SEG-07'][t])
        macro = {a: (scores[a]['piqa']+scores[a]['boolq'])/2 for a in arrays}
        samples['macro'][start:end] = macro['SEG-05']-np.maximum(macro['SEG-06'], macro['SEG-07'])
    return {t: {'ci95_pp': [float(v*100) for v in np.quantile(x, [.025, .975])]} for t, x in samples.items()}


def descriptive(counts):
    """PIQA and macro analogues of gain/lora/swap; descriptive only (rule 8)."""
    s = {a: {t: Fraction(*counts[a][t]) for t in TASKS} for a in counts}
    for a in s:
        s[a]['macro'] = (s[a]['piqa']+s[a]['boolq'])/2
    out = {}
    for t in (*TASKS, 'macro'):
        out[t] = {'gain': number(s['SEG-05'][t]-max(s['SEG-06'][t], s['SEG-07'][t])),
                  'lora': number(s['SEG-05'][t]-s['SEG-04'][t]),
                  'swap': number(s['SEG-05/normal'][t]-s['SEG-05/iv_swap'][t])}
    return out


def new_seed(seed, runs, intervention, source, evaluation_source, paired):
    if set(runs) != set(ARMS):
        raise ValueError(f'seed {seed} needs exactly SEG-04..07')
    manifests, refs, predictions = {}, {}, {}
    for arm, root in runs.items():
        manifests[arm], refs[arm], _ = validate(root, arm, source, seed)
        predictions[arm] = validate_rows(read(Path(root)/'predictions.json'), paired)
    root = Path(intervention)
    manifest = read(root/'manifest.json'); m = manifest['metrics']; prov = read(root/'provenance.json')
    if (manifest.get('status') != 'complete' or manifest.get('contract') != CONTRACT or
            manifest.get('seed') != seed or manifest.get('stage') != 'intervention' or
            manifest.get('arm') != 'SEG-05' or manifest.get('validator', {}).get('passed') is not True or
            prov.get('training') != refs['SEG-05'] or prov.get('training_source') != source or
            prov.get('evaluation_source') != evaluation_source):
        raise ValueError(f'seed {seed} intervention source/checkpoint provenance differs')
    if prov.get('checkpoint_files') != read(Path(runs['SEG-05'])/'checkpoint.json'):
        raise ValueError(f'seed {seed} intervention checkpoint inventory differs')
    conditions = read(root/'conditions.json')
    if set(conditions) != set(KINDS):
        raise ValueError(f'seed {seed} intervention condition set differs')
    for kind, cond in conditions.items():
        predictions['SEG-05/'+kind] = validate_rows(cond['predictions'], paired)
    off = conditions['iv_off']['predictions'] == conditions['Base_seg']['predictions']
    if not off or m.get('off.identical_to_Base_seg') is not True:
        raise ValueError(f'seed {seed} same-allocation off/Base identity failed')
    ids = [{'task': t, 'id': i} for t, i, _ in paired]
    donor = donor_indices(ids, 'R1')
    expected = [{'task': r['task'], 'recipient': r['id'], 'donor': ids[donor[i]]['id']} for i, r in enumerate(ids)]
    if conditions['iv_swap']['audit']['donors'] != expected:
        raise ValueError(f'seed {seed} fixed donor map differs')
    keep = manifests['SEG-05']['metrics']
    initial = (keep.get('initial.identical_to_Base_seg') is True and
               read(Path(runs['SEG-05'])/'initial-base-predictions.json') == read(Path(runs['SEG-05'])/'initial-predictions.json'))
    digests = {a: manifests[a]['metrics']['initial.adapter_sha256'] for a in ARMS}
    checks = {'off_matches_base': off, 'initial_matches_base': initial,
              'reload_identical': all(manifests[a]['metrics'].get('reload.identical') is True for a in ARMS),
              'initial_parameters_identical': digests['SEG-05'] == digests['SEG-06'] == digests['SEG-07']}
    flips = sum(a['prediction'] != b['prediction'] for a, b in zip(predictions['SEG-05'], predictions['SEG-05/normal']))
    meta = {'initial_adapter_sha256': digests, 'normal_training_prediction_flips': flips,
            'order_sha256': {a: manifests[a]['metrics']['plan.order_sha256'] for a in ARMS},
            'intervention': {'manifest': reference(root/'manifest.json'), 'provenance': reference(root/'provenance.json'),
                             'conditions': reference(root/'conditions.json')}}
    return manifests, refs, predictions, checks, meta


def complete_cost(inputs, roots, check_manifest):
    declared = inputs.get('accounted_attempts')
    if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared) or
            any(not isinstance(a, str) or re.fullmatch(r'ATT-\d+-\d+', a) is None for a in declared)):
        raise ValueError('analysis needs the explicit terminal task accounting list')
    inferred = {Path(p).resolve().name for p in roots}
    inferred.add(Path(check_manifest).resolve().parent.name)
    if any(re.fullmatch(r'ATT-\d+-\d+', a) is None for a in inferred):
        raise ValueError('all run/check paths must name their registered attempt')
    if not inferred <= set(declared):
        raise ValueError('task accounting list omits a required run/check attempt')
    cost = measured_cost(inputs['observations'])
    if {r['attempt'] for r in cost['rows']} != set(declared):
        raise ValueError('measured cost coverage differs from the complete task accounting list')
    cost.update({'required_attempts': sorted(declared), 'terminal_coverage_complete': True,
                 'cap_node_h': CAP_NODE_H, 'within_cap': cost['node_h'] <= CAP_NODE_H,
                 'current_analysis_usage': 'pending; add terminal measured usage in delivery'})
    return cost


def analyze(inputs):
    source = pinned_source(inputs['training_source'], 'training_source')
    if inputs.get('evaluation_source') != source:
        raise ValueError('intervention evaluation source must equal the reviewed training source')
    report0, counts0 = seed0(inputs['seed0_analysis'])
    _, base_ref, _ = validate(inputs['base_run'], 'SEG-00', BASE_SOURCE, 0)
    base_rows = validate_rows(read(Path(inputs['base_run'])/'predictions.json'))
    paired = [(r['task'], r['id'], r['gold']) for r in base_rows]
    if counts_from_rows(base_rows) != {t: tuple(report0['counts']['SEG-00'][t]) for t in TASKS}:
        raise ValueError('Base_seg run differs from the seed-0 analysis Base_seg')
    checked = read(inputs['deterministic_checks']['manifest_ref']['path'])
    if (reference(inputs['deterministic_checks']['manifest_ref']['path']) != inputs['deterministic_checks']['manifest_ref'] or
            checked.get('status') != 'complete' or checked.get('validator', {}).get('passed') is not True or
            checked.get('source_commit') != source or checked['metrics'].get('no_channel_identity') is not True or
            sorted(checked['metrics'].get('no_channel_seeds', [])) != list(NEW_SEEDS)):
        raise ValueError('deterministic no-channel checks missing at the reviewed source for both seeds')
    counts, checks, per, roots = {0: counts0}, {}, {}, []
    for seed in NEW_SEEDS:
        runs = inputs['runs'][str(seed)]; iv = inputs['interventions'][str(seed)]
        roots += [*runs.values(), iv]
        manifests, refs, predictions, check, meta = new_seed(seed, runs, iv, source, inputs['evaluation_source'], paired)
        checks[seed] = {**check, 'no_channel_identity': True}
        task_counts = {a: counts_from_rows(p) for a, p in predictions.items()}
        counts[seed] = {label: task_counts[label]['boolq'] for label in (*ARMS, 'SEG-05/normal', 'SEG-05/iv_swap')}
        binary = {a: {t: [int(r['gold'] == r['prediction']) for r in p if r['task'] == t] for t in TASKS}
                  for a, p in predictions.items()}
        per[str(seed)] = {'summary': {a: summary(p) for a, p in predictions.items()}, 'counts': task_counts,
            'descriptive': descriptive(task_counts),
            'bootstrap': {'repeats': 10000, 'seed': 20261101, 'pairs': paired_bootstrap(binary),
                          'max_control_gain': max_control_intervals(binary)},
            'training': {a: m['metrics'] for a, m in manifests.items()}, 'artifact_refs': refs, **meta}
    seed0_counts = {a: {t: tuple(report0['counts'][a][t]) for t in TASKS}
                    for a in (*ARMS, 'SEG-05/normal', 'SEG-05/iv_swap')}
    per['0'] = {'summary': {a: report0['summary'][a] for a in (*ARMS, *('SEG-05/'+k for k in KINDS))},
                'descriptive': descriptive(seed0_counts),
                'bootstrap': {'max_control_gain': report0['bootstrap']['max_control_gain']['128'],
                              'pairs': {k: v for k, v in report0['bootstrap']['pairs'].items()
                                        if k in ('SEG-05 - SEG-04', 'SEG-04 - SEG-05', 'SEG-05/normal - SEG-05/iv_swap')}},
                'source': inputs['seed0_analysis']}
    rules = apply_rules(counts, checks, tuple(report0['counts']['SEG-00']['boolq']))
    cost = complete_cost(inputs, roots, inputs['deterministic_checks']['manifest_ref']['path'])
    return {**rules, 'contract': CONTRACT, 'hypothesis': 'H-0020@1', 'stage': 'final',
            'base_seg': {'run': base_ref, 'summary': summary(base_rows)}, 'seeds': per,
            'deterministic_checks': inputs['deterministic_checks'], 'cost': cost,
            'limitations': ['Three seeds only; dev both selects and judges; exact point-estimate gates.',
                'Per-seed intervals are descriptive, condition on each trained model, exclude seed variation.',
                'RBc LR not recalibrated for r128 or the segmented protocol.',
                'BoolQ task level only; no confirmation evaluation; no practical or long-context claim.']}


def markdown(report):
    lines = ['# E-0027 BoolQ r128 three-seed replication', '',
             f"Direction: {report['direction']}; G {report['G']['pp']:.4f} pp, L {report['L']['pp']:.4f} pp, "
             f"W {report['W']['pp']:.4f} pp.", '',
             '| Seed | SEG-04 | SEG-05 | SEG-06 | SEG-07 | swap | gain | lora | swap drop |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s, q in report['per_seed'].items():
        b = q['boolq']
        lines.append(f"| {s} | {b['SEG-04']['pp']:.1f} | {b['SEG-05']['pp']:.1f} | {b['SEG-06']['pp']:.1f} | "
                     f"{b['SEG-07']['pp']:.1f} | {b['SEG-05/iv_swap']['pp']:.1f} | {q['gain']['pp']:+.2f} | "
                     f"{q['lora']['pp']:+.2f} | {q['swap']['pp']:+.2f} |")
    import json
    lines += ['', '## Rules', '']
    for k, rule in report['rules'].items():
        lines += [f'Rule {k}: `{json.dumps(rule, ensure_ascii=False)}`', '']
    lines += ['## Limits', '']+['- '+s for s in report['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, start = False, time.monotonic()
    try:
        report = analyze(read(args.input))
        write_json(out/'analysis.json', report)
        (out/'analysis.md').write_text(markdown(report))
        passed = True
    finally:
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': CONTRACT,
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': {'elapsed_s': time.monotonic()-start,
                'bootstrap.repeats': 10000, 'bootstrap.seed': 20261101, 'confirmation_evaluated': False},
            'artifacts': [{'path': str(f), 'role': f.name} for f in (out/'analysis.json', out/'analysis.md') if f.exists()],
            'validator': {'name': 'segmented_three_seed_analysis', 'passed': passed}})


if __name__ == '__main__':
    main()
