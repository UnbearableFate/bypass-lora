"""Read-only E-0014 screening analysis; run in PBS, never open confirmation data.

Gates use seed-0 final dev point estimates with exact count fractions. Bootstrap
intervals (V2's paired episode bootstrap) are descriptive and never gate.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path

import numpy as np

from v2_factorial_analysis import (BOOTSTRAP_REPEATS, BOOTSTRAP_SEED, AnalysisError, finite, paired_bootstrap,
                                   point, read, require)

CONTROLS = ('normal', 'zero', 'shuffled', 'permuted', 'oracle_key', 'oracle_value')
VARIANT_CONTROLS = ('normal', 'zero', 'shuffled')
ARMS = ('R1', 'R2', 'R3', 'R4', 'R5', 'R6')
RULES = {
    'H3': 'E=16 dev：EM_normal − max(EM_zero, EM_shuffled) ≥ 10 pp；NLL_normal < min(NLL_zero, NLL_shuffled)；目标事实平均地址权重 ≥ 0.125（E-0010@2）。',
    'R1': 'value 通路可用（R1，E=1 dev）：EM_normal − max(EM_zero, EM_shuffled) ≥ 50 pp。',
    'R2': 'E=4 寻址可用（R2，E=4 dev）：EM 差 ≥ 10 pp，NLL 条件同上，目标平均地址权重 ≥ 2/4。',
    'R5': '额外监督（目标事实索引）只在 R5；其结论只在有地址监督时成立。',
}
EXPECTED = {'R1': {'train_facts': 1, 'extra_dev': [1]}, 'R2': {'train_facts': 4, 'extra_dev': [4]},
            'R3': {'curriculum': [[1, 1000], [4, 1000], [8, 1000], [16, 3000]], 'extra_dev': [1, 4]},
            'R4': {'prior_rank': 32, 'prior_steps': 1000, 'prior_lr': 3e-4}, 'R5': {'address_loss_weight': 1.},
            'R6': {}}


def gate(controls, target_weight, *, margin, weight):
    normal, zero, shuffled = (controls[m] for m in ('normal', 'zero', 'shuffled'))
    n = normal['count']
    require(zero['count'] == n == shuffled['count'], 'gate needs equal counts')
    diff = Fraction(normal['correct_count'] - max(zero['correct_count'], shuffled['correct_count']), n)
    em_pass = diff >= margin
    nll_pass = normal['nll'] < min(zero['nll'], shuffled['nll'])
    weight_pass = None if weight is None else target_weight >= weight
    return {'n': n, 'correct': {m: controls[m]['correct_count'] for m in controls},
            'em': {m: controls[m]['em'] for m in controls}, 'nll': {m: controls[m]['nll'] for m in controls},
            'em_margin': float(diff), 'em_margin_fraction': str(diff), 'em_pass': em_pass, 'nll_pass': nll_pass,
            'target_weight': target_weight, 'weight_pass': weight_pass,
            'passed': em_pass and nll_pass and weight_pass is not False}


def bootstrap(records, controls):
    columns, names = [], []
    for mode in controls:
        names += [(mode, 'em'), (mode, 'nll')]
        columns += [[float(r['correct']) for r in records[mode]], [r['nll'] for r in records[mode]]]
    values = np.asarray(columns, dtype=np.float64).T
    boots, points = paired_bootstrap(values), values.mean(0)
    index = {name: i for i, name in enumerate(names)}
    return {f'normal-{mode}': {metric: {'estimate': float(points[index['normal', metric]] - points[index[mode, metric]]),
                                        'ci95': np.quantile(boots[:, index['normal', metric]] - boots[:, index[mode, metric]],
                                                            [.025, .975]).tolist()}
                               for metric in ('em', 'nll')} for mode in controls if mode != 'normal'}


def load_split(directory, controls, provenance, label):
    summary = read(directory/'dev.metrics.json', provenance)
    records, points = {}, {}
    for mode in controls:
        rows = read(directory/f'dev.{mode}.predictions.jsonl', provenance, lines=True)
        require(len({r['episode_id'] for r in rows}) == len(rows), f'{label}: duplicate episode')
        require(all(type(r['correct']) is bool and finite(r['nll']) for r in rows), f'{label}: invalid record')
        p = point(rows)
        require(p['count'] == summary[mode]['count'] and abs(p['nll'] - summary[mode]['nll']) < 1e-10
                and p['em'] == summary[mode]['em'], f'{label}: summary mismatch for {mode}')
        records[mode], points[mode] = rows, p
    ids = [r['episode_id'] for r in records['normal']]
    require(all([r['episode_id'] for r in records[m]] == ids for m in controls), f'{label}: unaligned controls')
    return summary, records, points


def curve(path, provenance, every=500):
    rows = read(path, provenance, lines=True)
    keys = [k for k in ('loss', 'answer_ce', 'address_loss') if k in rows[0]]
    return [{'steps': f'{start+1}-{min(start+every, len(rows))}',
             **{k: sum(r[k] for r in rows[start:start+every]) / len(rows[start:start+every]) for k in keys}}
            for start in range(0, len(rows), every)]


def load_arm(label, run_dir, arm_dir_name, *, reference=False):
    provenance = {}
    run_dir = Path(run_dir).resolve()
    manifest = read(run_dir/'manifest.json', provenance)
    require(manifest['status'] == 'complete' and manifest['validator']['passed'] is True, f'{label}: incomplete run')
    require(not manifest['smoke_steps'] and not manifest['smoke_eval_limit'], f'{label}: smoke run')
    arm = run_dir/arm_dir_name
    metrics = read(arm/'metrics.json', provenance)
    require(metrics['checkpoint_reload']['predictions_identical'] is True, f'{label}: reload mismatch')
    require(metrics['schedule']['steps'] == 6000, f'{label}: not 6000 steps')
    options = manifest.get('options', {})
    if not reference:
        for key, value in EXPECTED[label].items():
            require(options.get(key) == value, f'{label}: option {key}={options.get(key)} != {value}')
        require(options.get('step_budget') == 6000, f'{label}: step budget')
        others = {k: v for k, v in options.items() if k not in EXPECTED[label] and k not in ('step_budget', 'curriculum', 'extra_dev', 'train_facts')}
        require(all(not v for v in others.values()), f'{label}: unexpected extra options {others}')
    summary, records, points = load_split(arm, CONTROLS, provenance, label)
    normal = summary['normal']
    result = {'run': str(run_dir), 'config': metrics['config'], 'lr': metrics['lr'], 'options': options,
              'schedule': metrics['schedule'],
              'h3': gate(points, normal['diagnostics']['address.target_weight'], margin=Fraction(1, 10), weight=.125),
              'controls': {m: {'em': points[m]['em'], 'nll': points[m]['nll'], 'correct': points[m]['correct_count'],
                               'prior_em': summary[m].get('prior_em')} for m in CONTROLS},
              'diagnostics': {k: normal['diagnostics'].get(k) for k in
                              ('address.target_weight', 'address.n_eff', 'address.entropy', 'address.top1',
                               'read.normalized_error', 'read.direction_error', 'read.reconstruction_max_abs')},
              'last_position': {m: summary[m].get('last_position') for m in CONTROLS},
              'all_question_positions': summary['normal'].get('all_question_positions'),
              'bootstrap': bootstrap(records, CONTROLS),
              'train': {k: metrics['train'].get(k) for k in ('steps', 'tokens', 'training_seconds', 'stable_step_seconds',
                                                               'clip_fraction', 'trainable_parameters', 'monitors')},
              'curve': curve(arm/'train.jsonl', provenance),
              'learning_curves': {k: {m: {'em': v[m]['em'], 'nll': v[m]['nll']} for m in v}
                                  for k, v in metrics['train']['learning_curves'].items()},
              'wall_seconds': manifest['metrics']['wall_seconds'],
              'in_job_mig_instance_hours': manifest['metrics']['mig_instance_hours']}
    if 'prior_stage' in metrics:
        stage = metrics['prior_stage']
        result['prior_stage'] = {'lr': stage['lr'], 'steps': stage['train']['steps'], 'seconds': stage['seconds'],
                                 'frozen_during_memory_training': stage.get('frozen_during_memory_training'),
                                 'dev': {m: {'em': stage['dev'][m]['em'], 'nll': stage['dev'][m]['nll']} for m in stage['dev']},
                                 'curve': curve(arm/'prior_train.jsonl', provenance)}
    result['variants'] = {}
    for facts in options.get('extra_dev', []):
        label_e = f'e{facts}'
        v_summary, v_records, v_points = load_split(arm/label_e, VARIANT_CONTROLS, provenance, f'{label} {label_e}')
        weight = v_summary['normal']['diagnostics']['address.target_weight']
        result['variants'][label_e] = {
            'gate_R1': gate(v_points, weight, margin=Fraction(1, 2), weight=None) if facts == 1 else None,
            'gate_R2': gate(v_points, weight, margin=Fraction(1, 10), weight=.5) if facts == 4 else None,
            'controls': {m: {'em': v_points[m]['em'], 'nll': v_points[m]['nll'], 'correct': v_points[m]['correct_count']}
                         for m in VARIANT_CONTROLS},
            'target_weight': weight, 'n_eff': v_summary['normal']['diagnostics'].get('address.n_eff'),
            'last_position': v_summary['normal'].get('last_position'),
            'bootstrap': bootstrap(v_records, VARIANT_CONTROLS)}
    result['provenance'] = provenance
    return result


def markdown(report):
    lines = ['# E-0014 screening (seed 0, final checkpoints)', '', '## E=16 dev H3 gate', '',
             '| arm | normal / zero / shuffled correct (n=1000) | EM margin | NLL normal / zero / shuffled | target weight | n_eff | H3 |',
             '|---|---|---:|---|---:|---:|---|']
    for name, arm in report['arms'].items():
        h = arm['h3']
        lines.append(f"| {name} | {h['correct']['normal']} / {h['correct']['zero']} / {h['correct']['shuffled']} | "
                     f"{h['em_margin']*100:+.1f} pp | {h['nll']['normal']:.4f} / {h['nll']['zero']:.4f} / {h['nll']['shuffled']:.4f} | "
                     f"{h['target_weight']:.4f} | {arm['diagnostics']['address.n_eff']:.3f} | {'pass' if h['passed'] else 'fail'} |")
    lines += ['', '## E=16 dev other controls (EM, NLL)', '', '| arm | permuted | oracle_key | oracle_value |', '|---|---|---|---|']
    for name, arm in report['arms'].items():
        c = arm['controls']
        lines.append(f"| {name} | " + ' | '.join(f"{c[m]['em']:.3f}, {c[m]['nll']:.4f}" for m in ('permuted', 'oracle_key', 'oracle_value')) + ' |')
    lines += ['', '## Last question position, E=16 dev normal', '',
              '| arm | argmax target rate | most common slot (rate) | same-relation rate | target weight | fusion/residual mean (median) | prior/residual mean |',
              '|---|---:|---|---:|---:|---|---:|']
    for name, arm in report['arms'].items():
        last = arm['last_position']['normal'] or {}  # V2-03 predates these diagnostics
        counts = last.get('argmax_slot_counts', [])
        top = max(range(len(counts)), key=counts.__getitem__) if counts else None
        ratio = last.get('fusion_to_residual') or {}
        prior = last.get('prior_to_residual')
        lines.append(f"| {name} | {last.get('argmax_target_rate', float('nan')):.3f} | {top} ({last.get('argmax_most_common_slot_rate', float('nan')):.3f}) | "
                     f"{last.get('argmax_same_relation_rate', float('nan')):.3f} | {last.get('target_weight_mean', float('nan')):.4f} | "
                     f"{ratio.get('mean', float('nan')):.4g} ({ratio.get('quantiles', {}).get('0.5', float('nan')):.4g}) | "
                     f"{'' if not prior else format(prior['mean'], '.4g')} |")
    lines += ['', '## Variant dev (normal / zero / shuffled)', '',
              '| arm | split | correct | EM margin | NLL normal / zero / shuffled | target weight | gate |', '|---|---|---|---:|---|---:|---|']
    for name, arm in report['arms'].items():
        for split, v in arm['variants'].items():
            g = v['gate_R1'] or v['gate_R2']
            c = v['controls']
            margin = (c['normal']['correct'] - max(c['zero']['correct'], c['shuffled']['correct'])) / 1000
            verdict = '' if g is None else ('R1 gate ' if v['gate_R1'] else 'R2 gate ') + ('pass' if g['passed'] else 'fail')
            lines.append(f"| {name} | {split} | {c['normal']['correct']} / {c['zero']['correct']} / {c['shuffled']['correct']} | "
                         f"{margin*100:+.1f} pp | {c['normal']['nll']:.4f} / {c['zero']['nll']:.4f} / {c['shuffled']['nll']:.4f} | "
                         f"{v['target_weight']:.4f} | {verdict} |")
    lines += ['', 'Intervals (descriptive, paired bootstrap 10,000) are in report.json under bootstrap.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', action='append', required=True, metavar='R=RUN_DIR:ARM_SUBDIR')
    parser.add_argument('--reference', required=True, metavar='RUN_DIR:ARM_SUBDIR')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arms = {}
    for item in args.arm:
        name, _, rest = item.partition('=')
        run, _, sub = rest.partition(':')
        require(name in ARMS and name not in arms, 'arms are unique R1..R6')
        arms[name] = load_arm(name, run, sub)
    run, _, sub = args.reference.partition(':')
    reference = load_arm('V2-03', run, sub, reference=True)
    require(set(arms) == set(ARMS), 'all six arms are required')
    report = {'rules': RULES, 'arms': dict(sorted(arms.items())), 'reference_V2_03': reference,
              'h3_passing_arms': [a for a in ARMS if arms[a]['h3']['passed']],
              'h3_passing_without_extra_supervision': [a for a in ARMS if arms[a]['h3']['passed'] and a != 'R5'],
              'r1_value_path_gate': arms['R1']['variants']['e1']['gate_R1']['passed'],
              'r2_e4_address_gate': arms['R2']['variants']['e4']['gate_R2']['passed'],
              'bootstrap': {'unit': 'episode', 'repeats': BOOTSTRAP_REPEATS, 'seed': BOOTSTRAP_SEED,
                            'pairing': 'same episode draws for every control within an arm and split'},
              'spec_digest': os.environ.get('SR_SPEC_DIGEST')}
    report['arms']['V2-03'] = reference
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output/'report.md').write_text(markdown(report))
    artifacts = [{'path': str((args.output/f).resolve()), 'role': f} for f in ('report.json', 'report.md')]
    (args.output/'manifest.json').write_text(json.dumps({
        'status': 'complete', 'spec_digest': os.environ.get('SR_SPEC_DIGEST'),
        'metrics': {'h3_passing_arms': report['h3_passing_arms'], 'r1_value_path_gate': report['r1_value_path_gate'],
                    'r2_e4_address_gate': report['r2_e4_address_gate'],
                    **{f'{a}.em_margin': arms[a]['h3']['em_margin'] for a in ARMS},
                    **{f'{a}.target_weight': arms[a]['h3']['target_weight'] for a in ARMS}},
        'artifacts': artifacts, 'validator': {'name': 'v2r_analysis', 'passed': True}}, indent=2) + '\n')
    print(report['h3_passing_arms'])


if __name__ == '__main__':
    raise SystemExit(main())
