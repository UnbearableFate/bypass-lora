"""Authenticate the frozen SEG grid, apply rules, and bootstrap in PBS."""
import argparse
import os
from pathlib import Path
import time

from rb1_analysis import measured_cost
from rb1_artifacts import read, reference
from rb1_rules import counts_from_rows, paired_bootstrap
from seg_artifacts import validate, validate_rows
from seg_rules import analyze as apply_rules, gain_intervals
from seg_model import KEEP
from v5_train import write_json

RB1_REFERENCE_SOURCE = '7bf091d4af24c2a9fdc666a5f885c31817b97174'


def summary(rows):
    tasks = {}
    for task in ('piqa', 'boolq'):
        r = [row for row in rows if row['task'] == task]
        tasks[task] = {'accuracy': sum(x['prediction'] == x['gold'] for x in r)/len(r),
            'correct': sum(x['prediction'] == x['gold'] for x in r), 'total': len(r),
            'nll': sum(x['gold_nll_sum'] for x in r)/sum(x['gold_loss_tokens'] for x in r)}
    return {**tasks, 'macro_accuracy': sum(t['accuracy'] for t in tasks.values())/2,
            'macro_nll': sum(t['nll'] for t in tasks.values())/2}


def analyze(inputs, stage='final'):
    source = inputs['training_source']
    if len(source) != 40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('analysis needs the reviewed immutable training-source pin')
    roots = inputs['runs']
    required = {f'SEG-{i:02}' for i in range(5)}
    if set(roots) not in (required, required | {'SEG-05', 'SEG-06', 'SEG-07'}):
        raise ValueError('analysis needs all initial arms and optionally all three r128 arms')
    predictions, manifests, refs = {}, {}, {}
    for arm, root in roots.items():
        manifests[arm], refs[arm], _ = validate(root, arm, source)
        predictions[arm] = read(Path(root)/'predictions.json')
    paired = [(r['task'], r['id'], r['gold']) for r in predictions['SEG-00']]
    for p in predictions.values():
        validate_rows(p, paired)
    # Pre-existing full-context reference only, bound to the accepted final RB1.
    from rb1_artifacts import validate_formal_run
    full, _, _ = validate_formal_run(inputs['full_context_lora_run'], 'RB1-07')
    predictions['RB1-07'] = read(Path(inputs['full_context_lora_run'])/'predictions.json')
    validate_rows(predictions['RB1-07'], paired)
    reference_predictions = {}
    for label, ref in inputs.get('full_context_references', {}).items():
        if reference(ref['path']) != ref:
            raise ValueError('full-context reference hash changed')
        p = read(ref['path']); validate_rows(p, paired)
        reference_predictions[label] = summary(p)
    checks = dict(inputs['deterministic_checks'])
    if checks.get('manifest_ref') is None or reference(checks['manifest_ref']['path']) != checks['manifest_ref']:
        raise ValueError('missing bound deterministic-check manifest')
    checked = read(checks['manifest_ref']['path'])
    if checked.get('status') != 'complete' or checked.get('validator', {}).get('passed') is not True:
        raise ValueError('deterministic check run failed')
    no_channel = checked['metrics'].get('no_channel_identity') is True
    initial = all(m['metrics'].get('initial.identical_to_Base_seg') is True for a, m in manifests.items() if a in KEEP)
    reload_ok = all(m['metrics'].get('reload.identical') is True for a, m in manifests.items() if a != 'SEG-00')
    off_ok, provenance = True, {}
    for arm in (a for a in roots if a in KEEP):
        if arm not in inputs.get('interventions', {}):
            if stage == 'final':
                raise ValueError('segmented intervention grid missing')
            off_ok = False
            continue
        root = Path(inputs['interventions'][arm])
        manifest = read(root/'manifest.json'); m = manifest['metrics']; prov = read(root/'provenance.json')
        if (manifest.get('status') != 'complete' or manifest.get('contract') != 'E-0025@1' or
            manifest.get('stage') != 'intervention' or manifest.get('arm') != arm or
            manifest.get('validator', {}).get('passed') is not True or
            prov.get('training') != refs[arm] or prov.get('training_source') != source or
            prov.get('evaluation_source') != inputs['evaluation_source']):
            raise ValueError('segmented intervention source/checkpoint provenance differs')
        if prov.get('checkpoint_files') != read(Path(roots[arm])/'checkpoint.json'):
            raise ValueError('segmented intervention checkpoint inventory differs')
        conditions = read(root/'conditions.json')
        if set(conditions) != {'Base_seg', 'normal', 'iv_off', 'iv_swap', 'iv_reset'}:
            raise ValueError('segmented intervention condition set differs')
        for kind, cond in conditions.items():
            validate_rows(cond['predictions'], paired)
            predictions[arm+'/'+kind] = cond['predictions']
        actual_off = conditions['iv_off']['predictions'] == conditions['Base_seg']['predictions']
        if not actual_off or m.get('off.identical_to_Base_seg') is not True:
            raise ValueError('segmented same-allocation off/Base identity failed')
        off_ok &= actual_off
        donor_map = conditions['iv_swap']['audit']['donors']
        from rb_history import donor_indices
        ids = [{'task': r['task'], 'id': r['id']} for r in predictions['SEG-00']]
        donor = donor_indices(ids, 'R1')
        expected = [{'task': r['task'], 'recipient': r['id'], 'donor': ids[donor[i]]['id']} for i, r in enumerate(ids)]
        if donor_map != expected:
            raise ValueError('segmented fixed donor map differs')
        provenance[arm] = {'manifest': reference(root/'manifest.json'), 'provenance': reference(root/'provenance.json'),
            'conditions': reference(root/'conditions.json'), 'normal_matches_training': conditions['normal']['predictions'] == predictions[arm]}
    rule_checks = {'no_channel_identity': no_channel, 'initial_matches_base': initial,
                   'reload_identical': reload_ok, 'off_matches_base': off_ok}
    counts = {a: counts_from_rows(p) for a, p in predictions.items()}
    rules = apply_rules(counts, rule_checks, require_interventions=stage == 'final')
    binary = {a: {t: [int(r['gold'] == r['prediction']) for r in p if r['task'] == t]
                  for t in ('piqa', 'boolq')} for a, p in predictions.items()}
    return {**rules, 'contract': 'E-0025@1', 'clarification': 'D-0044', 'stage': stage,
        'summary': {a: summary(p) for a, p in predictions.items()}, 'counts': counts,
        'bootstrap': {'repeats': 10000, 'seed': 20261101, 'pairs': paired_bootstrap(binary),
                      'max_control_gain': gain_intervals(binary)},
        'training': {a: m['metrics'] for a, m in manifests.items()}, 'artifact_refs': refs,
        'intervention_refs': provenance, 'deterministic_checks': checks,
        'full_context_references_descriptive_only': reference_predictions,
        'cost': measured_cost(inputs.get('observations', [])),
        'limitations': ['Single trained seed, dev screening; bootstrap conditions on these trained models.',
            'Thresholds are exact point-estimate gates, not interval guarantees.',
            'Intervals are descriptive; not simultaneous; exclude seed and task/width selection uncertainty.',
            'RBc LR reused; segmented optimum may differ. Full-context references use a different protocol.',
            'No confirmation evaluation or compression/long-context/practical superiority claim.']}


def markdown(report):
    lines = ['# E-0025 segmented information access', '', 'Single-seed dev screening.', '',
             f"Primary: {report['primary']}; outcome: {report['direction']}; r128 release: {report['release_r128']}.", '',
             '| Arm | PIQA % | BoolQ % | Macro % | Macro NLL |', '|---|---:|---:|---:|---:|']
    for a, m in report['summary'].items():
        lines.append(f"| {a} | {100*m['piqa']['accuracy']:.2f} | {100*m['boolq']['accuracy']:.2f} | {100*m['macro_accuracy']:.2f} | {m['macro_nll']:.6f} |")
    lines += ['', '## Rules', '']
    import json
    for number, rule in report['rules'].items():
        lines += [f'Rule {number}: `{json.dumps(rule, ensure_ascii=False)}`', '']
    lines += ['## Limits', '']+['- '+s for s in report['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--stage', choices=('select', 'final'), default='final')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, start = False, time.monotonic()
    try:
        report = analyze(read(args.input), args.stage)
        write_json(out/'analysis.json', report)
        (out/'analysis.md').write_text(markdown(report))
        passed = True
    finally:
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0025@1',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': {'elapsed_s': time.monotonic()-start,
                'bootstrap.repeats': 10000, 'bootstrap.seed': 20261101, 'confirmation_evaluated': False},
            'artifacts': [{'path': str(f), 'role': f.name} for f in (out/'analysis.json', out/'analysis.md') if f.exists()],
            'validator': {'name': 'segmented_locked_analysis', 'passed': passed}})


if __name__ == '__main__':
    main()
