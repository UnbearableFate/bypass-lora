"""Authenticate and analyze the locked E-0030 A2 evaluation outputs in PBS."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from rb1_analysis import measured_cost
from rb1_artifacts import read, reference
from rb_history import donor_indices
from seg_a2_data import ARMS, CONTRACT, SEEDS, OPENING_PROTOCOL_DECISION, verify_lock, confirmation_decision
from seg_a2_rules import IMPLEMENTATION_CHECKS, analyze as apply_rules
from seg_a2_descriptive import describe_a2, describe_dev, compare_dev_to_e0027, load_e0027_dev_reference
from seg_artifacts import validate_rows as validate_dev_rows

TASKS = ('piqa', 'boolq')
CONDITIONS = ('SEG-00', *ARMS, 'SEG-05/iv_swap', 'SEG-05/iv_off')
ATTEMPT_RE = re.compile(r'ATT-\d+-\d+')


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return float(value)


def validate_prediction_rows(rows, expected=None, label='A2'):
    """Validate A2 rows and their candidate argmax, NLL, and paired order."""
    if not isinstance(rows, list):
        raise ValueError(f'{label}: predictions must be a list')
    identities = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'{label}: malformed prediction row')
        if (row.get('split') != 'a2' or row.get('task') not in TASKS or
                type(row.get('id')) is not int):
            raise ValueError(f'{label}: unexpected split/task/id')
        if type(row.get('gold')) is not int or row['gold'] not in (0, 1):
            raise ValueError(f'{label}: gold must be binary')
        scores = row.get('scores')
        if (not isinstance(scores, list) or len(scores) != 2 or
                any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
            raise ValueError(f'{label}: two finite candidate scores required')
        if type(row.get('prediction')) is not int or row['prediction'] not in (0, 1):
            raise ValueError(f'{label}: prediction must be binary')
        if row['prediction'] != (0 if scores[0] >= scores[1] else 1):
            raise ValueError(f'{label}: prediction does not follow the V1 tie rule')
        tokens = row.get('gold_loss_tokens')
        if type(tokens) is not int or tokens <= 0:
            raise ValueError(f'{label}: invalid gold answer token count')
        nll = _finite(row.get('gold_nll_sum'), f'{label}: gold NLL')
        if nll < 0:
            raise ValueError(f'{label}: negative gold NLL')
        gold_score = -nll / (tokens if row['task'] == 'piqa' else 1)
        if not math.isclose(scores[row['gold']], gold_score, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'{label}: gold candidate score differs from answer NLL')
        identities.append((row['task'], row['id'], row['gold'], tokens))
    if len({(task, row_id) for task, row_id, _, _ in identities}) != len(identities):
        raise ValueError(f'{label}: duplicate task/id')
    if len(rows) != 2000 or any(sum(row['task'] == task for row in rows) != 1000 for task in TASKS):
        raise ValueError(f'{label}: exactly 1000 rows per task are required')
    if expected is not None and identities != expected:
        raise ValueError(f'{label}: paired A2 IDs/gold/token order differs')
    return rows


def check_dev_gate(actual, expected, hardware, training_hardware):
    """Recompute the pre-opening gate using the locked same/different-slice rule."""
    validate_dev_rows(expected)
    validate_dev_rows(actual, [(row['task'], row['id'], row['gold']) for row in expected])
    if not isinstance(hardware, str) or not hardware or not isinstance(training_hardware, str) or not training_hardware:
        raise ValueError('dev gate requires recorded evaluation and training hardware')
    mismatches = sum(a['prediction'] != b['prediction'] for a, b in zip(actual, expected))
    same = hardware == training_hardware
    identical = actual == expected
    return {'passed': identical if same else mismatches <= 10,
            'same_hardware': same, 'prediction_mismatches': mismatches,
            'rows_identical': identical, 'hardware': hardware,
            'training_hardware': training_hardware}


def _summary(rows):
    result = {}
    for task in TASKS:
        subset = [row for row in rows if row['task'] == task]
        correct = sum(row['prediction'] == row['gold'] for row in subset)
        tokens = sum(row['gold_loss_tokens'] for row in subset)
        result[task] = {'correct': correct, 'total': len(subset), 'accuracy': correct / len(subset),
                        'nll': sum(row['gold_nll_sum'] for row in subset) / tokens,
                        'loss_tokens': tokens}
    result['macro_accuracy'] = sum(result[t]['accuracy'] for t in TASKS) / 2
    result['macro_nll'] = sum(result[t]['nll'] for t in TASKS) / 2
    return result


def _check_condition_metrics(rows, metrics, label):
    summary = _summary(rows)
    for task in TASKS:
        expected = {f'a2.{task}.accuracy': summary[task]['accuracy'],
                    f'a2.{task}.nll': summary[task]['nll'],
                    f'a2.{task}.examples': summary[task]['total'],
                    f'a2.{task}.loss_tokens': summary[task]['loss_tokens']}
        for key, value in expected.items():
            recorded = metrics.get(key)
            if type(recorded) not in (int, float) or not math.isfinite(recorded) or not math.isclose(
                    recorded, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'{label}: recorded metric {key} differs from predictions')
    if not math.isclose(metrics.get('a2.nll', float('nan')), summary['macro_nll'], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'{label}: recorded macro NLL differs from predictions')
    return summary


def _check_lock_checks(lock):
    pre = lock['preopening_checks']
    if pre.get('passed') is not True or not isinstance(pre.get('checks'), dict):
        raise ValueError('lock pre-opening checks need passed=true and per-seed checks')
    for field in ('deterministic', 'smoke'):
        ref = pre.get(field)
        if not isinstance(ref, dict) or reference(ref.get('path', '')) != ref:
            raise ValueError(f'pre-opening {field} producer reference changed')
        producer = read(ref['path'])
        if (producer.get('status') != 'complete' or producer.get('validator', {}).get('passed') is not True or
                re.fullmatch(r'[0-9a-f]{40}', str(producer.get('source_commit'))) is None or
                re.fullmatch(r'(?:blake2b:)?[0-9a-f]{64}', str(producer.get('spec_digest'))) is None):
            raise ValueError(f'pre-opening {field} producer did not complete under this spec')
        if field == 'deterministic' and (
                producer.get('source_commit') != lock['evaluation_source'] or
                producer.get('metrics', {}).get('no_channel_identity') is not True or
                producer.get('metrics', {}).get('no_channel_seeds') != list(SEEDS)):
            raise ValueError('deterministic producer does not bind the locked source and all new seeds')
        if field == 'smoke' and producer.get('metrics', {}).get('preopening.loader_equivalence_seeds') != [0, 1, 2]:
            raise ValueError('smoke producer lacks the locked loader equivalence check')
    if set(pre['checks']) != {str(seed) for seed in SEEDS}:
        raise ValueError('pre-opening checks must cover seeds 3, 4 and 5')
    checked = {}
    for seed in SEEDS:
        checks = pre['checks'][str(seed)]
        if not isinstance(checks, dict) or set(checks) != set(IMPLEMENTATION_CHECKS):
            raise ValueError(f'seed {seed} pre-opening implementation check set differs')
        checked[seed] = {key: checks[key] is True for key in IMPLEMENTATION_CHECKS}
        if not all(checked[seed].values()):
            raise ValueError(f'seed {seed} pre-opening implementation checks did not all pass')
    return checked


def _verify_eval(seed, root, lock, lock_path, lock_sha256, expected_ids):
    root = Path(root).resolve()
    if not root.is_dir() or ATTEMPT_RE.fullmatch(root.name) is None:
        raise ValueError(f'seed {seed} evaluation root must be a registered ATT directory')
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    metrics = manifest.get('metrics', {})
    if (manifest.get('status') != 'complete' or manifest.get('contract') != CONTRACT or
            manifest.get('arm') != 'A2' or manifest.get('seed') != seed or
            manifest.get('stage') != 'a2_evaluation' or manifest.get('validator', {}).get('passed') is not True or
            metrics.get('lock.sha256') != lock_sha256 or metrics.get('source.commit') != lock['evaluation_source']):
        raise ValueError(f'seed {seed} A2 evaluation manifest identity/lock/source is invalid')
    if metrics.get('conditions') != len(CONDITIONS) or metrics.get('a2_rows') != 2000:
        raise ValueError(f'seed {seed} A2 evaluation manifest condition/row count differs')
    if re.fullmatch(r'(?:blake2b:)?[0-9a-f]{64}', str(manifest.get('spec_digest'))) is None:
        raise ValueError(f'seed {seed} A2 evaluation spec digest is malformed')
    artifacts = {item.get('role'): item.get('path') for item in manifest.get('artifacts', [])}
    if set(artifacts) != {'dev-reproduction.json', 'conditions.json', 'provenance.json',
                          'a2-read.jsonl', 'a2-encodability.json'}:
        raise ValueError(f'seed {seed} A2 manifest artifact inventory differs')
    for role, path in artifacts.items():
        p = Path(path).resolve()
        if p != root / role or not p.is_file():
            raise ValueError(f'seed {seed} A2 artifact {role} path is invalid')
    provenance = read(root / 'provenance.json')
    decision = confirmation_decision(provenance.get('confirmation_decision'))
    if (metrics.get('confirmation_decision') != decision or metrics.get('a2.opened') is not True or
            manifest.get('opened') is not True or
            metrics.get('opening_protocol_decision') != OPENING_PROTOCOL_DECISION or
            provenance.get('opening_protocol_decision') != OPENING_PROTOCOL_DECISION):
        raise ValueError(f'seed {seed} confirmation decision or opening record differs')
    lock_ref = reference(lock_path)
    if (provenance.get('lock') != lock_ref or provenance.get('seed') != seed or
            provenance.get('training_source') != lock['training_source'] or
            provenance.get('evaluation_source') != lock['evaluation_source'] or
            provenance.get('inputs') != lock['runs'][str(seed)] or
            provenance.get('job_id') != metrics.get('exec.pbs_job_id') or
            provenance.get('hardware') != metrics.get('hardware')):
        raise ValueError(f'seed {seed} A2 provenance differs from locked source/run/receipt')
    receipt_file = root / 'a2-read.jsonl'
    receipts = [json.loads(line) for line in receipt_file.read_text().splitlines() if line.strip()]
    if len(receipts) != 1:
        raise ValueError(f'seed {seed} needs exactly one A2 read receipt')
    receipt = receipts[0]
    if (receipt.get('lock_sha256') != lock_sha256 or receipt.get('seed') != seed or
            receipt.get('arm') != 'SEG-05' or receipt.get('pbs_job_id') != provenance.get('job_id') or
            receipt.get('hardware') != provenance.get('hardware') or not receipt.get('time_utc') or
            receipt.get('confirmation_decision') != decision or
            receipt.get('opening_protocol_decision') != OPENING_PROTOCOL_DECISION):
        raise ValueError(f'seed {seed} A2 read receipt differs from its job/provenance')
    encodability = read(root/'a2-encodability.json')
    if encodability != {'passed': True, 'checked': {'piqa': 1000, 'boolq': 1000},
                       'status': 'encodable_no_model_scored_yet'}:
        raise ValueError(f'seed {seed} complete A2 encoding check missing')

    dev = read(root / 'dev-reproduction.json')
    entries = lock['runs'][str(seed)]
    expected_dev = read(entries['SEG-05']['predictions']['path'])
    expected_dev_ids = [(row['task'], row['id'], row['gold']) for row in expected_dev]
    validate_dev_rows(expected_dev)
    actual_dev = validate_dev_rows(dev.get('predictions'), expected_dev_ids)
    training_manifest = read(entries['SEG-05']['manifest']['path'])
    gate = check_dev_gate(actual_dev, expected_dev, provenance['hardware'],
                          training_manifest['metrics']['hardware'])
    if dev.get('check') != gate or metrics.get('path.dev_gate') != gate or not gate['passed']:
        raise ValueError(f'seed {seed} path dev reproduction gate failed or was altered')

    conditions = read(root / 'conditions.json')
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ValueError(f'seed {seed} A2 condition set differs from lock')
    predictions = {}
    summaries = {}
    canonical = None
    for condition in CONDITIONS:
        entry = conditions[condition]
        rows = validate_prediction_rows(entry.get('predictions'), canonical,
                                        f'seed {seed} {condition}')
        identities = [(r['task'], r['id'], r['gold'], r['gold_loss_tokens']) for r in rows]
        if canonical is None:
            canonical = identities
            if [(task, row_id) for task, row_id, _, _ in identities] != expected_ids:
                raise ValueError(f'seed {seed} {condition} IDs/order differ from the locked A2 split')
        summaries[condition] = _check_condition_metrics(rows, entry.get('metrics', {}),
                                                        f'seed {seed} {condition}')
        predictions[condition] = rows
    base_rows = predictions['SEG-00']
    off_rows = predictions['SEG-05/iv_off']
    off_identical = off_rows == base_rows
    if not off_identical or metrics.get('off.identical_to_Base_seg') is not True:
        raise ValueError(f'seed {seed} same-job A2 iv_off/Base identity failed')

    donor_rows = [{'task': row['task'], 'id': row['id']} for row in base_rows]
    donors = donor_indices(donor_rows, 'A2')
    expected_donors = [{'task': row['task'], 'recipient': row['id'],
                        'donor': donor_rows[donors[i]]['id']}
                       for i, row in enumerate(donor_rows)]
    actual_donors = conditions['SEG-05/iv_swap'].get('audit', {}).get('donors')
    if actual_donors != expected_donors:
        raise ValueError(f'seed {seed} iv_swap donor map differs from the locked A2 derangement')
    return {'root': str(root), 'manifest': reference(manifest_path),
            'artifacts': {role: reference(path) for role, path in artifacts.items()},
            'provenance': provenance, 'receipt': receipt, 'path_gate': gate,
            'predictions': predictions, 'summaries': summaries,
            'base_rows': base_rows, 'iv_off_matches_base': off_identical,
            'donor_map': actual_donors, 'confirmation_decision': decision}


def shared_confirmation_decision(verified):
    decisions = {confirmation_decision(v.get('confirmation_decision')) for v in verified.values()}
    if len(decisions) != 1:
        raise ValueError('confirmation decision differs across seeds')
    return next(iter(decisions))


def _counts(rows):
    result = {}
    for arm, predictions in rows.items():
        task_result = {}
        for task in TASKS:
            subset = [row for row in predictions if row['task'] == task]
            task_result[task] = (sum(row['prediction'] == row['gold'] for row in subset), len(subset))
        result[arm] = task_result
    return result


def _complete_cost(inputs, roots, lock):
    declared = inputs.get('accounted_attempts')
    if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared) or
            any(not isinstance(attempt, str) or ATTEMPT_RE.fullmatch(attempt) is None for attempt in declared)):
        raise ValueError('analysis requires the complete unique accounted_attempts list')
    required_paths = [Path(root).resolve() for root in roots]
    required_paths.extend(Path(entry['root']).resolve()
                          for by_arm in lock['runs'].values() for entry in by_arm.values())
    required_paths.extend(Path(lock['preopening_checks'][field]['path']).resolve().parent
                          for field in ('deterministic', 'smoke'))
    required_attempts = {path.name for path in required_paths}
    if any(ATTEMPT_RE.fullmatch(attempt) is None for attempt in required_attempts):
        raise ValueError('locked run, smoke, and pre-opening check paths must name registered ATT attempts')
    if not required_attempts <= set(declared):
        raise ValueError('accounting list omits a training, pre-opening, or A2 evaluation attempt')
    cost = measured_cost(inputs['observations'])
    if {row['attempt'] for row in cost['rows']} != set(declared):
        raise ValueError('measured terminal cost coverage differs from accounted_attempts')
    cost.update({'required_attempts': sorted(declared), 'terminal_coverage_complete': True,
                 'cap_node_h': 5.5, 'within_cap': cost['node_h'] <= 5.5,
                 'current_analysis_usage': 'pending; add terminal measured usage in delivery'})
    return cost


def analyze(inputs, lock_path, lock_sha256):
    if not isinstance(inputs, dict) or set(inputs) != {'evaluations', 'observations', 'accounted_attempts'}:
        raise ValueError('analysis input needs evaluations, observations, and accounted_attempts')
    if not isinstance(inputs['evaluations'], dict) or set(inputs['evaluations']) != {str(s) for s in SEEDS}:
        raise ValueError('analysis needs one A2 evaluation root for each seed 3, 4 and 5')
    lock_path = Path(lock_path).resolve()
    lock = verify_lock(lock_path, lock_sha256)
    lock_checks = _check_lock_checks(lock)
    split = read(lock['split_manifest']['path'])
    ids = split.get('ids')
    if not isinstance(ids, dict) or set(ids) != set(TASKS) or any(len(ids[t]) != 1000 for t in TASKS):
        raise ValueError('locked A2 split needs 1000 ordered IDs per task')
    expected_id_order = [(task, row_id) for task in TASKS for row_id in ids[task]]
    verified = {}
    for seed in SEEDS:
        verified[seed] = _verify_eval(seed, inputs['evaluations'][str(seed)], lock, lock_path,
                                      lock_sha256, expected_id_order)
    decision = shared_confirmation_decision(verified)
    # Cross-seed A2 membership, gold labels, and tokenization must be paired.
    canonical_rows = verified[3]['base_rows']
    canonical = [(r['task'], r['id'], r['gold'], r['gold_loss_tokens']) for r in canonical_rows]
    for seed in (4, 5):
        rows = verified[seed]['base_rows']
        if [(r['task'], r['id'], r['gold'], r['gold_loss_tokens']) for r in rows] != canonical:
            raise ValueError(f'seed {seed} A2 Base rows differ from seed 3 paired membership/gold')

    counts = {}
    interval_rows = {}
    checks = {}
    seed_reports = {}
    dev_predictions = {seed: {arm: validate_dev_rows(read(lock['runs'][str(seed)][arm]['predictions']['path']))
                              for arm in ARMS} for seed in SEEDS}
    for seed in SEEDS:
        predictions = verified[seed]['predictions']
        counts[seed] = {arm: _counts({arm: predictions[arm]})[arm]['boolq'] for arm in ARMS}
        counts[seed]['SEG-05/iv_swap'] = _counts({'SEG-05/iv_swap': predictions['SEG-05/iv_swap']})['SEG-05/iv_swap']['boolq']
        interval_rows[seed] = {arm: [int(row['prediction'] == row['gold']) for row in predictions[arm]
                                     if row['task'] == 'boolq']
                               for arm in (*ARMS, 'SEG-05/iv_swap')}
        check = dict(lock_checks[seed])
        check.update({'path_dev_same_hardware': verified[seed]['path_gate']['same_hardware'],
                      'path_dev_mismatch_count': verified[seed]['path_gate']['prediction_mismatches'],
                      'a2_iv_off_matches_base': verified[seed]['iv_off_matches_base']})
        checks[seed] = check
        seed_reports[str(seed)] = {
            'conditions': verified[seed]['summaries'],
            'dev_training': {
                arm: _summary(dev_predictions[seed][arm])
                for arm in ARMS
            },
            'boolq_correct_counts': {arm: list(counts[seed][arm]) for arm in counts[seed]},
            'path_dev_gate': verified[seed]['path_gate'],
            'iv_off_matches_base': verified[seed]['iv_off_matches_base'],
            'iv_swap_donors': verified[seed]['donor_map'],
            'provenance': verified[seed]['provenance'],
            'receipt': verified[seed]['receipt'],
            'artifact_refs': {'manifest': verified[seed]['manifest'], **verified[seed]['artifacts']},
        }
    base_count = _counts({'SEG-00': verified[3]['base_rows']})['SEG-00']['boolq']
    result = apply_rules(counts, checks, base_count, interval_rows)
    base_mismatches = {str(seed): sum(
        a['prediction'] != b['prediction'] for a, b in zip(verified[3]['base_rows'], verified[seed]['base_rows']))
        for seed in SEEDS}
    roots = [inputs['evaluations'][str(seed)] for seed in SEEDS]
    cost = _complete_cost(inputs, roots, lock)
    a2_descriptive = describe_a2({seed: {arm: verified[seed]['predictions'][arm]
                                        for arm in (*ARMS, 'SEG-05/iv_swap')} for seed in SEEDS})
    dev_descriptive = describe_dev(dev_predictions)
    prior_dev = load_e0027_dev_reference()
    return {**result, 'contract': CONTRACT, 'hypothesis': 'H-0029@1', 'stage': 'final',
            'lock': {'path': str(lock_path), 'sha256': lock_sha256}, 'seeds': seed_reports,
            'confirmation_decision': decision,
            'descriptive': {'a2': a2_descriptive, 'dev': dev_descriptive,
                'e0027_dev_reference': prior_dev,
                'dev_difference_from_e0027': compare_dev_to_e0027(dev_descriptive, prior_dev)},
            'base_seg_seed3_boolq': {'correct': base_count[0], 'total': base_count[1]},
            'cross_job_base_prediction_mismatches_vs_seed3': base_mismatches,
            'cost': cost,
            'limitations': [
                'The paired 95% intervals quantify A2 BoolQ row sampling only; they exclude variation across the three trained seeds.',
                'Only three new seeds were evaluated. The confirmatory claim is limited to A2 BoolQ at the locked r128 setting.',
                'PIQA, macro accuracy, NLL, per-seed results, and dev comparisons are descriptive.',
                'No hypothesis status is changed by this analysis.',
            ]}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--lock-sha256', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    passed = False
    try:
        report = analyze(read(args.input), args.lock, args.lock_sha256)
        analysis_path = output / 'analysis.json'
        _write_json(analysis_path, report)
        digest = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
        (output / 'analysis.sha256').write_text(f'{digest}  analysis.json\n')
        passed = True
    finally:
        _write_json(output / 'manifest.json', {
            'status': 'complete' if passed else 'partial', 'contract': CONTRACT,
            'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'metrics': {'elapsed_s': time.monotonic() - started, 'bootstrap.repeats': 10000,
                        'bootstrap.seed': 20261101, 'confirmation_evaluated': passed,
                        'lock.sha256': args.lock_sha256},
            'artifacts': [{'path': str(output / name), 'role': name}
                          for name in ('analysis.json', 'analysis.sha256') if (output / name).is_file()],
            'validator': {'name': 'segmented_locked_a2_analysis', 'passed': passed},
        })


if __name__ == '__main__':
    main()
