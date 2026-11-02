"""Synthetic row and path-gate checks for the locked E-0030 analysis."""
import copy
import json
from pathlib import Path

import pytest

import seg_a2_analysis as analysis
from rb1_artifacts import reference
from seg_a2_analysis import check_dev_gate, validate_prediction_rows


def _rows(split='a2'):
    result = []
    for task in ('piqa', 'boolq'):
        for row_id in range(1000):
            gold = row_id % 2
            scores = [-1.0, -2.0]
            scores[gold] = -1.0
            scores[1 - gold] = -2.0
            result.append({'task': task, 'split': split, 'id': row_id, 'gold': gold,
                           'prediction': gold, 'scores': scores, 'gold_nll_sum': 1.0,
                           'gold_loss_tokens': 1})
    return result


def _dev_rows():
    rows = _rows('dev')
    for row in rows:
        row['prediction'] = 0 if row['scores'][0] >= row['scores'][1] else 1
    return rows


def test_a2_predictions_validate_finite_argmax_and_locked_row_order():
    rows = _rows()
    expected = [(row['task'], row['id'], row['gold'], row['gold_loss_tokens']) for row in rows]
    assert validate_prediction_rows(rows, expected) == rows

    altered = copy.deepcopy(rows)
    altered[0]['scores'][0] = float('nan')
    with pytest.raises(ValueError, match='finite candidate scores'):
        validate_prediction_rows(altered)

    altered = copy.deepcopy(rows)
    altered[0]['prediction'] = 1
    with pytest.raises(ValueError, match='tie rule'):
        validate_prediction_rows(altered)

    with pytest.raises(ValueError, match='paired A2'):
        validate_prediction_rows(rows, list(reversed(expected)))


def test_a2_prediction_requires_gold_likelihood_consistency():
    rows = _rows()
    rows[0]['gold_nll_sum'] = 9.0
    with pytest.raises(ValueError, match='answer NLL'):
        validate_prediction_rows(rows)


def test_path_gate_recomputes_same_and_different_hardware_outcomes():
    expected = _dev_rows()
    assert check_dev_gate(expected, expected, 'MIG-1g', 'MIG-1g')['passed'] is True

    changed = copy.deepcopy(expected)
    for row in changed[:10]:
        row['prediction'] = 1 - row['prediction']
        row['scores'].reverse()
    gate = check_dev_gate(changed, expected, 'MIG-2g', 'MIG-1g')
    assert gate['passed'] is True and gate['prediction_mismatches'] == 10

    changed[10]['prediction'] = 1 - changed[10]['prediction']
    changed[10]['scores'].reverse()
    gate = check_dev_gate(changed, expected, 'MIG-2g', 'MIG-1g')
    assert gate['passed'] is False and gate['prediction_mismatches'] == 11


def test_eval_authentication_accepts_distinct_registered_spec_and_mixed_profile_gate(tmp_path, monkeypatch):
    lock_path = tmp_path / 'lock.json'
    lock_path.write_text('{}')
    lock_ref = reference(lock_path)
    training_root = tmp_path / 'ATT-0001-1'
    training_root.mkdir()
    expected_dev = _dev_rows()
    expected_dev_path = training_root / 'predictions.json'
    expected_dev_path.write_text(json.dumps(expected_dev))
    training_manifest_path = training_root / 'manifest.json'
    training_manifest_path.write_text(json.dumps({'metrics': {'hardware': 'MIG-1g'}}))
    entry = {
        'predictions': {'path': str(expected_dev_path)},
        'manifest': {'path': str(training_manifest_path)},
    }

    actual_dev = copy.deepcopy(expected_dev)
    for row in actual_dev[:10]:
        row['prediction'] = 1 - row['prediction']
        row['scores'].reverse()
    gate = check_dev_gate(actual_dev, expected_dev, 'MIG-2g', 'MIG-1g')
    assert gate['passed'] is True and gate['prediction_mismatches'] == 10

    a2_root = tmp_path / 'ATT-0002-1'
    a2_root.mkdir()
    conditions = {}
    a2_rows = _rows('a2')
    for label in analysis.CONDITIONS:
        conditions[label] = {
            'metrics': {
                'a2.piqa.accuracy': 1.0, 'a2.piqa.nll': 1.0,
                'a2.piqa.examples': 1000, 'a2.piqa.loss_tokens': 1000,
                'a2.boolq.accuracy': 1.0, 'a2.boolq.nll': 1.0,
                'a2.boolq.examples': 1000, 'a2.boolq.loss_tokens': 1000,
                'a2.nll': 1.0,
            },
            'predictions': a2_rows,
            'audit': {'donors': []},
        }
    conditions['SEG-05/iv_swap']['audit']['donors'] = [
        {'task': row['task'], 'recipient': row['id'], 'donor': row['id']} for row in a2_rows]
    monkeypatch.setattr(analysis, 'donor_indices', lambda rows, phase: list(range(len(rows))))

    dev_path = a2_root / 'dev-reproduction.json'
    dev_path.write_text(json.dumps({'check': gate, 'predictions': actual_dev}))
    conditions_path = a2_root / 'conditions.json'
    conditions_path.write_text(json.dumps(conditions))
    provenance = {
        'lock': lock_ref, 'seed': 3, 'training_source': 'a' * 40,
        'evaluation_source': 'b' * 40, 'inputs': {'SEG-05': entry},
        'job_id': '123.server', 'hardware': 'MIG-2g',
        'confirmation_decision': 'D-9999',
    }
    provenance_path = a2_root / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance))
    receipt_path = a2_root / 'a2-read.jsonl'
    receipt_path.write_text(json.dumps({
        'lock_sha256': lock_ref['sha256'], 'seed': 3, 'arm': 'SEG-05',
        'pbs_job_id': '123.server', 'hardware': 'MIG-2g',
        'time_utc': '2026-11-03T00:00:00+00:00',
        'confirmation_decision': 'D-9999',
    }) + '\n')
    encoding_path = a2_root/'a2-encodability.json'
    encoding_path.write_text(json.dumps({'passed': True, 'checked': {'piqa': 1000, 'boolq': 1000},
                                         'status': 'encodable_no_model_scored_yet'}))
    artifacts = [{'role': path.name, 'path': str(path)} for path in
                 (dev_path, conditions_path, provenance_path, receipt_path, encoding_path)]
    actual_lock_sha = lock_ref['sha256']
    eval_manifest = {
        'status': 'complete', 'contract': analysis.CONTRACT, 'arm': 'A2', 'seed': 3,
        'stage': 'a2_evaluation', 'spec_digest': 'c' * 64,
        'metrics': {'lock.sha256': actual_lock_sha, 'source.commit': 'b' * 40,
                    'conditions': len(analysis.CONDITIONS), 'a2_rows': 2000,
                    'exec.pbs_job_id': '123.server', 'hardware': 'MIG-2g',
                    'path.dev_gate': gate, 'off.identical_to_Base_seg': True},
        'artifacts': artifacts, 'validator': {'passed': True},
    }
    eval_manifest['metrics'].update({'confirmation_decision': 'D-9999', 'a2.opened': True})
    manifest_path = a2_root / 'manifest.json'
    manifest_path.write_text(json.dumps(eval_manifest))
    lock = {'training_source': 'a' * 40, 'evaluation_source': 'b' * 40,
            'runs': {'3': {'SEG-05': entry}}}
    expected_ids = [(row['task'], row['id']) for row in a2_rows]

    result = analysis._verify_eval(3, a2_root, lock, lock_path, actual_lock_sha, expected_ids)
    assert result['path_gate']['passed'] is True
    assert result['path_gate']['same_hardware'] is False
    assert result['path_gate']['prediction_mismatches'] == 10

    receipt = json.loads(receipt_path.read_text())
    del receipt['confirmation_decision']
    receipt_path.write_text(json.dumps(receipt)+'\n')
    with pytest.raises(ValueError, match='receipt differs'):
        analysis._verify_eval(3, a2_root, lock, lock_path, actual_lock_sha, expected_ids)
    receipt['confirmation_decision'] = 'D-9998'
    receipt_path.write_text(json.dumps(receipt)+'\n')
    with pytest.raises(ValueError, match='receipt differs'):
        analysis._verify_eval(3, a2_root, lock, lock_path, actual_lock_sha, expected_ids)
    receipt['confirmation_decision'] = 'D-9999'
    receipt_path.write_text(json.dumps(receipt)+'\n')

    eval_manifest['spec_digest'] = ''
    manifest_path.write_text(json.dumps(eval_manifest))
    with pytest.raises(ValueError, match='spec digest'):
        analysis._verify_eval(3, a2_root, lock, lock_path, actual_lock_sha, expected_ids)


def test_confirmation_decision_must_match_across_all_seeds():
    verified = {s: {'confirmation_decision': 'D-9999'} for s in (3, 4, 5)}
    assert analysis.shared_confirmation_decision(verified) == 'D-9999'
    verified[4]['confirmation_decision'] = 'D-9998'
    with pytest.raises(ValueError, match='differs across seeds'):
        analysis.shared_confirmation_decision(verified)
    del verified[4]['confirmation_decision']
    with pytest.raises(ValueError, match='confirmation decision'):
        analysis.shared_confirmation_decision(verified)
