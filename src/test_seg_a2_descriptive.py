"""Synthetic exact-statistic tests for locked E-0030 descriptives."""
import hashlib
import json

import pytest

import seg_a2_descriptive as descriptive


def rows_with_correct(correct, total=10):
    rows = []
    for task in ('piqa', 'boolq'):
        for row_id in range(total):
            gold = row_id % 2
            prediction = gold if row_id < correct[task] else 1 - gold
            rows.append({'task': task, 'id': row_id, 'gold': gold, 'prediction': prediction})
    return rows


def a2_seed(piqa, boolq):
    correct = {
        'SEG-04': {'piqa': 6, 'boolq': 6},
        'SEG-05': {'piqa': piqa, 'boolq': boolq},
        'SEG-06': {'piqa': 6, 'boolq': 6},
        'SEG-07': {'piqa': 5, 'boolq': 5},
        'SEG-05/iv_swap': {'piqa': 4, 'boolq': 4},
    }
    return {arm: rows_with_correct(scores) for arm, scores in correct.items()}


def test_exact_task_and_macro_gain_use_best_control_after_macro_scores():
    conditions = {}
    correct = {
        'SEG-04': {'piqa': 7, 'boolq': 7},
        'SEG-05': {'piqa': 8, 'boolq': 8},
        'SEG-06': {'piqa': 9, 'boolq': 5},
        'SEG-07': {'piqa': 7, 'boolq': 9},
        'SEG-05/iv_swap': {'piqa': 6, 'boolq': 6},
    }
    conditions[3] = {arm: rows_with_correct(scores) for arm, scores in correct.items()}
    conditions[4] = a2_seed(7, 7)
    conditions[5] = a2_seed(9, 9)
    result = descriptive.describe_a2(conditions)
    seed3 = result['per_seed']['3']
    assert seed3['piqa']['gain'] == {'numerator': -1, 'denominator': 10, 'pp': -10.0}
    assert seed3['boolq']['gain'] == {'numerator': -1, 'denominator': 10, 'pp': -10.0}
    # Macro scores are 0.8 Keep vs max(0.7, 0.8) for the controls: zero gain.
    assert seed3['macro']['gain'] == {'numerator': 0, 'denominator': 1, 'pp': 0.0}
    assert seed3['macro']['lora'] == {'numerator': 1, 'denominator': 10, 'pp': 10.0}
    assert 'swap' in seed3['macro']


def test_means_weight_seeds_equally_and_dev_omits_swap():
    a2 = {3: a2_seed(7, 7), 4: a2_seed(8, 8), 5: a2_seed(10, 10)}
    result = descriptive.describe_a2(a2)
    assert result['mean']['boolq']['gain'] == {'numerator': 7, 'denominator': 30,
                                               'pp': pytest.approx(23.3333333333)}

    dev = {seed: {arm: a2[seed][arm] for arm in descriptive.ARMS} for seed in (3, 4, 5)}
    dev_result = descriptive.describe_dev(dev)
    assert dev_result['mean']['piqa']['gain'] == {'numerator': 7, 'denominator': 30,
                                                  'pp': pytest.approx(23.3333333333)}
    assert dev_result['swap'] == 'not available on dev'
    assert all(set(metrics) == {'gain', 'lora'} for seed in dev_result['per_seed'].values()
               for metrics in seed.values())
    compared = descriptive.compare_dev_to_e0027(dev_result, result)
    assert compared['piqa']['gain']['difference']['numerator'] == 0


def _reference_report():
    report = {'contract': 'E-0027@1', 'hypothesis': 'H-0020@1', 'stage': 'final', 'seeds': {}}
    for seed, keep in zip((0, 1, 2), (7, 8, 10)):
        labels = {
            'SEG-04': 6, 'SEG-05': keep, 'SEG-06': 6, 'SEG-07': 5,
            'SEG-05/normal': keep, 'SEG-05/iv_swap': 4,
        }
        summary = {arm: {
            task: {'correct': count*100, 'total': 1000}
            for task in ('piqa', 'boolq')
        } for arm, count in labels.items()}
        record = {'summary': summary}
        if seed in (1, 2):
            record['training'] = {arm: {'source.commit': descriptive.TRAINING_SOURCE}
                                  for arm in descriptive.ARMS}
        report['seeds'][str(seed)] = record
    return report


def test_pinned_e0027_reference_loader_verifies_source_and_exact_mean(tmp_path, monkeypatch):
    report_path = tmp_path / 'analysis.json'
    payload = json.dumps(_reference_report(), sort_keys=True).encode()
    report_path.write_bytes(payload)
    monkeypatch.setattr(descriptive, 'E0027_ANALYSIS', {
        **descriptive.E0027_ANALYSIS,
        'path': str(report_path), 'sha256': hashlib.sha256(payload).hexdigest(),
        'bytes': len(payload),
    })
    result = descriptive.load_e0027_dev_reference()
    assert result['contract'] == 'E-0027@1'
    assert result['source_commit'] == descriptive.TRAINING_SOURCE
    assert result['mean']['boolq']['gain'] == {'numerator': 7, 'denominator': 30,
                                                'pp': pytest.approx(23.3333333333)}

    report_path.write_text('{}')
    with pytest.raises(ValueError, match='hash/size'):
        descriptive.load_e0027_dev_reference()
