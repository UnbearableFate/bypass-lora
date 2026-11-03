"""Locked E-0030 descriptive accuracy contrasts and E-0027 dev reference."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

TASKS = ('piqa', 'boolq')
SEEDS = (3, 4, 5)
REFERENCE_SEEDS = (0, 1, 2)
ARMS = ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07')
SWAP = 'SEG-05/iv_swap'
E0027_ARMS = (*ARMS, 'SEG-05/normal', SWAP)
TRAINING_SOURCE = '054a54a8d4c39cbf43b1fcee681b3e4b49bc35e7'
E0027_ANALYSIS = {
    'path': '/work/xg24i002/x10041/bypass-lora/runs/ATT-0339-1/analysis.json',
    'sha256': 'f8d4916b3740a751884383b9303541f6fad6edd7192fe8a8683f083acf68b440',
    'bytes': 128960,
    'contract': 'E-0027@1',
    'stage': 'final',
    'source_commit': TRAINING_SOURCE,
}


def _fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'pp': float(100 * value)}


def _accuracy(rows, label):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{label}: nonempty prediction rows required')
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'{label}: prediction rows must be mappings')
    scores = {}
    identities = []
    for task in TASKS:
        subset = [row for row in rows if row.get('task') == task]
        if not subset:
            raise ValueError(f'{label}: both PIQA and BoolQ rows are required')
        if any(type(row.get('gold')) is not int or type(row.get('prediction')) is not int or
               row['gold'] not in (0, 1) or row['prediction'] not in (0, 1) for row in subset):
            raise ValueError(f'{label}: gold and prediction must be binary integers')
        scores[task] = Fraction(sum(row['gold'] == row['prediction'] for row in subset), len(subset))
    for row in rows:
        if row.get('task') not in TASKS or type(row.get('id')) is not int:
            raise ValueError(f'{label}: unexpected task/id')
        identities.append((row['task'], row['id'], row['gold']))
    if len(set((task, row_id) for task, row_id, _ in identities)) != len(identities):
        raise ValueError(f'{label}: duplicate task/id')
    scores['macro'] = (scores['piqa'] + scores['boolq']) / 2
    return scores, identities


def describe_a2(condition_rows):
    """Compute per-seed and equal-weight means for A2 gain, LoRA, and swap.

    Input is ``{seed: {condition: prediction_rows}}`` for seeds 3, 4, 5.
    Conditions are SEG-04..07 and SEG-05/iv_swap; SEG-05 is the normal Keep
    condition. Accuracy, max-control, macro, and all contrasts are evaluated
    as exact Fractions before serialization.
    """
    if set(condition_rows) != set(SEEDS):
        raise ValueError('A2 descriptives require seeds 3, 4 and 5')
    per_seed = {}
    for seed in SEEDS:
        by_condition = condition_rows[seed]
        if set(by_condition) != {*ARMS, SWAP}:
            raise ValueError(f'seed {seed} requires SEG-04..07 and SEG-05/iv_swap')
        scores = {}
        paired = None
        for condition, rows in by_condition.items():
            scores[condition], identities = _accuracy(rows, f'seed {seed} {condition}')
            if paired is None:
                paired = identities
            elif identities != paired:
                raise ValueError(f'seed {seed}: A2 conditions are not paired in the same row order')
        scores['SEG-05/normal'] = scores['SEG-05']
        per_seed[str(seed)] = {}
        for task in (*TASKS, 'macro'):
            keep = scores['SEG-05'][task]
            per_seed[str(seed)][task] = {
                'gain': _fraction(keep - max(scores['SEG-06'][task], scores['SEG-07'][task])),
                'lora': _fraction(keep - scores['SEG-04'][task]),
                'swap': _fraction(scores['SEG-05/normal'][task] - scores[SWAP][task]),
            }
    means = {task: {
        metric: _fraction(sum((Fraction(per_seed[str(seed)][task][metric]['numerator'],
                                        per_seed[str(seed)][task][metric]['denominator'])
                              for seed in SEEDS), Fraction()) / len(SEEDS))
        for metric in ('gain', 'lora', 'swap')
    } for task in (*TASKS, 'macro')}
    return {'per_seed': per_seed, 'mean': means, 'seed_weights': 'equal over seeds 3,4,5'}


def describe_dev(condition_rows):
    """Compute per-seed and equal-weight dev gain/LoRA; dev swap is undefined."""
    if set(condition_rows) != set(SEEDS):
        raise ValueError('dev descriptives require seeds 3, 4 and 5')
    per_seed = {}
    for seed in SEEDS:
        by_condition = condition_rows[seed]
        if set(by_condition) != set(ARMS):
            raise ValueError(f'seed {seed} dev inputs require only SEG-04..07')
        scores, paired = {}, None
        for condition, rows in by_condition.items():
            scores[condition], identities = _accuracy(rows, f'seed {seed} dev {condition}')
            if paired is None:
                paired = identities
            elif identities != paired:
                raise ValueError(f'seed {seed}: dev conditions are not paired in the same row order')
        per_seed[str(seed)] = {}
        for task in (*TASKS, 'macro'):
            keep = scores['SEG-05'][task]
            per_seed[str(seed)][task] = {
                'gain': _fraction(keep - max(scores['SEG-06'][task], scores['SEG-07'][task])),
                'lora': _fraction(keep - scores['SEG-04'][task]),
            }
    means = {task: {
        metric: _fraction(sum((Fraction(per_seed[str(seed)][task][metric]['numerator'],
                                        per_seed[str(seed)][task][metric]['denominator'])
                              for seed in SEEDS), Fraction()) / len(SEEDS))
        for metric in ('gain', 'lora')
    } for task in (*TASKS, 'macro')}
    return {'per_seed': per_seed, 'mean': means, 'seed_weights': 'equal over seeds 3,4,5',
            'swap': 'not available on dev'}


def compare_dev_to_e0027(dev, reference):
    """Return exact differences between E-0030 dev means and pinned E-0027 means."""
    comparison = {}
    for task in (*TASKS, 'macro'):
        comparison[task] = {}
        for metric in ('gain', 'lora'):
            current = Fraction(dev['mean'][task][metric]['numerator'],
                               dev['mean'][task][metric]['denominator'])
            prior = Fraction(reference['mean'][task][metric]['numerator'],
                             reference['mean'][task][metric]['denominator'])
            comparison[task][metric] = {
                'e0030_dev': dev['mean'][task][metric],
                'e0027_dev': reference['mean'][task][metric],
                'difference': _fraction(current - prior),
            }
    return comparison


def load_e0027_dev_reference(path=None):
    """Load only the pinned E-0027 analysis summaries; never opens dataset rows."""
    path = Path(E0027_ANALYSIS['path'] if path is None else path).resolve()
    data = path.read_bytes()
    if len(data) != E0027_ANALYSIS['bytes'] or hashlib.sha256(data).hexdigest() != E0027_ANALYSIS['sha256']:
        raise ValueError('E-0027 analysis reference hash/size differs from ATT-0339-1 pin')
    report = json.loads(data)
    if (report.get('contract') != E0027_ANALYSIS['contract'] or
            report.get('stage') != E0027_ANALYSIS['stage'] or
            report.get('hypothesis') != 'H-0020@1'):
        raise ValueError('pinned E-0027 analysis identity differs')
    per_seed = {}
    for seed in REFERENCE_SEEDS:
        key = str(seed)
        seed_record = report.get('seeds', {}).get(key)
        if not isinstance(seed_record, dict) or not isinstance(seed_record.get('summary'), dict):
            raise ValueError(f'E-0027 analysis lacks seed {seed} dev summaries')
        summary = seed_record['summary']
        if not set(E0027_ARMS) <= set(summary):
            raise ValueError(f'E-0027 seed {seed} dev summary arm set is incomplete')
        if seed in (1, 2):
            training = seed_record.get('training', {})
            if set(training) != set(ARMS) or any(
                    metrics.get('source.commit') != TRAINING_SOURCE for metrics in training.values()):
                raise ValueError(f'E-0027 seed {seed} training source is not the pinned 054a54a release')
        accuracies = {}
        for arm in E0027_ARMS:
            accuracies[arm] = {}
            for task in TASKS:
                item = summary[arm].get(task, {})
                correct, total = item.get('correct'), item.get('total')
                if (type(correct) is not int or type(total) is not int or total != 1000 or
                        not 0 <= correct <= total):
                    raise ValueError(f'E-0027 seed {seed} {arm}/{task} accuracy counts are invalid')
                accuracies[arm][task] = Fraction(correct, total)
            accuracies[arm]['macro'] = (accuracies[arm]['piqa'] + accuracies[arm]['boolq']) / 2
        per_seed[key] = {}
        for task in (*TASKS, 'macro'):
            keep = accuracies['SEG-05'][task]
            per_seed[key][task] = {
                'gain': _fraction(keep - max(accuracies['SEG-06'][task], accuracies['SEG-07'][task])),
                'lora': _fraction(keep - accuracies['SEG-04'][task]),
                'swap': _fraction(accuracies['SEG-05/normal'][task] - accuracies[E0027_ARMS[-1]][task]),
            }
    means = {task: {
        metric: _fraction(sum((Fraction(per_seed[str(seed)][task][metric]['numerator'],
                                        per_seed[str(seed)][task][metric]['denominator'])
                              for seed in REFERENCE_SEEDS), Fraction()) / len(REFERENCE_SEEDS))
        for metric in ('gain', 'lora', 'swap')
    } for task in (*TASKS, 'macro')}
    return {'source': {key: E0027_ANALYSIS[key] for key in ('path', 'sha256', 'bytes')},
            'contract': E0027_ANALYSIS['contract'], 'source_commit': TRAINING_SOURCE,
            'per_seed': per_seed, 'mean': means, 'seed_weights': 'equal over seeds 0,1,2',
            'accuracy_nll_summaries': {str(s): report['seeds'][str(s)]['summary'] for s in REFERENCE_SEEDS}}
