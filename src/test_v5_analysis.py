"""Synthetic contract tests for V5 paired analysis."""
import json
from fractions import Fraction

import numpy as np
import pytest

from v5_analysis import ARMS, SEEDS, TASKS, AnalysisError, _holm, _paired_bootstrap, analyze
from v5_analysis import classify_primary


def _build(tmp_path, c_good, l_good, *, f_good=None, wiki=False, mismatch=None, opposite=False,
           split='dev', omit_base=False):
    """Write a compact artifact set; each value is the correct-row count."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    f_good = l_good if f_good is None else f_good
    entries = []
    n_by_task = {'piqa': 1838, 'boolq': 3270} if split == 'validation' else {'piqa': 40, 'boolq': 40}
    for arm, good in (('V5-C', c_good), ('V5-F', f_good), ('V5-L', l_good)):
        for seed in (1, 2, 3):
            rows = []
            for task in ('piqa', 'boolq'):
                n = n_by_task[task]
                for i in range(n):
                    correct = i < good if task == 'piqa' or not opposite else i < n - good
                    ident = f'{task}-{i}'
                    if mismatch == (arm, seed, task) and i == 0:
                        ident += '-excluded-mismatch'
                    rows.append({'task': task, 'split': split, 'id': ident, 'gold': i % 2,
                                 'prediction': (i % 2) if correct else (1 - i % 2),
                                 'scores': [0., -1.], 'gold_nll_sum': float(i + 1),
                                 'gold_loss_tokens': i + 1})
            pred_path = tmp_path / f'{arm}-{seed}.jsonl'
            pred_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            entry = {'arm': arm, 'seed': seed, 'predictions': pred_path.name}
            if split == 'validation':
                metrics_path = tmp_path / f'{arm}-{seed}-metrics.json'
                metrics_path.write_text(json.dumps({'validation': {
                    task: {'raw_examples': n_by_task[task], 'examples': n_by_task[task],
                           'excluded_ids': [], 'protocol_deviation': False} for task in ('piqa', 'boolq')}}))
                entry['metrics'] = metrics_path.name
            entries.append(entry)
    wiki_data = None
    if wiki or split == 'validation':
        rows = [{'task': task, 'split': split, 'id': f'{task}-{i}', 'gold': i % 2,
                 'prediction': i % 2, 'scores': [0., -1.], 'gold_nll_sum': float(i + 1),
                 'gold_loss_tokens': i + 1} for task in ('piqa', 'boolq')
                 for i in range(n_by_task[task])]
        base_path = tmp_path / 'Base.json'
        base_path.write_text(json.dumps(rows))
        if not omit_base:
            base_entry = {'arm': 'Base', 'seed': None, 'predictions': base_path.name}
            if split == 'validation':
                metrics_path = tmp_path / 'Base-metrics.json'
                metrics_path.write_text(json.dumps({'validation': {
                    task: {'raw_examples': n_by_task[task], 'examples': n_by_task[task],
                           'excluded_ids': [], 'protocol_deviation': False} for task in ('piqa', 'boolq')}}))
                base_entry['metrics'] = metrics_path.name
            entries.append(base_entry)
        wiki_data = {'Base': {'nll': 2.0}} if (wiki or split == 'validation') else None
        if split == 'validation':
            for arm, level in (('V5-C', 2.2), ('V5-F', 1.8), ('V5-L', 2.1)):
                wiki_data[arm] = {str(seed): level for seed in (1, 2, 3)}
        elif wiki and not omit_base:
            # Dev manifests may include the Base reference only.
            pass
    manifest = {'entries': entries}
    if wiki_data is not None:
        manifest['wikitext'] = wiki_data
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def test_supported_challenged_and_inconclusive_boundaries(tmp_path):
    supported = analyze(_build(tmp_path / 'supported', 36, 4), 'dev', repetitions=1000)
    assert supported['primary']['decision'] == 'supported'
    assert supported['primary']['delta'] >= .01
    assert supported['primary']['ci95'][0] > 0

    challenged_dir = tmp_path / 'challenged'
    challenged_dir.mkdir()
    challenged = analyze(_build(challenged_dir, 4, 36), 'dev', repetitions=1000)
    assert challenged['primary']['decision'] == 'challenged'
    assert challenged['primary']['ci95'][1] < .01

    uncertain_dir = tmp_path / 'uncertain'
    uncertain_dir.mkdir()
    uncertain = analyze(_build(uncertain_dir, 21, 20), 'dev', repetitions=1000)
    assert uncertain['primary']['decision'] == 'inconclusive'


def test_bootstrap_is_paired_and_task_stratified(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    path = _build(tmp_path, 25, 25, f_good=15, opposite=True)
    result = analyze(path, 'dev', repetitions=1000)
    # Identical C/L correctness at each ID must cancel in every shared-index draw.
    assert result['primary']['ci95'] == [0., 0.]
    # PIQA and BoolQ have opposite correctness ordering, so the task differences
    # retain their stratum-specific signs rather than being pooled row-wise.
    s1 = result['secondary']['S1']
    assert s1['task_ci95']['piqa'][0] > 0
    assert s1['task_ci95']['boolq'][1] < 0


def test_bootstrap_matches_replicate_major_piqa_then_boolq_reference():
    arrays = {}
    for arm_index, arm in enumerate(ARMS):
        for seed in SEEDS:
            arrays[(arm, seed)] = {}
            for task_index, task in enumerate(TASKS):
                correct = np.asarray([(i + arm_index + seed + task_index) % (arm_index + 2) == 0
                                      for i in range(7)], dtype=np.float64)
                arrays[(arm, seed)][task] = {'correct': correct}
    got = _paired_bootstrap(arrays, repetitions=25, chunk_size=4)
    got_other_chunk = _paired_bootstrap(arrays, repetitions=25, chunk_size=11)
    assert all(np.array_equal(got[name]['samples'], got_other_chunk[name]['samples'])
               for name in got)

    rng = np.random.default_rng(20261029)
    expected = {name: [] for name in ('primary', 'S1', 'S2')}
    pairs = {'primary': ('V5-C', 'V5-L'), 'S1': ('V5-C', 'V5-F'), 'S2': ('V5-F', 'V5-L')}
    for _ in range(25):
        task_values = {}
        for task in TASKS:
            n = len(arrays[('V5-C', 1)][task]['correct'])
            ix = rng.integers(0, n, size=n)
            task_values[task] = {}
            for name, (left, right) in pairs.items():
                row_delta = np.mean([arrays[(left, seed)][task]['correct'] -
                                     arrays[(right, seed)][task]['correct'] for seed in SEEDS], axis=0)
                task_values[task][name] = float(row_delta[ix].mean())
        for name in pairs:
            expected[name].append((task_values['piqa'][name] + task_values['boolq'][name]) / 2)
    for name in expected:
        assert np.array_equal(got[name]['samples'], np.asarray(expected[name]))


def test_holm_ordering_and_step_down_rejection():
    result = _holm({'S1': .01, 'S2': .03})
    assert result['order'] == ['S1', 'S2']
    assert result['adjusted_p'] == {'S1': .02, 'S2': .03}
    assert result['reject'] == {'S1': True, 'S2': True}
    stopped = _holm({'S1': .03, 'S2': .04})
    assert stopped['reject'] == {'S1': False, 'S2': False}


def test_rejects_misaligned_ids_and_inconsistent_exclusions(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    path = _build(tmp_path, 20, 20, mismatch=('V5-L', 2, 'piqa'))
    with pytest.raises(AnalysisError, match='IDs/exclusions differ'):
        analyze(path, 'dev', repetitions=10)


def test_nll_aggregates_sums_over_tokens_and_wikitext_deltas(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    path = _build(tmp_path, 20, 20, wiki=True, split='validation')
    result = analyze(path, 'validation', repetitions=100)
    # Each task's ratio is sum(i+1)/sum(i+1), because loss sum and token
    # count are intentionally the same per row; macro is the task mean.
    task = result['arms']['V5-C']['seeds']['1']['piqa']
    assert task['gold_nll_sum'] == task['gold_loss_tokens']
    assert task['nll'] == 1.0
    wiki = result['wikitext']['delta_nll_vs_base']
    assert wiki['V5-C']['seed_mean'] == pytest.approx(.2)
    assert wiki['V5-F']['seed_mean'] == pytest.approx(-.2)
    assert result['accuracy_fractions']['V5-C']['piqa']['1'] == {'numerator': 10, 'denominator': 919}


def test_validation_requires_base_and_complete_wikitext(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    path = _build(tmp_path, 20, 20, split='validation', omit_base=True)
    with pytest.raises(AnalysisError, match='requires a Base entry'):
        analyze(path, 'validation', repetitions=10)


@pytest.mark.parametrize('delta,ci,seed_values,task_values,expected', [
    ('1/100', [.001, .02], ['1/100'] * 3, ['-1/100', '3/100'], 'supported'),
    ('1/100', [0., .02], ['1/100'] * 3, ['-1/100', '3/100'], 'inconclusive'),
    ('999/100000', [.001, .02], ['1/100'] * 3, ['0', '2/100'], 'inconclusive'),
    ('1/100', [.001, .02], ['0', '1/100', '2/100'], ['0', '2/100'], 'inconclusive'),
    ('1/100', [.001, .02], ['1/100'] * 3, ['-1001/100000', '3001/100000'], 'inconclusive'),
    ('0', [-.01, .01], ['0'] * 3, ['0'] * 2, 'inconclusive'),
    ('0', [-.01, .009999], ['0'] * 3, ['0'] * 2, 'challenged'),
])
def test_exact_inclusive_point_gates_and_strict_interval_boundaries(delta, ci, seed_values, task_values, expected):
    assert classify_primary(Fraction(delta), ci, map(Fraction, seed_values), map(Fraction, task_values)) == expected


def test_nested_metrics_preserve_validation_exclusion_audit(tmp_path):
    path = _build(tmp_path, 20, 20, split='validation')
    manifest = json.loads(path.read_text())
    for entry in manifest['entries']:
        metrics_path = tmp_path / entry['metrics']
        sidecar = json.loads(metrics_path.read_text())
        sidecar['metrics'] = {'wikitext.nll': 2.0}
        metrics_path.write_text(json.dumps(sidecar))
    assert analyze(path, 'validation', repetitions=10)['validation_audit']['piqa']['raw_examples'] == 1838


def test_validation_rejects_unreported_exclusions_and_partial_wikitext(tmp_path):
    path = _build(tmp_path, 20, 20, split='validation')
    manifest = json.loads(path.read_text())
    manifest['wikitext']['V5-L'].pop('3')
    path.write_text(json.dumps(manifest))
    with pytest.raises(AnalysisError, match='WikiText NLL requires'):
        analyze(path, 'validation', repetitions=10)
    manifest['wikitext']['V5-L']['3'] = 2.0
    path.write_text(json.dumps(manifest))
    metrics_path = tmp_path / manifest['entries'][0]['metrics']
    sidecar = json.loads(metrics_path.read_text())
    sidecar['validation']['piqa']['examples'] -= 1
    metrics_path.write_text(json.dumps(sidecar))
    with pytest.raises(AnalysisError, match='kept plus excluded'):
        analyze(path, 'validation', repetitions=10)
