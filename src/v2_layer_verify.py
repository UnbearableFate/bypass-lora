"""Independent E-0012 output audit and descriptive summary; run inside PBS.

This checks frozen artifacts, never refits probes or changes an interpretation.
It deliberately does not import the extraction/fitting implementation.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time


TAPS = ('direct_hidden', 'qknorm_pre_rope')
SPEC = 'blake2b:729189789e7a89994f917e5c7f1f28b5f346d8ab9db4a4cfa97b683f211ce70d'
FULL_COMMIT = '8cfcb1a8c696cec3af2b611bdb33a25a5b51e801'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-7, (a, b)


def expected_rows(arms):
    expected = set()
    for tap in TAPS:
        for position, target in (('tail_name_end', 'tail'), ('head_name_end', 'head'), ('question_head_end', 'qhead')):
            expected.add(('L0', tap, 'Base', position, target, 0))
        for position, target in (('delimiter', 'tail'), ('delimiter', 'head'), ('delimiter', 'relation'),
                                 ('question_head_end', 'qhead'), ('question_head_end', 'qrelation'),
                                 ('question_last', 'qhead'), ('question_last', 'qrelation')):
            expected.add(('L1', tap, 'Base', position, target, 0))
        for position in ('question_head_end', 'question_last'):
            for rank in (128, 256, 512, 1024):
                expected.add(('L1_address', tap, 'Base', position, 'fact_index', rank))
    choices = [(e['arm'], e['tap']) for e in arms] + [(f'random.{tap}', tap) for tap in TAPS]
    for arm, tap in choices:
        for position, target in (('delimiter', 'tail'), ('delimiter', 'head'), ('delimiter', 'relation'),
                                 ('question_head_end', 'qhead'), ('question_head_end', 'qrelation'),
                                 ('question_last', 'qhead'), ('question_last', 'qrelation')):
            expected.add(('L2', tap, arm, position, target, 0))
    assert len(expected) == 120
    return expected


def audit_data(project, report):
    directory = project / 'runs/ATT-0039-1/data'
    names = [e['name'] for e in json.loads((directory / 'name_pools.json').read_text())['pools']['seen']]
    assert names == report['probe_design']['label_names'] and len(names) == 92
    relations = ['emails', 'photographs', 'tutors', 'rescues']
    assert relations == report['probe_design']['label_relations']
    name_ids, relation_ids = {s: i for i, s in enumerate(names)}, {s: i for i, s in enumerate(relations)}
    counters, identities, contents, dev_lengths = {}, set(), set(), []
    for split, size in (('train', 16384), ('calibration', 2048), ('dev', 1000)):
        targets = {k: Counter() for k in ('tail', 'head', 'relation', 'qhead', 'qrelation', 'fact_index')}
        count = 0
        with (directory / f'{split}.jsonl').open() as handle:
            for index, line in enumerate(handle):
                episode = json.loads(line)
                assert episode['identity']['index'] == index and episode['identity']['split'] == split
                assert episode['episode_id'] not in identities and episode['content_sha256'] not in contents
                identities.add(episode['episode_id']); contents.add(episode['content_sha256'])
                facts, query = episode['facts'], episode['query']
                assert len(facts) == 16 and facts[query['target_fact_index']]['head'] == query['head']
                for fact in facts:
                    targets['tail'][name_ids[fact['tail']]] += 1
                    targets['head'][name_ids[fact['head']]] += 1
                    targets['relation'][relation_ids[fact['relation']]] += 1
                targets['qhead'][name_ids[query['head']]] += 1
                targets['qrelation'][relation_ids[query['relation']]] += 1
                targets['fact_index'][query['target_fact_index']] += 1
                if split == 'dev':
                    dev_lengths.append(len(episode['question_input_ids']))
                count += 1
        assert count == size
        counters[split] = targets
    return counters, dev_lengths


def audit_probes(report, arms, counters):
    actual = set()
    for row in report['probes']:
        key = (row['level'], row['tap'], row.get('arm', 'Base'), row['position'], row['target'], row.get('rank', 0))
        assert key not in actual, key
        actual.add(key)
        support = row['position'] in ('delimiter', 'head_name_end', 'tail_name_end')
        assert row['n'] == (32768 if support else 2048)
        assert isinstance(row['correct'], int) and 0 <= row['correct'] <= row['n']
        close(row['accuracy'], row['correct'] / row['n'])
        if row['level'] == 'L0':
            assert row['accuracy'] >= .9
        classes = 16 if row['level'] == 'L1_address' else (4 if 'relation' in row['target'] else 92)
        close(row['random_accuracy'], 1 / classes)
        frequency = counters['train'][row['target']]
        majority = min(frequency, key=lambda label: (-frequency[label], label))
        baseline = row['constant_baseline']
        assert baseline['class'] == majority
        assert baseline['train_count'] == frequency[majority]
        assert baseline['train_n'] == sum(frequency.values())
        assert baseline['n'] == row['n']
        assert baseline['correct'] == counters['calibration'][row['target']][majority]
        close(baseline['accuracy'], baseline['correct'] / baseline['n'])
        selected = row['selection']
        assert selected['fit_examples'] == (229376 if support else 14336)
        assert len(selected['trials']) == 12
        max_epoch = 64 if row['level'] == 'L0' else 32
        for trial in selected['trials']:
            assert trial['n'] == (32768 if support else 2048)
            assert trial['l2'] in (0., 1e-4, 1e-2)
            assert trial['epoch'] in (max_epoch // 8, max_epoch // 4, max_epoch // 2, max_epoch)
            assert 0 <= trial['correct'] <= trial['n'] and math.isfinite(trial['nll']) and trial['nll'] >= 0
            close(trial['accuracy'], trial['correct'] / trial['n'])
        best = max(selected['trials'], key=lambda s: (s['correct'] / s['n'], -s['nll'], s['l2'], -s['epoch']))
        assert selected['selected'] == best
    assert len(report['probes']) == 120 and actual == expected_rows(arms)


def audit_distribution(row, n, nonnegative=False):
    assert row['n'] == n
    quantiles = [v for _, v in sorted(row['quantiles'].items(), key=lambda kv: float(kv[0]))]
    assert all(math.isfinite(v) for v in quantiles + [row['mean']])
    assert quantiles == sorted(quantiles)
    assert quantiles[0] - 1e-6 <= row['mean'] <= quantiles[-1] + 1e-6
    if nonnegative:
        assert quantiles[0] >= 0


def audit_histogram(row, n):
    assert row['n'] == n and len(row['slot_counts']) == len(row['slot_frequencies']) == 16
    assert all(isinstance(v, int) and v >= 0 for v in row['slot_counts'])
    assert sum(row['slot_counts']) == n
    for count, frequency in zip(row['slot_counts'], row['slot_frequencies']):
        close(count / n, frequency)
    assert 0 <= row['target_count'] <= row['same_relation_count'] <= n
    close(row['target_count'] / n, row['target_rate'])
    close(row['same_relation_count'] / n, row['same_relation_rate'])


def audit_l3_l4(report, full, arms, dev_lengths):
    arm_names = {a['arm'] for a in arms}
    assert set(report['l3']) == arm_names
    for arm, row in report['l3'].items():
        assert row['train_examples'] == 4096 and row['test_examples'] == 2048 and row['ridge'] == 1.
        original = json.loads((full / 'l3' / arm / 'manifest.json').read_text())
        assert original['status'] == 'complete' and original['validator']['passed']
        assert original['spec_digest'] == SPEC
        assert original['metrics']['train_examples'] == 4096 and original['metrics']['test_examples'] == 2048
        for name in ('clean_to_clean', 'read_to_read', 'clean_to_read'):
            score = row[name]
            assert score['n'] == 2048 and 0 <= score['correct'] <= 2048
            close(score['accuracy'], score['correct'] / 2048)
            close(score['accuracy'], original['metrics'][name])
    random = {f'random.{tap}.{backend}' for tap in TAPS for backend in ('kernel_matrix', 'soft_slot')}
    assert set(report['l4']) == arm_names | random
    for arm, row in report['l4'].items():
        audit_distribution(row['key_pair_cosine'], 120000)
        assert row['key_pair_cosine']['quantiles']['0.0'] >= -1e-5
        assert row['key_pair_cosine']['quantiles']['1.0'] <= 1.00001
        for name, n in (('all', sum(dev_lengths)), ('head', 1000), ('last', 1000)):
            entry = row[name]
            audit_histogram(entry['read_argmax'], n)
            audit_distribution(entry['target_margin'], n)
            audit_distribution(entry['fusion_to_residual_norm'], n, nonnegative=True)
            audit_distribution(entry['fusion_to_post_attention_norm'], n, nonnegative=True)
        accumulated = [0] * 16
        assert set(row['question_positions']) == {str(p) for p in range(max(dev_lengths))}
        for position, histogram in row['question_positions'].items():
            n = sum(length > int(position) for length in dev_lengths)
            audit_histogram(histogram, n)
            accumulated = [a + b for a, b in zip(accumulated, histogram['slot_counts'])]
        assert accumulated == row['all']['read_argmax']['slot_counts']


def summary(report):
    rows = report['probes']
    def score(row):
        return f"{row['correct']}/{row['n']} ({row['accuracy']:.4%})"
    def bounds(candidates):
        low, high = min(candidates, key=lambda r: r['accuracy']), max(candidates, key=lambda r: r['accuracy'])
        return f"{score(low)} [{low.get('arm', 'Base')}] – {score(high)} [{high.get('arm', 'Base')}]"
    lines = ['# E-0012 verified descriptive readout', '',
        'All six L0 controls passed. The 120 probe rows, ten L3 arms and fourteen L4 entries passed independent artifact checks.', '',
        '| Feature | L1 delimiter→tail | L2 random value→tail | L2 trained value→tail range |',
        '|---|---:|---:|---:|']
    for tap in TAPS:
        frozen = next(r for r in rows if r['level'] == 'L1' and r['tap'] == tap and r['target'] == 'tail')
        random = next(r for r in rows if r['level'] == 'L2' and r['arm'] == f'random.{tap}' and r['target'] == 'tail')
        trained = [r for r in rows if r['level'] == 'L2' and r['tap'] == tap and r['arm'].startswith('V2-') and r['target'] == 'tail']
        lines.append(f'| {tap} | {score(frozen)} | {score(random)} | {bounds(trained)} |')
    lines += ['', '| Feature | L1 head-position address, ranks128–1024 | L1 last-position address, ranks128–1024 |', '|---|---:|---:|']
    for tap in TAPS:
        group = [r for r in rows if r['level'] == 'L1_address' and r['tap'] == tap]
        lines.append(f"| {tap} | {bounds([r for r in group if r['position'] == 'question_head_end'])} | {bounds([r for r in group if r['position'] == 'question_last'])} |")
    lines += ['', '| L3 arm | Clean→clean | Read→read | Clean→read |', '|---|---:|---:|---:|']
    for arm, row in report['l3'].items():
        lines.append('| ' + arm + ' | ' + ' | '.join(score(row[k]) for k in ('clean_to_clean', 'read_to_read', 'clean_to_read')) + ' |')
    lines += ['', '| L4 arm | Key cosine mean / n | Last target / n | Last same relation / n | Last slot mode: count / n | Last fusion/residual mean / n |', '|---|---:|---:|---:|---:|---:|']
    for arm, row in report['l4'].items():
        hist, ratio, cosine = row['last']['read_argmax'], row['last']['fusion_to_residual_norm'], row['key_pair_cosine']
        slot = max(range(16), key=lambda i: hist['slot_counts'][i])
        lines.append(f"| {arm} | {cosine['mean']:.6f} / {cosine['n']} | {hist['target_count']}/{hist['n']} | {hist['same_relation_count']}/{hist['n']} | {slot}: {hist['slot_counts'][slot]}/{hist['n']} | {ratio['mean']:.6g} / {ratio['n']} |")
    lines += ['', 'These are observations under the specified probes and splits. L1/L2 are softmax probes with train-internal model selection; L3 uses independent ridge heads and a smaller fitting set, so its numbers are separate diagnostics.',
        'The tables do not identify a unique cause, establish information absence, change H-0003, or select a subsequent experiment. All raw counts, per-position histograms, quantiles and selected hyperparameters remain in the full report.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='/work/xg24i002/x10041/bypass-lora')
    parser.add_argument('--full-output', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project, full, out = Path(args.project), Path(args.full_output).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    success, start = False, time.monotonic()
    metrics = {}
    try:
        report = json.loads((full / 'report.json').read_text())
        manifest = json.loads((full / 'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['validator']['passed'] and manifest['spec_digest'] == SPEC
        assert report['complete'] and report['l0_pass'] and report['scope'] == 'full diagnostic, seed 0'
        assert report['split_counts'] == {'train': 16384, 'calibration': 2048, 'dev': 1000}
        assert report['fit_episodes'] == 14336 and report['internal_validation_episodes'] == 2048
        snapshot = project / 'snapshots' / FULL_COMMIT
        for filename, expected in report['source_sha256'].items():
            assert digest(snapshot / 'src' / filename) == expected, filename
        for filename, expected in report['input_sha256'].items():
            assert str(project) in filename and 'confirmation' not in filename
            assert digest(filename) == expected, filename
        assert {r['tap'] for r in report['hook_verification']} == set(TAPS)
        for row in report['hook_verification']:
            assert all(v == 0. for v in row['max_abs'].values())
            assert math.isfinite(row['read_z_max_abs'])
        arms = json.loads((project / 'research/artifacts/T-0011/training-run-map.json').read_text())['training']
        counters, dev_lengths = audit_data(project, report)
        audit_probes(report, arms, counters)
        audit_l3_l4(report, full, arms, dev_lengths)
        l0 = [r for r in report['probes'] if r['level'] == 'L0']
        metrics = {'probe_rows': 120, 'l0_controls': 6, 'l0_min_accuracy': min(r['accuracy'] for r in l0),
                   'l3_arms': 10, 'l4_arms': 14, 'dev_question_tokens': sum(dev_lengths),
                   'source_files_hashed': len(report['source_sha256']), 'input_files_hashed': len(report['input_sha256']),
                   'full_report_sha256': digest(full / 'report.json')}
        (out / 'verified-summary.md').write_text(summary(report))
        write_json(out / 'checks.json', {'passed': True, **metrics,
            'checked': ['exact 120-row protocol matrix and six L0 gates', 'all calibration counts and accuracy arithmetic',
                        'constant baselines independently recomputed from released train/calibration',
                        'train-internal selection scores, epoch/L2 grid and selected optimum',
                        'ten original L3 manifests and exact counts', 'fourteen L4 distribution/histogram totals and quantile ranges',
                        'data identities/splits and source/input hashes against executed snapshot']})
        success = True
    finally:
        metrics['elapsed_s'] = time.monotonic() - start
        artifacts = [{'path': str(out / name), 'role': role} for name, role in
                     (('checks.json', 'independent_artifact_validation'), ('verified-summary.md', 'descriptive_measurements'))] if success else []
        write_json(out / 'manifest.json', {'status': 'complete' if success else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics, 'artifacts': artifacts,
            'validator': {'name': 'E0012_independent_artifact_audit', 'passed': success}})
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
