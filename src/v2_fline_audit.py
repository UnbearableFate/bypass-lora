"""Independent saved-output audit of E-0008; execute in PBS."""
import argparse
import json
import math
import os
from pathlib import Path
from v2_schedule import load_data, make_schedule


def rows(path):
    return [json.loads(line) for line in path.open()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.run / 'metrics.json').read_text())['metrics']
    train = load_data(args.data / 'train.jsonl', 'train')
    schedule = make_schedule(train, 4000000)
    saved = json.loads((args.run / 'schedule.json').read_text())
    assert schedule == saved
    learning = rows(args.run / 'learning.jsonl')
    assert len(learning) == schedule['steps'] == 6000
    total = 0
    for step, (record, batch) in enumerate(zip(learning, schedule['batches']), 1):
        total += sum(len(train[i]['support_input_ids']) + len(train[i]['question_input_ids']) for i in batch)
        assert record['step'] == step and record['tokens'] == total
        assert all(math.isfinite(record[key]) for key in ('loss', 'lr', 'grad_norm'))
    assert total == 4000470 and learning[-2]['tokens'] < 4000000
    assert learning[-1]['lr'] == 0
    report = {'schedule_match': True, 'all_training_steps_finite': True, 'splits': {}}
    for split, size in [('calibration', 2048), ('dev', 1000)]:
        data = load_data(args.data / (split + '.jsonl'), split)
        original = rows(args.run / (split + '.jsonl'))
        repeated = rows(args.run / (split + '-reloaded.jsonl'))
        assert original == repeated and len(original) == size
        assert [r['episode_id'] for r in original] == [r['episode_id'] for r in data]
        assert [r['target'] for r in original] == [r['query']['answer_token_id'] for r in data]
        for name, subset in [('all', original), ('nonself', [r for r in original if not r['self_edge']])]:
            n = len(subset)
            values = {'examples': n, 'em': sum(r['prediction'] == r['target'] for r in subset) / n,
                      'nll': sum(r['nll'] for r in subset) / n,
                      'subject_copy': sum(r['prediction'] == r['subject'] for r in subset) / n,
                      'rank_mean': sum(r['rank'] for r in subset) / n}
            for key, val in values.items():
                assert math.isclose(val, metrics[f'{split}.{name}.{key}'], rel_tol=1e-12, abs_tol=1e-12)
        report['splits'][split] = {'n': size, 'reload_exact': True,
            'self_edges': sum(r['self_edge'] for r in original),
            'errors': [r for r in original if r['prediction'] != r['target']],
            'max_nll_record': max(original, key=lambda r: r['nll'])}
    curves = rows(args.run / 'curve.jsonl')
    assert [r['step'] for r in curves] == [1500, 3000, 4500]
    for curve in curves:
        records = rows(args.run / f"dev-step-{curve['step']}.jsonl")
        assert len(records) == 1000
        assert sum(r['prediction'] == r['target'] for r in records) / 1000 == curve['metrics']['all.em']
        assert math.isclose(sum(r['nll'] for r in records) / 1000, curve['metrics']['all.nll'])
    report['curves_verified'] = True
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps({'status': 'complete',
        'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': {'audit.pass': True},
        'artifacts': [{'path': str((args.output / 'audit.json').resolve()), 'role': 'saved-output audit'}],
        'validator': {'name': 'v2_fline_audit', 'passed': True}}, indent=2) + '\n')


if __name__ == '__main__':
    main()
