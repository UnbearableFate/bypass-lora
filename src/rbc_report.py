"""Verify RBc run artifacts and call the reviewed extension/lock rules in PBS.

This module reads calibration artifacts only. It changes no training or scoring
path. The final lock is produced by the existing rb_train lock CLI; the common
sample-order hash is added as provenance without altering its choices.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from rb_train import GRIDS, PARAMETERS, REPRESENTATIVES, extension_lr

WORKSPACE = Path('/work/xg24i002/x10041/bypass-lora')
REVIEWED = '183ad685785b8249d8a1c679a09e8dd5280d8cd5'
SPLIT = '34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc'
ORDER = '053284831520676982857bf9677a9fa3e338b0bbad4d2ce2af1c3f88d20457d3'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
CORE = ('bypass_core.py', 'bypass_model.py', 'rb_train.py', 'rb_data.py',
        'rb_history.py', 'v1_run.py', 'v1_data.py', 'v1_lora.py', 'v5_train.py',
        'calibrate.py')


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def verify(row):
    folder = Path(row['output']).resolve()
    assert folder == WORKSPACE/'runs'/row['attempt']
    value = read(folder/'manifest.json')
    metrics = value['metrics']
    assert value['status'] == 'complete' and value['validator']['passed'] is True
    assert value['contract'] == 'E-0020@1' and value['stage'] == 'calibration'
    assert value['phase'] == 'V0' and value['seed'] == 0
    assert value['arm'] == row['arm'] == REPRESENTATIVES[row['family']]
    assert value['family'] == row['family'] and value['spec_digest'] == row['spec_digest']
    assert metrics['learning_rate'] == row['learning_rate']
    assert metrics['source.commit'] == row['code_commit']
    snapshot = WORKSPACE/'snapshots'/row['code_commit']
    reviewed = WORKSPACE/'snapshots'/REVIEWED
    for name in CORE:
        assert ref(snapshot/'src'/name)['sha256'] == ref(reviewed/'src'/name)['sha256'], name
    assert ref(snapshot/'uv.lock')['sha256'] == ref(reviewed/'uv.lock')['sha256']
    assert metrics['environment.lock_sha256'] == ref(snapshot/'uv.lock')['sha256']
    assert metrics['torch'] == '2.10.0+cu128' and metrics['cuda'] == '12.8'
    assert metrics['model_revision'] == metrics['tokenizer_revision'] == REVISION
    assert metrics['split_manifest.sha256'] == SPLIT
    assert ref(snapshot/'research/artifacts/T-0022/rb-confirmation.json')['sha256'] == SPLIT
    assert metrics['plan.order_sha256'] == ORDER
    assert metrics['plan.steps'] == 1096 and metrics['plan.examples'] == 2192
    assert metrics['plan.input_tokens'] == 200229
    assert not any(key.startswith('dev.') for key in metrics)
    assert not list(folder.glob('dev-step-*.json'))
    assert not (folder/'adapter').exists() and not (folder/'checkpoint.json').exists()
    assert 'reload.identical' not in metrics and 'save.seconds' not in metrics
    config = read(folder/'config.json')
    assert config['stage'] == 'calibration' and config['command'] == 'run'
    assert config['arm'] == row['arm'] and config['seed'] == 0
    assert config['lr'] == row['learning_rate']
    assert config['eval_limit'] is None and config['smoke_steps'] is None
    assert not config['three_epochs'] and config['lr_lock'] is None
    assert Path(config['output']).resolve() == folder
    assert config['split_manifest'] == 'research/artifacts/T-0022/rb-confirmation.json'
    allocation = read(folder/'allocation.json')
    assert allocation['ok'] is True and allocation['job_state'] == 'R'
    context = allocation['context']
    assert context['allocation_confirmed'] is context['runtime_execution_allowed'] is True
    assert context['pbs_job_id'] == row['job_id'] and context['system'] == 'Miyabi-G'
    # D-0018/D-0019: retain actual profile as metadata; never gate or compare it.
    profile = read(folder/'mig-profile.json')
    assert row['scheduler_exit_status'] == 0 and row['used_MIG_seconds'] > 0
    assert row['used_node_h'] == row['used_MIG_seconds']/14400
    failed = metrics['failed']
    assert type(failed) is bool
    if not failed:
        nll = metrics['calibration.nll']
        assert type(nll) in (int, float) and math.isfinite(nll) and nll > 0
        assert metrics['train.steps'] == 1096 and metrics['train.examples'] == 2192
        assert metrics['train.input_tokens'] == 200229
        assert metrics['params.stored'] == metrics['params.trainable'] == metrics['params.with_grad'] == PARAMETERS[row['arm']]
        assert metrics['calibration.export_reload_skipped'] is True
        assert metrics['intermediate.seconds'] == 0
        steps = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(steps) == 1096 and [step['step'] for step in steps] == list(range(1, 1097))
        assert sum(step['input_tokens'] for step in steps) == 200229
        assert sum(step['input_tokens'] for step in steps[:-1]) < 200000
        assert all(math.isfinite(step[key]) for step in steps
                   for key in ('example_mean_nll', 'preclip_norm', 'lr', 'seconds'))
        predictions = read(folder/'predictions.json')
        assert len(predictions) == 4096
        assert all(pred['split'] == 'calibration' for pred in predictions)
        task_nll = []
        for task in ('piqa', 'boolq'):
            rows = [pred for pred in predictions if pred['task'] == task]
            assert len(rows) == len({pred['id'] for pred in rows}) == 2048
            total, count = 0., 0
            for pred in rows:
                assert math.isfinite(pred['gold_nll_sum'])
                assert all(math.isfinite(score) for score in pred['scores'])
                total += pred['gold_nll_sum']
                count += pred['gold_loss_tokens']
            assert metrics[f'calibration.{task}.examples'] == 2048
            assert metrics[f'calibration.{task}.loss_tokens'] == count
            assert metrics[f'calibration.{task}.nll'] == total/count
            task_nll.append(total/count)
        assert metrics['calibration.nll'] == sum(task_nll)/2
        accounting = read(folder/'accounting.json')
        assert accounting == metrics['runtime.accounting']
        if row['arm'] == 'RB1-07':
            assert accounting['layers'] == {}
        else:
            assert set(accounting['layers']) == {'6', '13', '20', '27'}
            for counts in accounting['layers'].values():
                assert counts['attention'] == counts['ffn'] == 2192
                assert counts['valid_tokens'] == 200229
                assert counts['writes'] == counts['reads'] == (0 if row['arm'] == 'RB1-01' else 200229)
    else:
        assert metrics['calibration.nll'] is None and metrics.get('failure_reason')
        nll = None
    return {**row, 'manifest': ref(folder/'manifest.json'), 'allocation': ref(folder/'allocation.json'),
            'predictions': ref(folder/'predictions.json') if not failed else None,
            'profile_metadata': profile, 'finite': not failed, 'nll': nll,
            'steps': metrics.get('train.steps'), 'tokens': metrics.get('train.input_tokens'),
            'steady_step_seconds': metrics.get('train.steady_step_seconds'),
            'train_seconds': metrics.get('train.seconds'), 'eval_seconds': metrics.get('eval.seconds')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--mode', choices=('initial', 'final'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out/'manifest.json').exists() or (out/'report.json').exists():
        raise ValueError('refuse to replace a prior report')
    out.mkdir(parents=True, exist_ok=True)
    passed, metrics, artifacts = False, {}, []
    start = time.monotonic()
    try:
        jobs = read(args.jobs)
        assert jobs['experiment'] == 'E-0020@1' and jobs['task'] == 'T-0024'
        rows = [verify(row) for row in jobs['runs']]
        keys = [(row['family'], row['learning_rate']) for row in rows]
        assert len(keys) == len(set(keys))
        families = {}
        for family, grid in GRIDS.items():
            candidates = {row['learning_rate']: row['nll'] for row in rows if row['family'] == family}
            initial = {lr: candidates[lr] for lr in grid}
            extension = extension_lr(family, initial)
            expected = set(grid) | ({extension} if args.mode == 'final' and extension is not None else set())
            assert set(candidates) == expected
            finite = {lr: nll for lr, nll in initial.items() if nll is not None}
            best = min(finite, key=lambda lr: (finite[lr], lr))
            families[family] = {'raw_initial_minimum_lr': best,
                                'raw_initial_minimum_nll': finite[best], 'required_extension_lr': extension}
        if args.mode == 'final':
            raw_path = out/'lr-lock-reviewed.json'
            subprocess.run([sys.executable, 'src/rb_train.py', 'lock', '--manifests',
                            *(row['manifest']['path'] for row in rows), '--output', str(raw_path)], check=True)
            lock = read(raw_path)
            assert lock['experiment'] == 'E-0020@1' and lock['split_manifest_sha256'] == SPLIT
            assert lock['inputs'] == [row['manifest'] for row in rows]
            lock['plan_order_sha256'] = ORDER
            (out/'lr-lock.json').write_text(json.dumps(lock, indent=2, allow_nan=False)+'\n')
            artifacts.extend([ref(raw_path), ref(out/'lr-lock.json')])
            for family in GRIDS:
                finite = {row['learning_rate']: row['nll'] for row in rows
                          if row['family'] == family and row['finite']}
                minimum = min(finite.values())
                families[family].update(locked_lr=lock['locked_lrs'][family],
                    final_minimum_nll=minimum, boundary_minimum=lock['boundary_minimum'][family],
                    relative_excess={str(lr): (nll-minimum)/minimum for lr, nll in sorted(finite.items())})
        report = {'experiment': 'E-0020@1', 'mode': args.mode,
                  'split_manifest_sha256': SPLIT, 'plan_order_sha256': ORDER,
                  'jobs': ref(args.jobs), 'families': families, 'candidates': rows,
                  'candidate_MIG_seconds': sum(row['used_MIG_seconds'] for row in rows),
                  'candidate_node_h': sum(row['used_MIG_seconds'] for row in rows)/14400}
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        lines = ['# E-0020 RBc candidate table', '',
                 '| Family | LR | NLL | Finite | Steps | Tokens | Stable step s | node·h | Attempt |',
                 '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |']
        for row in sorted(rows, key=lambda r: (r['family'], r['learning_rate'])):
            lines.append(f"| {row['family']} | {row['learning_rate']:g} | {row['nll']} | {row['finite']} | {row['steps']} | {row['tokens']} | {row['steady_step_seconds']} | {row['used_node_h']:.9f} | {row['attempt']} |")
        lines += ['', 'Only full calibration is scored. Profile is metadata only. Stable step times are descriptive run metrics, not a controlled timing comparison.', '']
        (out/'candidate-table.md').write_text('\n'.join(lines))
        artifacts.extend([ref(out/'report.json'), ref(out/'candidate-table.md')])
        metrics.update(candidates=len(rows), finite_candidates=sum(row['finite'] for row in rows),
                       families=families, candidate_node_h=report['candidate_node_h'],
                       split_manifest_sha256=SPLIT, plan_order_sha256=ORDER)
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({'status': 'complete' if passed else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'contract': 'E-0020@1',
            'source_commit': os.environ['SR_CODE_COMMIT'], 'metrics': metrics,
            'artifacts': artifacts, 'validator': {'name': 'rbc_artifact_provenance_and_reviewed_rules',
                                               'passed': passed}}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
