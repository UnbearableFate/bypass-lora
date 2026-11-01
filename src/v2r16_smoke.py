"""Registered E-0016 smoke, old/new identity and D3 cost gate; PBS only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from v2r16_environment import verify_gpu_profile

PROJECT = Path('/work/xg24i002/x10041/bypass-lora')
REFERENCE = PROJECT/'snapshots/e1660c954ff1f1efb2738a1708d3eb095f060541'
DATA = PROJECT/'runs/ATT-0039-1/data'


def execute(argv, log):
    with Path(log).open('w') as handle:
        process = subprocess.run([sys.executable, *map(str, argv)], stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'child exit {process.returncode}: {log}')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    metrics, passed, failure = {}, False, None
    try:
        hardware = verify_gpu_profile(out)
        metrics['hardware.profile_verified'] = hardware['passed']
        with tempfile.TemporaryDirectory(dir=out, prefix='tests-') as scratch:
            execute(['-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                     'src/test_v2r16_objective.py', 'src/test_v2r16_schedule.py',
                     'src/test_v2r16_memory.py', 'src/test_v2r16_analysis.py',
                     f'--basetemp={scratch}', f'--junitxml={out/"tests.xml"}'], out/'tests.log')
        metrics['tests.passed'] = True
        with tempfile.TemporaryDirectory(dir=out, prefix='identity-') as scratch:
            for label, source in (('reference', REFERENCE/'src'), ('candidate', Path.cwd()/'src')):
                execute(['src/v2r_default_regression.py', 'dump', '--src', source, '--data-dir', DATA,
                         '--output', Path(scratch)/f'{label}.pt'], out/f'identity-{label}.log')
            execute(['src/v2r_default_regression.py', 'compare', Path(scratch)/'reference.pt',
                     Path(scratch)/'candidate.pt', '--output', out/'identity.json'], out/'identity-compare.log')
        metrics['default.identical'] = read(out/'identity.json')['identical']
        execute(['src/v2_train.py', '--data-dir', DATA, '--output', out/'baseline', '--lr', '.003',
                 '--tap', 'direct_hidden', '--backend', 'soft_slot', '--fusion', 'F_out',
                 '--step-budget', '6000', '--smoke-steps', '20', '--smoke-eval-limit', '32',
                 '--eval-splits', 'dev'], out/'baseline.log')
        baseline = read(out/'baseline/direct_hidden.soft_slot.F_out/metrics.json')['train']['stable_step_seconds']
        metrics['baseline.stable_step_seconds'] = baseline
        for arm in ('D1', 'D3', 'D2a', 'D2b'):
            command = ['src/v2r16_train.py', '--arm', arm, '--data-dir', DATA, '--output', out/arm,
                       '--smoke-steps', '20', '--smoke-eval-limit', '32']
            if arm.startswith('D2'):
                command += ['--checkpoint', out/'D1/adapter']
            execute(command, out/f'{arm}.log')
            record = read(out/arm/'metrics.json')
            manifest = read(out/arm/'manifest.json')
            if manifest['status'] != 'complete' or not manifest['validator']['passed']:
                raise RuntimeError(f'{arm}: incomplete smoke application')
            metrics[f'{arm}.stable_step_seconds'] = record['train']['stable_step_seconds']
            metrics[f'{arm}.reload_identical'] = record['checkpoint_reload']['full_records_identical']
            metrics[f'{arm}.evaluation_seconds'] = record['evaluation_seconds']
            metrics[f'{arm}.curve_seconds'] = record['train']['learning_curve_evaluation_seconds']
            metrics[f'{arm}.wall_seconds'] = record['wall_seconds']
            if arm == 'D3':
                ratio = record['train']['stable_step_seconds']/baseline
                metrics['D3.step_time_ratio'] = ratio
                metrics['D3.cost_gate_passed'] = ratio <= 3
                if ratio > 3:
                    raise RuntimeError(f'E-0016 D3 cost stop: {ratio:.4f} times V2-03 (>3)')
        passed = True
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        metrics['wall_seconds'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({
            'status': 'complete' if passed else 'partial', 'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'metrics': metrics, 'failure': failure,
            'artifacts': [{'path': str(p.resolve()), 'role': p.name} for p in out.rglob('*')
                          if p.is_file() and p.name != 'manifest.json'],
            'validator': {'name': 'v2r16-smoke-default-regression-and-cost-gate', 'passed': passed}}, indent=2)+'\n')


if __name__ == '__main__':
    main()
