"""Aggregate E-0003 eight-arm smoke, including the analysis regression suite."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


ARMS = ('F_out', 'F_pre', 'C_gate', 'C_up', 'C_dual', 'linear', 'lora', 'lora28')
EXPECTED_PARAMS = {
    'F_out': 262144,
    'F_pre': 262144,
    'C_gate': 524288,
    'C_up': 524288,
    'C_dual': 917504,
    'linear': 917504,
    'lora': 901120,
    'lora28': 1261568,
}
EXPECTED_PLAN_STEPS = 13077
EXPECTED_PLAN_TOKENS = 2000040
SMOKE_STEPS = 20
DEV_POINTS = (math.ceil(.25 * SMOKE_STEPS), 10, 15)


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _metric(metrics, *keys):
    """Return a metric from flat or nested metric dictionaries."""
    for key in keys:
        if key in metrics:
            return metrics[key]
        value = metrics
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            return value
    return None


def valid_e3_smoke(manifest, arm, steps=SMOKE_STEPS, intermediate_dir=None):
    """Validate the E3 contract; this run intentionally evaluates dev only."""
    metrics = manifest.get('metrics', {})
    if (manifest.get('status') != 'complete'
            or not manifest.get('validator', {}).get('passed')
            or metrics.get('train.diverged')
            or metrics.get('run.failed_lr')
            or metrics.get('train.steps') != steps
            or metrics.get('reload.identical') is not True
            or metrics.get('init.identical') is not True
            or metrics.get('plan.steps') != EXPECTED_PLAN_STEPS
            or metrics.get('plan.input_tokens') != EXPECTED_PLAN_TOKENS):
        return False

    expected = EXPECTED_PARAMS[arm]
    stored = _metric(metrics, 'params.stored')
    if not isinstance(stored, int) or stored < expected:
        return False
    for name in ('adapter_stored', 'trainable', 'with_grad'):
        if _metric(metrics, f'params.{name}') != expected:
            return False

    # E3's evaluation is dev-only. Calibration metrics indicate the wrong mode.
    if any(key.startswith('calibration.') for key in metrics):
        return False
    for task in ('piqa', 'boolq'):
        for name in ('nll', 'accuracy'):
            if not _finite_number(_metric(metrics, f'dev.{task}.{name}')):
                return False
    if not _finite_number(_metric(metrics, 'dev.nll')):
        return False

    for point in DEV_POINTS:
        if intermediate_dir is None:
            return False
        point_path = Path(intermediate_dir)/f'dev-step-{point}.json'
        if not point_path.is_file():
            return False
        try:
            point_record = json.loads(point_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if point_record.get('step') != point:
            return False
        point_metrics = point_record.get('metrics', {})
        for task in ('piqa', 'boolq'):
            for name in ('nll', 'accuracy'):
                value = _metric(point_metrics, f'dev.{task}.{name}', f'{task}.{name}')
                if not _finite_number(value):
                    return False
        if not _finite_number(_metric(point_metrics, 'dev.nll', 'nll')):
            return False
    return True


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    start = time.monotonic()
    metrics = {}
    artifacts = []
    passed = False
    try:
        test_files = ('test_core.py', 'test_model.py', 'test_v1_data.py',
                      'test_v1_lora.py', 'test_v1_run.py', 'test_v1_analysis.py')
        code = pytest.main(['-v', '-p', 'no:cacheprovider', '--basetemp=' + str(out/'tmp'),
                            '--junitxml=' + str(out/'checks.xml'),
                            *[str(src/name) for name in test_files]])
        metrics['pytest.exit_code'] = int(code)
        artifacts.append({'path': str(out/'checks.xml'), 'role': 'pytest_results'})
        if code:
            raise RuntimeError('V0/V1/E3 analysis regression tests failed')

        run_start = time.monotonic()
        order_digest = None
        for arm in ARMS:
            arm_out = out/arm
            command = [sys.executable, str(src/'v1_run.py'), '--arm', arm, '--e3',
                       '--tokens', '2000000', '--lr', '.0003', '--smoke-steps', str(SMOKE_STEPS),
                       '--eval-limit', '32', '--check-init', '--data-dir',
                       '/work/xg24i002/x10041/bypass-lora/data/v1', '--output', str(arm_out)]
            arm_start = time.monotonic()
            result = subprocess.run(command)
            arm_elapsed = time.monotonic() - arm_start
            manifest_path = arm_out/'manifest.json'
            if not manifest_path.is_file():
                raise RuntimeError(f'{arm} produced no manifest')
            manifest = json.loads(manifest_path.read_text())
            arm_metrics = manifest.get('metrics', {})
            excluded = json.loads((arm_out/'excluded-train.json').read_text())
            assert [(row['task'],row['id'],row['candidate_lengths']) for row in excluded] == [('piqa',5528,[537,537])]
            assert arm_metrics.get('plan.dev_steps') == [3270,6539,9808]
            if order_digest is None:
                order_digest = arm_metrics['plan.order_sha256']
            assert arm_metrics.get('plan.order_sha256') == order_digest
            metrics[f'{arm}.smoke.elapsed_s'] = arm_elapsed
            for key, value in arm_metrics.items():
                metrics[f'{arm}.{key}'] = value
            usage_path = arm_out/'usage.json'
            if usage_path.is_file():
                usage = json.loads(usage_path.read_text())
                for key, value in usage.items():
                    metrics[f'{arm}.usage.{key}'] = value
            artifacts.append({'path': str(manifest_path), 'role': f'{arm}_manifest'})
            if result.returncode or not valid_e3_smoke(manifest, arm, intermediate_dir=arm_out):
                raise RuntimeError(f'{arm} E3 smoke failed (exit {result.returncode})')
        metrics['arms.elapsed_s'] = time.monotonic() - run_start
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic() - start
        metrics['train.input_tokens'] = sum(
            metrics.get(f'{arm}.train.input_tokens', 0) for arm in ARMS)
        metrics['train.loss_tokens'] = sum(
            metrics.get(f'{arm}.train.loss_tokens', 0) for arm in ARMS)
        metrics['train.examples'] = sum(
            metrics.get(f'{arm}.train.examples', 0) for arm in ARMS)
        (out/'manifest.json').write_text(json.dumps({
            'status': 'complete' if passed else 'partial',
            'spec_digest': os.environ.get('SR_SPEC_DIGEST'),
            'metrics': metrics,
            'artifacts': artifacts,
            'validator': {'name': 'v1_e3_smoke', 'passed': passed},
        }, indent=2) + '\n')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
