"""PBS formal-run wrapper: guard hardware and preserve application failure facts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from v2r16_environment import verify_gpu_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    child_manifest = out/'arm/manifest.json'
    report, error = None, None
    try:
        hardware = verify_gpu_profile(out)
        command = [sys.executable, 'src/v2r16_train.py', '--arm', args.arm,
                   '--data-dir', '/work/xg24i002/x10041/bypass-lora/runs/ATT-0039-1/data',
                   '--output', str(out/'arm')]
        if args.checkpoint:
            command += ['--checkpoint', str(args.checkpoint)]
        completed = subprocess.run(command)
        if child_manifest.exists():
            report = json.loads(child_manifest.read_text())
            report['hardware_verification'] = hardware
        if completed.returncode or report is None or report['status'] != 'complete' or not report['validator']['passed']:
            raise RuntimeError(f'training child failed: exit {completed.returncode}')
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        if report is None:
            report = {'status': 'partial', 'spec_digest': os.environ['SR_SPEC_DIGEST'],
                      'metrics': {}, 'validator': {'name': 'v2r16-formal-launch', 'passed': False}}
        report['launch_error'] = error
        report.setdefault('artifacts', []).extend({'path': str(p), 'role': p.name} for p in out.glob('*.json')
                                                 if p.name != 'manifest.json')
        (out/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
