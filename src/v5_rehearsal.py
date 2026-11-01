"""T-0014 PBS-only tests, dev-artifact rehearsal, and Base WikiText timing.

This entry point has no validation input or call to the confirmation command.
The three seed files are copies of each V1 seed-0 prediction artifact, solely to
exercise the pipeline. They are neither V5 results nor independent replicates.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as xml

from v5_confirm import run_wikitext_base, write_json

PROJECT = Path('/work/xg24i002/x10041/bypass-lora')
PROXIES = {'Base': 'ATT-0024-1', 'V5-C': 'ATT-0028-1',
           'V5-F': 'ATT-0025-1', 'V5-L': 'ATT-0032-1'}


def rehearse(output, *, sealed_manifest, skip_wikitext=False):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metrics, passed = {'train.input_tokens': 0}, False
    try:
        with tempfile.TemporaryDirectory(prefix='synthetic-', dir=output) as test_tmp, (output / 'pytest.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                'src/test_v5_confirm.py', 'src/test_v5_analysis.py', 'src/test_v1_data.py',
                f'--basetemp={test_tmp}', f'--junitxml={output / "pytest.xml"}'], stdout=log, stderr=subprocess.STDOUT)
        metrics['pytest.exit_code'] = result.returncode
        if (output / 'pytest.xml').exists():
            suite = xml.parse(output / 'pytest.xml').getroot()
            suites = list(suite) if suite.tag == 'testsuites' else [suite]
            for key in ('tests', 'failures', 'errors', 'skipped'):
                metrics[f'pytest.{key}'] = sum(int(s.attrib.get(key, 0)) for s in suites)
        if result.returncode:
            raise RuntimeError(f'V5 regression failed: {output / "pytest.log"}')
        input_dir = output / 'dev-inputs'
        input_dir.mkdir()
        entries = []
        for arm, attempt in PROXIES.items():
            source = PROJECT / 'runs' / attempt / 'predictions.json'
            for seed in ([None] if arm == 'Base' else [1, 2, 3]):
                filename = f'{arm}-s{seed}.json'
                destination = input_dir / filename
                shutil.copyfile(source, destination)
                entries.append({'arm': arm, 'seed': seed, 'predictions': str(destination),
                                'source_attempt': attempt, 'source_seed': 0})
        manifest = {'entries': entries, 'procedural_only': True,
                    'source_attempts': PROXIES, 'source_seed': 0,
                    'proxy_note': 'V1 C_up r32, F_out r32, LoRA28 r2 seed 0 copied to seeds 1/2/3; this is only a software rehearsal.'}
        if not skip_wikitext:
            wiki = run_wikitext_base(sealed_manifest, output / 'base-wikitext')
            metrics.update(wiki)
            manifest['wikitext'] = {'Base': {'nll': wiki['wikitext.nll']}}
        write_json(output / 'analysis-input.json', manifest)
        tick = time.monotonic()
        from v5_analysis import analyze, write_outputs
        analysis = analyze(manifest, 'dev')
        analysis['source_manifest'] = manifest
        analysis['procedural_only'] = True
        write_outputs(analysis, output / 'analysis')
        metrics['analysis.seconds'] = time.monotonic() - tick
        metrics['analysis.bootstrap_repetitions'] = analysis['bootstrap']['repetitions']
        metrics['analysis.piqa_rows'] = analysis['counts_by_task']['piqa']
        metrics['analysis.boolq_rows'] = analysis['counts_by_task']['boolq']
        passed = True
    except Exception as exc:
        metrics['error'] = repr(exc)
        raise
    finally:
        import torch
        import transformers
        metrics['elapsed_s'] = time.monotonic() - started
        write_json(output / 'manifest.json', {'status': 'complete' if passed else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics,
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'cuda': torch.version.cuda, 'transformers': transformers.__version__,
                            'hardware': torch.cuda.get_device_name() if torch.cuda.is_available() else 'unknown'},
            'artifacts': [{'path': str(p.resolve()), 'role': p.name} for p in output.rglob('*')
                          if p.is_file() and p.name != 'manifest.json'],
            'validator': {'name': 'v5-dev-rehearsal-not-scientific-evidence', 'passed': passed}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sealed-manifest', type=Path, required=True)
    parser.add_argument('--skip-wikitext', action='store_true')
    args = parser.parse_args()
    rehearse(args.output.resolve(), sealed_manifest=args.sealed_manifest, skip_wikitext=args.skip_wikitext)


if __name__ == '__main__':
    main()
