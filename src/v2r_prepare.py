"""PBS entry point: E-0014 variant data invariant tests and deterministic generation."""
import argparse
import json
import os
from pathlib import Path
import time

import pytest
from transformers import AutoTokenizer

from v2r_data import MODEL, REVISION, VARIANTS, file_digest, generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--v2-data-dir', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    code = pytest.main(['-v', '-p', 'no:cacheprovider', '--basetemp=' + str(args.output / 'tmp'),
                        '--junitxml=' + str(args.output / 'checks.xml'),
                        str(Path(__file__).with_name('test_v2r_data.py'))])
    if code:
        raise RuntimeError('E-0014 data invariant tests failed')
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    sizes = None if args.full else {'train': 64, 'calibration': 16, 'dev': 16, 'confirmation': 16}
    metrics, artifacts = {'pytest.exit_code': int(code), 'full': args.full}, ['checks.xml']
    for n_facts in VARIANTS:
        tick = time.monotonic()
        directory = args.output / f'e{n_facts}'
        metadata = generate_dataset(tok, directory, n_facts, v2_data_dir=args.v2_data_dir, sizes=sizes)
        metrics[f'e{n_facts}'] = {
            'generation_s': time.monotonic() - tick,
            'metadata_sha256': file_digest(directory / 'metadata.json'),
            'unique_identities': metadata['unique_identities'],
            'prior_dev_em': metadata['prior']['dev_em'],
            **{f'{split}.{key}': metadata['splits'][split][key] for split in ('train', 'calibration', 'dev')
               for key in ('sha256', 'unique_contents', 'repeated_content_rows', 'rows_redrawn', 'input_tokens')}}
        artifacts += [f'e{n_facts}/{name}' for name in ('metadata.json', 'name_pools.json', 'train.jsonl',
                                                       'calibration.jsonl', 'dev.jsonl', 'confirmation.registry.jsonl')]
    metrics['elapsed_s'] = time.monotonic() - start
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    manifest = {'status': 'complete', 'spec_digest': os.environ.get('SR_SPEC_DIGEST'), 'metrics': metrics,
                'artifacts': [{'path': str((args.output / f).resolve()), 'role': f} for f in artifacts + ['metrics.json']],
                'validator': {'name': 'v2r_prepare', 'passed': True}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
