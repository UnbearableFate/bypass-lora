"""Read-only inventory of the legacy data available for E-0005 overlap checks.

Run only inside a registered PBS allocation. No legacy Python is imported and
NumPy object deserialization is prohibited. This probe does not claim overlap=0.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {'legacy_commit': 'a255e814740ff85ab3b4ec4836c88f0a8f802b2d',
              'files': {}, 'overlap_verdict': 'not_checked', 'archives': {}}
    for path in sorted(args.legacy_dir.iterdir()):
        if path.is_file():
            report['files'][path.name] = {'sha256': file_sha256(path),
                                        'bytes': path.stat().st_size}
    registry = args.legacy_dir / 'experiment_registry.csv'
    with registry.open(newline='') as handle:
        reader = csv.DictReader(handle)
        report['registry_columns'] = reader.fieldnames
        versions, tasks, sample_paths = Counter(), Counter(), set()
        count = 0
        for row in reader:
            count += 1
            versions[row.get('data_version', '')] += 1
            tasks[row.get('task', '')] += 1
            if row.get('samples'):
                sample_paths.add(row['samples'])
        report['registry_rows'] = count
        report['data_versions'] = dict(versions)
        report['tasks'] = dict(tasks)
        report['distinct_referenced_sample_paths'] = len(sample_paths)
        report['available_referenced_sample_paths'] = sum(Path(p).is_file() for p in sample_paths)
    for path in sorted(args.legacy_dir.glob('*.npz')):
        arrays = {}
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                try:
                    value = archive[key]
                except ValueError as exc:
                    arrays[key] = {'unreadable_without_pickle': str(exc)}
                    continue
                preview = value.reshape(-1)[:8].tolist()
                arrays[key] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                               'first_values': [str(x)[:300] for x in preview]}
        report['archives'][path.name] = arrays
    from transformers import AutoTokenizer
    revision = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', revision=revision,
                                              local_files_only=True)
    report['tokenizer_revision'] = revision
    report['delimiter_encodings'] = {s: tokenizer.encode(s, add_special_tokens=False)
                                   for s in [';', ' ;', '\n', ' |', ' [END]']}
    report['elapsed_s'] = time.monotonic() - start
    output = args.output / 'legacy_inventory.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    manifest = {'status': 'complete', 'spec_digest': os.environ['SR_SPEC_DIGEST'],
                'metrics': {'legacy.registry_rows': count,
                            'legacy.sample_archives': len(report['archives']),
                            'elapsed_s': report['elapsed_s']},
                'artifacts': [{'path': str(output.resolve()), 'role': 'legacy_data_inventory'}],
                'validator': {'name': 'v2_legacy_probe', 'passed': True}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
