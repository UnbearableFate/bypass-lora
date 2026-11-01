"""PBS entry point for T2 invariant tests, generation, and shortcut gates."""
import argparse
import json
import os
from pathlib import Path
import time

import pytest
from transformers import AutoTokenizer

from v2_prepare import legacy_disjointness
from v3_data import MODEL, REVISION, SPLITS, SEALED, generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--v2-data-dir', type=Path, required=True)
    parser.add_argument('--legacy-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('V3', 'V4'), default='V3')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    proof = legacy_disjointness(args.legacy_dir)
    proof['T2_applicability'] = 'Same disjoint labels in every T2 fact; question/termination cannot make unequal fact sets equal.'
    (args.output / 'legacy_disjointness.json').write_text(json.dumps(proof, indent=2) + '\n')
    code = pytest.main(['-v', '-p', 'no:cacheprovider', '--basetemp=' + str(args.output / 'tmp'),
                       '--junitxml=' + str(args.output / 'checks.xml'),
                       str(Path(__file__).with_name('test_v3_data.py'))])
    if code:
        raise RuntimeError('T2 invariant tests failed')
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    sizes = None if args.full else dict.fromkeys(SPLITS, 60)
    tick = time.monotonic()
    metadata = generate_dataset(tok, args.output / 'data', v2_data_dir=args.v2_data_dir,
                                phase=args.phase, sizes=sizes)
    generation_s = time.monotonic() - tick
    gates_passed = all(s['gates_passed'] for s in metadata['shortcuts'].values())
    if args.full and metadata['train_budget']['available_prompt_tokens_3_epochs'] < 5500000:
        raise ValueError('train cannot cover 5.5M prompt tokens in three epochs')
    metrics = {'pytest.exit_code': int(code), 'generation_s': generation_s,
               'elapsed_s': time.monotonic() - start, 'unique_identities': metadata['unique_identities'],
               'unique_semantic_contents': metadata['unique_semantic_contents'],
               'v2.actual_hash_overlap': metadata['v2_overlap']['actual_hashes'],
               'v2.normalized_overlap': metadata['v2_overlap']['normalized_facts_query'],
               'legacy.overlap': proof['overlap_count'], 'shortcuts.gates_passed': gates_passed,
               'full': args.full}
    (args.output / 'metrics.json').write_text(json.dumps({'metrics': metrics, 'shortcuts': metadata['shortcuts']}, indent=2) + '\n')
    artifacts = ['checks.xml', 'legacy_disjointness.json', 'metrics.json', 'data/metadata.json', 'data/name_pools.json']
    artifacts += ['data/' + s + ('.registry.jsonl' if s in SEALED else '.jsonl') for s in SPLITS]
    manifest = {'status': 'complete', 'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics,
                'artifacts': [{'path': str((args.output / f).resolve()), 'role': f} for f in artifacts],
                'validator': {'name': 'v3_prepare', 'passed': True}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)
    # A failed shortcut gate is a complete diagnostic result, not a code crash.
    # Model entry points independently refuse such a dataset.


if __name__ == '__main__':
    main()
