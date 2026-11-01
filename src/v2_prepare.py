"""PBS entry point for V2 data checks and deterministic materialization."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import time

import pytest
from transformers import AutoTokenizer
from v2_data import MODEL, REVISION, RELATIONS, SIZES, file_digest, generate_dataset


def legacy_disjointness(directory):
    """Inspect pinned source as text/AST, never import or execute legacy code."""
    hashes = {'render.py': 'eecc3ba0188e5815523dc15d7bd1046bb0a8e77cd05c948f8506e5c8ab1c9aab',
              'relations.py': '94f8df780515d158e64fdd415d6a684e45a7805bb6d52aaad0422d8302d1f3a5',
              'experiment_registry.csv': 'f2419a7a6992b9a62a536641023d868baff77c7a8362c3d7b960f26efe53a251'}
    for name, expected in hashes.items():
        if file_digest(directory / name) != expected:
            raise ValueError('legacy audit source digest mismatch: ' + name)
    tree = ast.parse((directory / 'render.py').read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'RELATION_PHRASES' for t in node.targets)]
    if len(assignments) != 1:
        raise ValueError('legacy relation list not found')
    old_relations = ast.literal_eval(assignments[0].value)
    if set(RELATIONS) & set(old_relations) or any(r.startswith('has relation ') for r in RELATIONS):
        raise ValueError('new and old relation labels overlap')
    with (directory / 'experiment_registry.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or {r['data_version'] for r in rows} != {'relations-v1'}:
        raise ValueError('legacy registry includes an unaudited data version')
    return {'legacy_commit': 'a255e814740ff85ab3b4ec4836c88f0a8f802b2d', 'source_sha256': hashes,
            'registry_rows': len(rows), 'legacy_relation_phrases': old_relations,
            'legacy_relation_fallback': 'has relation {nonnegative integer} to',
            'new_relation_phrases': list(RELATIONS), 'overlap_count': 0,
            'method': 'construction proof: every labeled fact uses a disjoint relation label',
            'scope': 'named relation triples plus question; does not exclude isomorphic abstract graphs',
            'source_lines': {'render.py': '19-23,77-88,105-128,163-176', 'relations.py': '319-347'},
            'coordinator_authorization': 'MSG-0093'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full', action='store_true')
    p.add_argument('--legacy-dir', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    proof = legacy_disjointness(args.legacy_dir)
    (args.output / 'legacy_disjointness.json').write_text(json.dumps(proof, indent=2) + '\n')
    code = pytest.main(['-v', '-p', 'no:cacheprovider', '--basetemp=' + str(args.output / 'tmp'),
                       '--junitxml=' + str(args.output / 'checks.xml'),
                       str(Path(__file__).with_name('test_v2_data.py'))])
    if code:
        raise RuntimeError('V2 data invariant tests failed')
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tick = time.monotonic()
    sizes = SIZES if args.full else {'train': 32, 'calibration': 16, 'dev': 16, 'confirmation': 16}
    metadata = generate_dataset(tok, args.output / 'data', sizes=sizes)
    if args.full and metadata['train_budget']['available_tokens_3_epochs'] < 4000000:
        raise ValueError('train dataset cannot cover per-arm budget within three epochs')
    metrics = {'pytest.exit_code': int(code), 'generation_s': time.monotonic() - tick,
               'elapsed_s': time.monotonic() - start,
               'unique_identities': metadata['unique_identities'],
               'unique_semantic_contents': metadata['unique_semantic_contents']}
    files = ['checks.xml', 'legacy_disjointness.json', 'data/metadata.json', 'data/name_pools.json', 'data/train.jsonl',
             'data/calibration.jsonl', 'data/dev.jsonl', 'data/confirmation.registry.jsonl']
    manifest = {'status': 'complete', 'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics,
                'artifacts': [{'path': str((args.output / f).resolve()), 'role': f} for f in files],
                'validator': {'name': 'v2_prepare', 'passed': True}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'metrics': metrics, 'metadata_sha256': file_digest(args.output / 'data/metadata.json')}))


if __name__ == '__main__':
    main()
