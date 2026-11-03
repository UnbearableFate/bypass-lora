"""Pre-opening V1 data preparation that never materializes sealed A2 fields.

The fixed source files are hashed as bytes first. Rows whose IDs are in the
sealed A2 manifest become ID-only placeholders; PIQA JSON and labels are not
decoded for them, and BoolQ parquet columns are filtered before ``to_pylist``.
"""
import hashlib
import json
from pathlib import Path

from rb_data import apply_confirmation
from v1_data import make_v1_splits

PIQA_SHA256 = 'f805f9c96518d7b59f2f870d90f712457cda949979469d9fa3fd706ce22e0a1a'
PIQA_LABELS_SHA256 = 'df8ae080c41f02f88e81f6932d425f4b7ee72f433ad552e6a30d45e358e8f8bb'
BOOLQ_SHA256 = '4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623'
SPLIT_MAPPING_SHA256 = '47cf5c777d07863b8def6903487697eafc970b0cd5b5d80babb7f20ded98571b'
CONFIRMATION_MANIFEST_SHA256 = '34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc'
SOURCES = {
    'piqa': {'repo': 'ybisk/ybisk.github.io',
             'revision': '21edab439af693b961be2f069e8690a88d3b4e37'},
    'boolq': {'repo': 'google/boolq',
              'revision': '35b264d03638db9f4ce671b711558bf7ff0f80d5'},
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _nonblank_lines(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            value = line.strip()
            if value:
                yield value


def _piqa_records(jsonl_lines, label_lines, held_ids):
    """Parse allowed PIQA rows, substituting ID-only placeholders for held IDs."""
    held_ids = set(held_ids)
    rows = []
    labels = iter(label_lines)
    for line in jsonl_lines:
        if not line.strip():
            continue
        try:
            label_text = next(labels)
        except StopIteration as exc:
            raise ValueError('PIQA row/label count mismatch') from exc
        row_id = len(rows)
        if row_id in held_ids:
            rows.append({'task': 'piqa', 'id': row_id})
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'PIQA row {row_id} is not valid JSON') from exc
        if not isinstance(value, dict):
            raise ValueError(f'PIQA row {row_id}: expected a JSON object')
        try:
            label = int(label_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'PIQA label {row_id} is not an integer') from exc
        if label not in (0, 1):
            raise ValueError(f'PIQA label {row_id} must be 0 or 1')
        if not all(isinstance(value.get(key), str) for key in ('goal', 'sol1', 'sol2')):
            raise ValueError(f'PIQA row {row_id} is missing string goal/sol1/sol2 fields')
        rows.append({'task': 'piqa', 'id': row_id, 'goal': value['goal'],
                     'solutions': [value['sol1'], value['sol2']], 'label': label})
    try:
        next(labels)
    except StopIteration:
        pass
    else:
        raise ValueError('PIQA row/label count mismatch')
    return rows


def _boolq_record(row, row_id):
    if not isinstance(row.get('passage'), str) or not isinstance(row.get('question'), str):
        raise ValueError(f'BoolQ row {row_id} is missing string passage/question fields')
    answer = row.get('answer')
    if not isinstance(answer, bool):
        raise ValueError(f'BoolQ row {row_id} answer must be a JSON boolean')
    return {'task': 'boolq', 'id': row_id, 'passage': row['passage'],
            'question': row['question'], 'label': answer}


def _boolq_records_parquet(table, held_ids):
    """Filter Arrow rows before Python conversion, reading only ID values first."""
    import pyarrow as pa

    if 'id' in table.column_names:
        ids = table['id'].to_pylist()
    else:
        ids = list(range(table.num_rows))
    if len(ids) != table.num_rows:
        raise ValueError('BoolQ Arrow ID count differs from row count')
    held_ids = set(held_ids)
    keep = [row_id not in held_ids for row_id in ids]
    allowed = iter(table.filter(pa.array(keep, type=pa.bool_())).to_pylist())
    result = []
    for row_id in ids:
        if row_id in held_ids:
            result.append({'task': 'boolq', 'id': row_id})
        else:
            result.append(_boolq_record(next(allowed), row_id))
    try:
        next(allowed)
    except StopIteration:
        pass
    else:
        raise ValueError('BoolQ Arrow filtering returned extra rows')
    return result


def _read_boolq(path, held_ids):
    path = Path(path)
    if path.suffix.lower() != '.parquet':
        raise ValueError('pre-opening BoolQ reader requires the pinned parquet source')
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError('reading BoolQ parquet requires pyarrow') from exc
    return _boolq_records_parquet(parquet.read_table(path), held_ids)


def _read_manifest(path):
    path = Path(path)
    if _sha256(path) != CONFIRMATION_MANIFEST_SHA256:
        raise ValueError('A2 ID manifest differs from the sealed D-0025 file')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('sealed') is not True or manifest.get('model_evaluated') is not False:
        raise ValueError('A2 ID manifest is not sealed or records model evaluation')
    return manifest


def prepare_preopening(data_dir, split_manifest):
    """Build the usual V1 splits while keeping A2 row text/labels inaccessible."""
    data_dir = Path(data_dir)
    files = {
        'piqa': data_dir / 'piqa-train.jsonl',
        'piqa_labels': data_dir / 'piqa-train-labels.lst',
        'boolq': data_dir / 'boolq-train.parquet',
    }
    expected = {'piqa': PIQA_SHA256, 'piqa_labels': PIQA_LABELS_SHA256,
                'boolq': BOOLQ_SHA256}
    for key, path in files.items():
        if _sha256(path) != expected[key]:
            raise ValueError(f'source checksum mismatch: {key}')
    manifest = _read_manifest(split_manifest)
    if not isinstance(manifest.get('ids'), dict) or set(manifest['ids']) != {'piqa', 'boolq'}:
        raise ValueError('A2 manifest task ID map is invalid')

    piqa = _piqa_records(_nonblank_lines(files['piqa']),
                         _nonblank_lines(files['piqa_labels']), manifest['ids']['piqa'])
    boolq = _read_boolq(files['boolq'], manifest['ids']['boolq'])
    original = make_v1_splits(
        piqa, boolq, source_files=files, source_metadata=SOURCES,
    )
    if any(original['metadata']['source_files'][key]['sha256'] != digest
           for key, digest in expected.items()):
        raise ValueError('source files changed while constructing the V1 split')
    if original['metadata']['split_mapping_sha256'] != SPLIT_MAPPING_SHA256:
        raise ValueError('split mapping differs from the pinned V1 mapping')
    # This regenerates the manifest from IDs alone, compares it exactly, then
    # removes placeholders and held IDs from the training rows.
    filtered = apply_confirmation(original, manifest)
    for task in ('piqa', 'boolq'):
        if any(not {'task', 'id', 'label'} <= set(row)
               for row in filtered['splits'][task]['train']):
            raise ValueError(f'{task} training rows still contain an A2 placeholder')
    return filtered


def load_data(data_dir, split_manifest, tok):
    """Pre-opening RB-style loader returning filtered splits and audit exclusions."""
    from v1_data import audit_training_exclusions

    splits = prepare_preopening(data_dir, split_manifest)
    excluded = audit_training_exclusions(splits, tok)
    return splits, excluded
