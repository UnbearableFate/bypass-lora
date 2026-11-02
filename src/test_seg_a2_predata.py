import sys
from types import SimpleNamespace

import pytest

import seg_a2_predata as predata
from rb_data import apply_confirmation, confirmation_manifest
from v1_data import make_v1_splits


def test_piqa_held_rows_skip_json_and_label_conversion():
    rows = predata._piqa_records(
        ['not-json-held', '{"goal":"allowed","sol1":"a","sol2":"b"}'],
        ['not-an-int-held', '1'], {0})
    assert rows == [
        {'task': 'piqa', 'id': 0},
        {'task': 'piqa', 'id': 1, 'goal': 'allowed', 'solutions': ['a', 'b'], 'label': 1},
    ]


def test_boolq_arrow_filters_before_python_row_conversion(monkeypatch):
    class Column:
        def __init__(self, values):
            self.values = values

        def to_pylist(self):
            return self.values

    class Table:
        column_names = ['id', 'passage', 'question', 'answer']

        def __init__(self, rows):
            self.rows = rows
            self.num_rows = len(rows)

        def __getitem__(self, key):
            return Column([row[key] for row in self.rows])

        def filter(self, mask):
            return Filtered([row for row, keep in zip(self.rows, mask) if keep])

    class Filtered:
        def __init__(self, rows):
            self.rows = rows

        def to_pylist(self):
            assert [row['id'] for row in self.rows] == ['keep']
            return self.rows

    monkeypatch.setitem(sys.modules, 'pyarrow', SimpleNamespace(array=lambda values, type: values,
                                                                 bool_=bool))
    result = predata._boolq_records_parquet(Table([
        {'id': 'sealed', 'passage': 'must not materialize', 'question': 'x', 'answer': 'poison'},
        {'id': 'keep', 'passage': 'visible', 'question': 'ok', 'answer': True},
    ]), {'sealed'})
    assert result == [
        {'task': 'boolq', 'id': 'sealed'},
        {'task': 'boolq', 'id': 'keep', 'passage': 'visible', 'question': 'ok', 'label': True},
    ]


def test_placeholders_are_removed_from_training_before_audit(monkeypatch):
    # Enough rows to produce a realistic D-0025 1,000-ID sample in each train split.
    piqa = [{'task': 'piqa', 'id': i, 'goal': 'g', 'solutions': ['a', 'b'],
             'label': i % 2} for i in range(4100)]
    boolq = [{'task': 'boolq', 'id': f'b{i}', 'passage': 'p', 'question': 'q',
              'label': bool(i % 2)} for i in range(4100)]
    original = make_v1_splits(piqa, boolq)
    manifest = confirmation_manifest(original)
    held = {task: set(manifest['ids'][task]) for task in ('piqa', 'boolq')}
    placeholder_splits = {'metadata': original['metadata'], 'splits': {}}
    for task in ('piqa', 'boolq'):
        placeholder_splits['splits'][task] = {}
        for split, rows in original['splits'][task].items():
            placeholder_splits['splits'][task][split] = [
                ({'task': task, 'id': row['id']} if row['id'] in held[task] else row)
                for row in rows
            ]
    filtered = apply_confirmation(placeholder_splits, manifest)

    def audit(splits, tok):
        for task in ('piqa', 'boolq'):
            for row in splits['splits'][task]['train']:
                assert 'label' in row
                assert ('goal' in row and 'solutions' in row) if task == 'piqa' else (
                    'passage' in row and 'question' in row)
        return []

    monkeypatch.setattr(predata, 'prepare_preopening', lambda data_dir, split_manifest: filtered)
    monkeypatch.setattr('v1_data.audit_training_exclusions', audit)
    splits, excluded = predata.load_data('unused', 'unused', object())
    assert splits is filtered
    assert excluded == []
    for task in ('piqa', 'boolq'):
        assert not held[task].intersection(row['id'] for row in splits['splits'][task]['train'])


def test_invalid_allowed_piqa_row_still_fails():
    with pytest.raises(ValueError, match='not valid JSON'):
        predata._piqa_records(['not-json'], ['0'], set())
