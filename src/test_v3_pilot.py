"""V3 data-selection/schedule/stratification regressions; run only in PBS."""
import copy
from pathlib import Path
import random

import pytest

from v1_data import derive_seed
from v2_schedule import make_schedule
from v3_pilot_data import load_data, stratify

DATA_DIR = Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0077-1/data')


def test_final_v3_data_pin_released_only_and_complete(tmp_path):
    train = load_data(DATA_DIR / 'train.jsonl', 'train')
    calibration = load_data(DATA_DIR / 'calibration.jsonl', 'calibration')
    assert len(train) == 18432 and len(calibration) == 2048
    assert {r['identity']['phase'] for r in train} == {'V3'}
    assert {r['axes']['hop'] for r in train} == {1, 2, 3}
    for split in ('confirmation', 'ood_h4', 'ood_h6', 'ood_h8', 'ood_unseen'):
        with pytest.raises(ValueError, match='released'):
            load_data(DATA_DIR / (split + '.registry.jsonl'), split)
    with pytest.raises(ValueError, match='released'):
        load_data(DATA_DIR / 'dev.jsonl', 'calibration')
    (tmp_path / 'metadata.json').write_text('{}')
    with pytest.raises(ValueError, match='metadata digest'):
        load_data(tmp_path / 'train.jsonl', 'train')


def test_phase_schedule_preserves_v2_default_and_uses_v3_seed_purpose():
    rows = [dict(episode_id=f'episode-{i}', support_input_ids=[1] * (i + 2), question_input_ids=[2])
            for i in range(32)]
    v2 = make_schedule(rows, 200)
    assert v2 == make_schedule(rows, 200, phase='V2')
    v3 = make_schedule(rows, 200, phase='V3')
    expected = []
    for epoch in range(3):
        indices = list(range(32))
        random.Random(derive_seed(20261029, 'concat_loop_v2', 'V3', f'train-order-{epoch}')).shuffle(indices)
        expected.extend(indices)
    assert v3['order'] == expected and v3['order'] != v2['order']
    assert v3['tokens'] >= 200
    previous = sum(len(rows[i]['support_input_ids']) + len(rows[i]['question_input_ids'])
                   for batch in v3['batches'][:-1] for i in batch)
    assert previous < 200
    assert all(len(batch) == 8 for batch in v3['batches'])


def test_stratification_joins_identity_and_reports_meaningful_hop_order_scores():
    rows = [dict(episode_id=f'row-{i}', axes={'hop': 1 + i % 3,
                  'support_order': 'natural' if i % 2 == 0 else 'shuffled'},
                 query={'answer_token_id': 10 + i}) for i in range(12)]
    records = [dict(episode_id=row['episode_id'], target=10 + i, prediction=10 + i if i % 2 else 99,
                    subject=99, nll=float(i)) for i, row in enumerate(rows)]
    metrics = stratify(rows, list(reversed(records)), 11)
    assert metrics['hop=2.examples'] == metrics['hop=3.examples'] == 4
    assert metrics['hops2-3.examples'] == 8 and metrics['hops2-3.em'] == .5
    assert metrics['order=natural.em'] == 0 and metrics['order=shuffled.em'] == 1
    assert metrics['hop=2,order=natural.subject_copy'] == 1
    assert metrics['hop=2.nll'] == sum((1., 4., 7., 10.)) / 4
    for invalid in (records[:-1], records[:-1] + [records[0]]):
        with pytest.raises(ValueError, match='count|identity'):
            stratify(rows, invalid, 11)
    wrong = copy.deepcopy(records)
    wrong[0]['target'] = -1
    with pytest.raises(ValueError, match='target'):
        stratify(rows, wrong, 11)
