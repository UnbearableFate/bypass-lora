"""Pure E-0016 scheduling/provenance checks; no model import is required."""
import hashlib
import json
import random

import pytest

from v1_data import derive_seed
from v2_schedule import ORDER_HASH_ENCODING, lr_multiplier, make_step_schedule
from v2r16_train import (checkpoint_records, evaluation_protocols, make_arm_schedule,
                         resume_provenance, trim_smoke_schedule)


def rows(count=32):
    return [{'episode_id': f'episode-{i}', 'support_input_ids': [1]*(i % 11+1),
             'question_input_ids': [2, 3]} for i in range(count)]


@pytest.mark.parametrize('arm', ('D1', 'D3', 'D4'))
def test_from_scratch_schedule_keeps_released_v2_order(arm):
    data = rows()
    assert make_arm_schedule(data, arm, 10) == make_step_schedule(data, 10)


@pytest.mark.parametrize('arm', ('D2a', 'D2b', 'D5'))
def test_resume_order_uses_new_arm_namespace_for_each_complete_epoch(arm):
    data = rows()
    plan = make_arm_schedule(data, arm, 12)
    seed = derive_seed(20261029, 'concat_loop_v2', 'V2R16', arm, 0)
    expected = []
    for epoch in range(3):
        indices = list(range(len(data)))
        random.Random(seed+epoch).shuffle(indices)
        expected.extend(indices)
    assert plan['order'] == expected
    assert plan['batches'] == [expected[i:i+8] for i in range(0, 96, 8)]
    for start in range(0, 96, 32):
        assert sorted(plan['order'][start:start+32]) == list(range(32))
    payload = ''.join(data[i]['episode_id']+'\n' for i in expected).encode('utf-8')
    assert plan['order_sha256'] == hashlib.sha256(payload).hexdigest()
    assert plan['order_hash_encoding'] == ORDER_HASH_ENCODING
    assert plan['tokens'] == sum(len(data[i]['support_input_ids'])+2 for i in expected)
    assert plan['order_seed'] == seed


def test_resume_arms_do_not_reuse_each_other_or_d1_order():
    data = rows()
    orders = [tuple(make_arm_schedule(data, arm, 10)['order']) for arm in ('D1', 'D2a', 'D2b', 'D5')]
    assert len(set(orders)) == 4


@pytest.mark.parametrize('arm', ('D1', 'D2a', 'D2b', 'D3', 'D4', 'D5'))
def test_formal_resume_has_fresh_warmup_and_smoke_recomputes_actual_tokens(arm):
    data = rows(16384)
    plan = make_arm_schedule(data, arm)
    assert plan['steps'] == 6000
    assert plan['warmup_steps'] == 300
    assert lr_multiplier(0, plan['steps'], plan['warmup_steps']) == 1/300
    assert lr_multiplier(299, plan['steps'], plan['warmup_steps']) == 1
    assert lr_multiplier(5999, plan['steps'], plan['warmup_steps']) == 0
    smoke = trim_smoke_schedule(plan, data, 20)
    assert smoke['steps'] == 20 and smoke['warmup_steps'] == 1
    assert smoke['tokens'] == sum(len(data[i]['support_input_ids'])+2
                                  for batch in smoke['batches'] for i in batch)
    assert len(plan['batches']) == 6000
    assert smoke['order_sha256'] == plan['order_sha256']


@pytest.mark.parametrize('arm,steps,count', [('bad', 1, 32), ('D2a', 0, 32), ('D2b', 13, 32),
                                            ('D2a', 1, 31), ('D2b', 1, 0)])
def test_invalid_order_request_rejected(arm, steps, count):
    with pytest.raises(ValueError):
        make_arm_schedule(rows(count), arm, steps)


def source_checkpoint(tmp_path, *, smoke=0, arm='D1'):
    checkpoint = tmp_path/'adapter'
    checkpoint.mkdir()
    (checkpoint/'adapter.pt').write_bytes(b'fixture adapter bytes')
    (checkpoint/'config.json').write_text('{}')
    (tmp_path/'metrics.json').write_text(json.dumps({'train': {'steps': 6000},
        'checkpoint_reload': {'full_records_identical': True, 'summaries_identical': True}}))
    manifest = {'status': 'complete', 'arm': arm, 'smoke_steps': smoke, 'contract': 'E-0016@1',
                'seed': 0, 'validator': {'passed': True},
                'checkpoint_files': checkpoint_records(checkpoint)}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return checkpoint


def test_resume_records_all_checkpoint_and_source_hashes(tmp_path):
    checkpoint = source_checkpoint(tmp_path)
    provenance = resume_provenance(checkpoint, False)
    assert provenance['checkpoint_files'] == checkpoint_records(checkpoint)
    assert set(provenance['source_manifest']) == {'sha256', 'bytes'}
    assert set(provenance['source_metrics']) == {'sha256', 'bytes'}
    assert provenance['adapter_only'] and not provenance['optimizer_state_loaded']
    (checkpoint/'adapter.pt').write_bytes(b'changed adapter')
    with pytest.raises(ValueError, match='bytes differ'):
        resume_provenance(checkpoint, False)


def test_formal_resume_rejects_smoke_source(tmp_path):
    checkpoint = source_checkpoint(tmp_path, smoke=20)
    with pytest.raises(ValueError, match='smoke checkpoint'):
        resume_provenance(checkpoint, False)
    assert resume_provenance(checkpoint, True)['adapter_only']


def test_resume_requires_d1_parent(tmp_path):
    checkpoint = source_checkpoint(tmp_path, arm='D2a')
    with pytest.raises(ValueError, match='completed D1'):
        resume_provenance(checkpoint, True)


@pytest.mark.parametrize('failure', ('steps', 'reload', 'contract', 'validator'))
def test_resume_refuses_incomplete_or_wrong_protocol_source(tmp_path, failure):
    checkpoint = source_checkpoint(tmp_path)
    if failure in ('steps', 'reload'):
        path = tmp_path/'metrics.json'
        value = json.loads(path.read_text())
        if failure == 'steps':
            value['train']['steps'] = 5999
        else:
            value['checkpoint_reload']['full_records_identical'] = False
    else:
        path = tmp_path/'manifest.json'
        value = json.loads(path.read_text())
        if failure == 'contract':
            value['contract'] = 'E-0014@1'
        else:
            value['validator']['passed'] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        resume_provenance(checkpoint, False)


def test_primary_and_additional_evaluation_protocol_coverage():
    assert evaluation_protocols('D1')['teacher'] == (
        'standard', ('tf_normal', 'tf_shuffled', 'tf_wrong', 'zero'))
    d3 = evaluation_protocols('D3')
    assert d3['hard'][0] == 'hard' and len(d3['hard'][1]) == 6
    assert d3['soft'][0] == 'soft' and len(d3['soft'][1]) == 6
    assert d3['marginal'] == ('marginal', ('normal', 'zero', 'shuffled', 'permuted'))
