"""A2 access cannot precede the locked dev gate; synthetic rows only."""
from types import SimpleNamespace
import copy
import json

import pytest

import seg_a2_eval as runner
from test_seg_artifacts import predictions


def test_same_slice_requires_row_identity_and_other_slice_class_limit():
    expected = predictions()
    actual = [dict(r) for r in expected]
    assert runner.dev_gate(actual, expected, '2g', '2g')['passed']
    actual[0]['scores'] = [s+1 for s in actual[0]['scores']]
    assert not runner.dev_gate(actual, expected, '2g', '2g')['passed']
    assert runner.dev_gate(actual, expected, '1g', '2g')['passed']
    for i in range(10):
        actual[i].update(prediction=1-expected[i]['prediction'], scores=list(reversed(expected[i]['scores'])))
    assert runner.dev_gate(actual, expected, '1g', '2g')['passed']
    actual[10].update(prediction=1-expected[10]['prediction'], scores=list(reversed(expected[10]['scores'])))
    assert not runner.dev_gate(actual, expected, '1g', '2g')['passed']


def test_failed_lock_or_missing_unseal_never_opens_a2(monkeypatch, tmp_path):
    monkeypatch.setenv('SR_SPEC_DIGEST', 'synthetic')
    monkeypatch.setattr(runner, 'read_a2', lambda *a, **k: pytest.fail('A2 opened'))
    monkeypatch.setattr(runner, 'verify_lock', lambda *a: (_ for _ in ()).throw(ValueError('lock rejected')))
    args = SimpleNamespace(output=tmp_path, unseal=False, lock=tmp_path/'lock', lock_sha256='0'*64,
                           seed=3, dev_replay_only=False, require_training_profile=False)
    with pytest.raises(ValueError, match='unseal'):
        runner.run(args)
    args.unseal = True
    with pytest.raises(ValueError, match='lock rejected'):
        runner.run(args)


@pytest.fixture
def simulated_run(monkeypatch, tmp_path):
    monkeypatch.setenv('SR_SPEC_DIGEST', 'synthetic')
    monkeypatch.setenv('SR_CODE_COMMIT', 'b'*40)
    expected = predictions(); actual = copy.deepcopy(expected)
    entries = {a: {'root': str(tmp_path/a), 'predictions': {'path': 'synthetic-dev'}} for a in runner.ARMS}
    lock = {'training_source': 'a'*40, 'runs': {'3': entries},
            'split_manifest': {'path': str(tmp_path/'ids.json')}}
    monkeypatch.setattr(runner, 'verify_lock', lambda *a: lock)
    monkeypatch.setattr(runner, 'validate', lambda *a: ({'metrics': {'hardware': '2g'}}, {}, tmp_path/'adapter'))
    monkeypatch.setattr(runner.torch.cuda, 'get_device_name', lambda: '2g')
    monkeypatch.setattr(runner, 'tokenizer', lambda: object())
    monkeypatch.setattr(runner, 'prepare_preopening', lambda *a: {})
    model = SimpleNamespace(); model.eval = lambda: model
    monkeypatch.setattr(runner, 'make_model', lambda *a, **k: model)
    monkeypatch.setattr(runner, 'evaluate', lambda *a, **k: ({}, actual, {}))
    monkeypatch.setattr(runner, 'read', lambda *a: expected)
    monkeypatch.setattr(runner, 'read_a2', lambda *a, **k: pytest.fail('A2 read after failed dev gate'))
    args = SimpleNamespace(output=tmp_path, unseal=True, lock=tmp_path/'lock',
        lock_sha256='0'*64, seed=3, data_dir='unused', confirmation_decision='D-9999',
        dev_replay_only=False, require_training_profile=False)
    return args, actual


@pytest.mark.parametrize('hardware,reason', [('2g', 'same_profile_dev_gate_failure'),
                                           ('1g', 'cross_profile_dev_gate_failure')])
def test_failed_dev_gate_writes_partial_without_a2_receipt(monkeypatch, simulated_run, hardware, reason):
    args, actual = simulated_run
    monkeypatch.setattr(runner.torch.cuda, 'get_device_name', lambda: hardware)
    for row in actual[:11]:
        row.update(prediction=1, scores=[-3., -2.])
    with pytest.raises(ValueError, match='dev reproduction gate failed'):
        runner.run(args)
    manifest = json.loads((args.output/'manifest.json').read_text())
    assert manifest['status'] == 'partial'
    assert manifest['opened'] is False
    assert manifest['metrics']['a2.opened'] is False
    assert manifest['metrics']['failure_reason'] == reason
    assert not (args.output/'a2-read.jsonl').exists()


@pytest.mark.parametrize('hardware,flips', [('2g', 0), ('1g', 11)])
def test_dev_replay_mode_never_reads_a2_and_does_not_gate_cross_profile(monkeypatch, simulated_run, hardware, flips):
    args, actual = simulated_run
    args.unseal = False; args.dev_replay_only = True; args.confirmation_decision = None
    monkeypatch.setattr(runner.torch.cuda, 'get_device_name', lambda: hardware)
    for row in actual[:flips]:
        row.update(prediction=1, scores=[-3., -2.])
    runner.run(args)
    manifest = json.loads((args.output/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['stage'] == 'dev_replay'
    assert manifest['opened'] is False
    assert manifest['metrics']['path.dev_gate']['prediction_mismatches'] == flips
    assert manifest['metrics']['dev_replay.cross_profile_gate_enforced'] is False
    assert not (args.output/'a2-read.jsonl').exists()


def test_same_profile_dev_replay_failure_requires_repair(simulated_run):
    args, actual = simulated_run
    args.unseal = False; args.dev_replay_only = True; args.confirmation_decision = None
    actual[0].update(prediction=1, scores=[-3., -2.])
    with pytest.raises(ValueError, match='same-profile dev replay differs'):
        runner.run(args)
    manifest = json.loads((args.output/'manifest.json').read_text())
    assert manifest['status'] == 'partial' and manifest['opened'] is False


def test_cross_profile_retry_can_exit_before_dev_or_a2(monkeypatch, simulated_run):
    args, _ = simulated_run
    args.require_training_profile = True
    monkeypatch.setattr(runner.torch.cuda, 'get_device_name', lambda: '1g')
    monkeypatch.setattr(runner, 'tokenizer', lambda: pytest.fail('retry reached dev preparation on wrong profile'))
    with pytest.raises(ValueError, match='retry did not land on training profile'):
        runner.run(args)
    manifest = json.loads((args.output/'manifest.json').read_text())
    assert manifest['status'] == 'partial' and manifest['opened'] is False
    assert manifest['metrics']['failure_reason'] == 'retry_profile_miss'
    assert not (args.output/'dev-reproduction.json').exists()


def test_unencodable_a2_stops_before_any_a2_model_or_score(monkeypatch, tmp_path):
    from test_v1_data import CharacterTokenizer
    row = {'task': 'piqa', 'id': 1, 'goal': 'goal',
           'solutions': ['w'*510, 'answer'], 'label': 0}
    splits = {'splits': {'piqa': {'a2': [row]}, 'boolq': {'a2': []}}}
    monkeypatch.setattr(runner, 'make_model', lambda *a, **k: pytest.fail('A2 model loaded before encoding passed'))
    monkeypatch.setattr(runner, 'evaluate', lambda *a, **k: pytest.fail('A2 model scored before encoding passed'))
    with pytest.raises(runner.A2Unencodable, match='no A2 model scored'):
        runner.evaluate_a2_conditions({}, splits, CharacterTokenizer(), {}, 3, tmp_path)
    audit = json.loads((tmp_path/'a2-encodability.json').read_text())
    assert audit['status'] == 'unencodable_no_model_scored'
    assert audit['passed'] is False
    assert not (tmp_path/'conditions.json').exists()
