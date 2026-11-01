"""E-0017 seed options: seed 0 keeps the recorded orders and initialization, seeds 1 and 2
give new ones, and the D5 start run must carry the expected seed. The released-data
checks read train splits only; the initialization check needs a GPU allocation."""
import json
from pathlib import Path
import random
import shutil

import pytest

from v1_data import derive_seed
from v2_schedule import load_data, make_step_schedule, train_order
from v2r16_train import make_arm_schedule

RUNS = '/work/xg24i002/x10041/bypass-lora/runs'
V2 = f'{RUNS}/ATT-0039-1/data'
E1 = f'{RUNS}/ATT-0112-1/e1'


def rows(count=32):
    return [{'episode_id': f'episode-{i}', 'support_input_ids': [1]*(i % 11+1),
             'question_input_ids': [2, 3]} for i in range(count)]


def test_seed_zero_is_the_recorded_order_and_other_seeds_append_the_seed():
    data = rows()
    assert make_step_schedule(data, 10, seed=0) == make_step_schedule(data, 10)
    assert 'order_seed' not in make_step_schedule(data, 10)
    orders = {s: tuple(make_step_schedule(data, 12, seed=s)['order']) for s in (0, 1, 2)}
    assert len(set(orders.values())) == 3
    expected = []
    for epoch in range(3):
        indices = list(range(32))
        random.Random(derive_seed(20261029, 'concat_loop_v2', 'V2', f'train-order-{epoch}', 2)).shuffle(indices)
        expected.extend(indices)
    assert list(orders[2]) == expected
    assert make_step_schedule(data, 12, seed=2)['order_seed'] == 2
    for bad in (-1, 1.0, True):
        with pytest.raises(ValueError):
            train_order(data, 'V2', bad)


def test_continuation_seed_only_for_resumed_arms():
    data = rows()
    assert make_arm_schedule(data, 'D5', 12, 0) == make_arm_schedule(data, 'D5', 12)
    plan = make_arm_schedule(data, 'D5', 12, 1)
    assert plan['order_seed'] == derive_seed(20261029, 'concat_loop_v2', 'V2R16', 'D5', 1)
    assert plan['continuation_seed'] == 1
    assert 'arm, 1);' in plan['order_seed_rule'] and 'continuation_seed' not in make_arm_schedule(data, 'D5', 12)
    assert len({tuple(make_arm_schedule(data, 'D5', 12, s)['order']) for s in (0, 1, 2)}) == 3
    for arm in ('D1', 'D3', 'D4'):
        with pytest.raises(ValueError):
            make_arm_schedule(data, arm, 12, 1)


def test_released_data_seed_zero_reproduces_the_recorded_r1_and_d5_orders():
    from v2r_data import load_variant
    e1 = load_variant(E1, 1, 'train')[1]
    r1 = json.loads(open(f'{RUNS}/ATT-0122-1/schedule.json').read())
    plan = make_step_schedule(e1, 6000)
    assert plan['order_sha256'] == r1['order_sha256'] and plan['tokens'] == r1['tokens']
    assert make_step_schedule(e1, 6000, seed=1)['order_sha256'] != r1['order_sha256']
    v2 = load_data(f'{V2}/train.jsonl', 'train')
    d5 = json.loads(open(f'{RUNS}/ATT-0159-1/run/schedule.json').read())
    assert make_arm_schedule(v2, 'D5', 6000)['order_sha256'] == d5['order_sha256']
    seeded = {s: make_arm_schedule(v2, 'D5', 6000, s) for s in (1, 2)}
    assert len({d5['order_sha256'], *(p['order_sha256'] for p in seeded.values())}) == 3
    assert all(p['warmup_steps'] == 300 and p['steps'] == 6000 for p in seeded.values())


def r1_copy(tmp_path, runner, name='r1', **manifest_changes):
    copy = tmp_path/name
    (copy/runner.ARM_DIR).mkdir(parents=True)
    shutil.copytree(runner.R1_RUN/runner.ARM_DIR/'adapter', copy/runner.ARM_DIR/'adapter')
    shutil.copy(runner.R1_RUN/'metrics.json', copy/'metrics.json')
    manifest = json.loads((runner.R1_RUN/'manifest.json').read_text())
    (copy/'manifest.json').write_text(json.dumps(dict(manifest, **manifest_changes)))
    return copy


def test_start_run_provenance_checks_the_recorded_seed(tmp_path):
    import v2r16_isolated as runner
    from v2_memory import config
    cfg = config('direct_hidden', 'soft_slot', 'F_out')
    copy = r1_copy(tmp_path, runner, seed=1)
    pin = runner.R1_ADAPTER_SHA256
    assert runner.r1_provenance(copy, cfg, 1, adapter_sha256=pin)['start_seed'] == 1
    with pytest.raises(ValueError, match='at seed 0'):
        runner.r1_provenance(copy, cfg, adapter_sha256=pin)
    with pytest.raises(ValueError, match='at seed 2'):
        runner.r1_provenance(copy, cfg, 2, adapter_sha256=pin)
    assert runner.r1_provenance(runner.R1_RUN, cfg, 0, adapter_sha256=pin)['start_seed'] == 0


def test_start_adapter_is_bound_to_the_expected_producer_digest(tmp_path):
    """RV-0050: an unchanged or byte-identical source passes; a different loadable
    same-config adapter.pt, a wrong digest or no digest fails before loading."""
    import v2r16_isolated as runner
    from v2_memory import config
    from v2r16_train import file_record
    cfg = config('direct_hidden', 'soft_slot', 'F_out')
    pin = runner.R1_ADAPTER_SHA256
    assert file_record(runner.R1_RUN/runner.ARM_DIR/'adapter'/'adapter.pt')['sha256'] == pin
    record = runner.r1_provenance(r1_copy(tmp_path, runner, 'same'), cfg, adapter_sha256=pin)
    assert record['expected_adapter_sha256'] == pin and record['checkpoint_files']['adapter.pt']['sha256'] == pin
    swapped = r1_copy(tmp_path, runner, 'swapped')
    d5 = Path(RUNS)/'ATT-0159-1'/'run'/'adapter'
    assert (d5/'config.json').read_bytes() == (swapped/runner.ARM_DIR/'adapter'/'config.json').read_bytes()
    shutil.copy(d5/'adapter.pt', swapped/runner.ARM_DIR/'adapter'/'adapter.pt')
    with pytest.raises(ValueError, match='expected producer digest'):
        runner.r1_provenance(swapped, cfg, adapter_sha256=pin)
    # The swapped source passes only with its own digest, which the job must pin explicitly.
    own = file_record(d5/'adapter.pt')['sha256']
    assert runner.r1_provenance(swapped, cfg, adapter_sha256=own)['checkpoint_files']['adapter.pt']['sha256'] == own
    for bad in (None, '', pin.upper(), pin[:-1], 'x'*64):
        with pytest.raises(ValueError, match='expected lowercase hex sha256'):
            runner.r1_provenance(runner.R1_RUN, cfg, adapter_sha256=bad)


def test_cli_requires_a_pinned_digest_for_other_start_runs(monkeypatch, tmp_path):
    import sys
    import v2r16_isolated as runner
    called = []
    monkeypatch.setattr(runner, 'run', lambda args: called.append(args) or 0)
    common = ['v2r16_isolated.py', '--arm', 'D5', '--data-dir', V2, '--output', str(tmp_path/'out'),
              '--experiment', 'E-0017', '--seed', '1']
    monkeypatch.setattr(sys, 'argv', common + ['--r1-seed', '1', '--r1-run', str(tmp_path/'stage-a')])
    with pytest.raises(SystemExit):
        runner.main()
    monkeypatch.setattr(sys, 'argv', common + ['--r1-seed', '1', '--r1-run', str(tmp_path/'stage-a'),
                                               '--r1-adapter-sha256', 'a'*64])
    assert runner.main() == 0 and called[-1].r1_adapter_sha256 == 'a'*64
    monkeypatch.setattr(sys, 'argv', common)  # ATT-0122-1 defaults to the trusted R1 digest
    assert runner.main() == 0 and called[-1].r1_adapter_sha256 == runner.R1_ADAPTER_SHA256


def test_adapter_initialization_follows_the_seed():
    import torch
    from calibrate import load_base
    from v2_memory import config
    from v2_train import adapted
    base = load_base()
    cfg = config('direct_hidden', 'soft_slot', 'F_out')

    def initial(**kwargs):
        with adapted(base, cfg=cfg, **kwargs) as model:
            return {n: p.detach().clone() for n, p in model.adapters()[14].side.named_parameters()}
    default, zero, one, two = initial(), initial(seed=0), initial(seed=1), initial(seed=2)
    assert set(default) == {'projection.A_q', 'projection.A_k', 'projection.A_v', 'reader.Rq', 'reader.Ro', 'fusion.P'}
    assert all(torch.equal(default[n], zero[n]) for n in default)
    # Random projections follow the seed.
    for name in ('projection.A_q', 'projection.A_k', 'projection.A_v'):
        assert not torch.equal(default[name], one[name]) and not torch.equal(one[name], two[name])
        assert not torch.equal(default[name], two[name])
    # Deterministic initializers do not: identity readers and the zero exit.
    for state in (default, one, two):
        for name in ('reader.Rq', 'reader.Ro'):
            assert torch.equal(state[name], torch.eye(state[name].shape[0], device=state[name].device))
        assert torch.count_nonzero(state['fusion.P']) == 0


def comparison(*, mismatches=4, nll=.3117866516113281, same=True, episodes=1000):
    return {'normal': {'recorded': 'x', 'episodes': episodes, 'same_episode_order': same,
                       'prediction_mismatches': mismatches if same else None,
                       'nll_max_abs_difference': nll if same else None}}


def test_start_replay_records_numerical_differences_and_rejects_other_episodes():
    import v2r16_isolated as runner
    # Aligned replay with numerical differences (as ATT-0146-1 recorded for R1 on another slice): recorded.
    record = runner.replay_record(comparison(), 1000)
    assert record['prediction_mismatches'] == 4 and record['nll_max_abs_difference'] == .3117866516113281
    assert record['numerical_differences'].startswith('recorded only')
    assert runner.replay_record(comparison(mismatches=0, nll=0.), 1000)['prediction_mismatches'] == 0
    with pytest.raises(AssertionError, match='same dev episodes'):
        runner.replay_record(comparison(same=False), 1000)
    with pytest.raises(AssertionError, match='same dev episodes'):
        runner.replay_record(comparison(episodes=999), 1000)
