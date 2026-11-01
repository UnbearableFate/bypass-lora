"""E-0017 seed options: seed 0 keeps the recorded orders and initialization, seeds 1 and 2
give new ones, and the D5 start run must carry the expected seed. The released-data
checks read train splits only; the initialization check needs a GPU allocation."""
import json
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


def test_start_run_provenance_checks_the_recorded_seed(tmp_path):
    import v2r16_isolated as runner
    from v2_memory import config
    cfg = config('direct_hidden', 'soft_slot', 'F_out')
    copy = tmp_path/'r1'
    (copy/runner.ARM_DIR).mkdir(parents=True)
    shutil.copytree(runner.R1_RUN/runner.ARM_DIR/'adapter', copy/runner.ARM_DIR/'adapter')
    shutil.copy(runner.R1_RUN/'metrics.json', copy/'metrics.json')
    manifest = json.loads((runner.R1_RUN/'manifest.json').read_text())
    (copy/'manifest.json').write_text(json.dumps(dict(manifest, seed=1)))
    assert runner.r1_provenance(copy, cfg, 1)['start_seed'] == 1
    with pytest.raises(ValueError, match='at seed 0'):
        runner.r1_provenance(copy, cfg)
    with pytest.raises(ValueError, match='at seed 2'):
        runner.r1_provenance(copy, cfg, 2)
    assert runner.r1_provenance(runner.R1_RUN, cfg, 0)['start_seed'] == 0


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
    assert all(torch.equal(default[n], zero[n]) for n in default)
    random_names = [n for n in default if torch.count_nonzero(default[n])]
    assert random_names
    assert all(not torch.equal(default[n], one[n]) and not torch.equal(one[n], two[n]) for n in random_names)
    # Zero-initialized parameters (the exit) stay zero at every seed.
    assert all(torch.count_nonzero(one[n]) == 0 for n in default if n not in random_names)
