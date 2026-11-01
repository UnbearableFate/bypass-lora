"""E-0014 option regressions (centered cosine read, prior path, address loss, step and
curriculum schedules, parameter freezing). Execute in the registered GPU allocation."""
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from bypass_core import LatentReader, MemoryState, centered_cosine_scores, elu_plus_one, rmsnorm_no_affine
from bypass_model import AdapterConfig, CENTERED_COS_TEMPERATURE_INIT
from calibrate import MODEL, REVISION, load_base
from v2_memory import (CONTROLS, address_log_weight, build_support, capture_projection, collate, config,
                       evaluate_batch, read_vectors, slot_columns, train_forward)
from v2_run import fixture
from v2_schedule import make_curriculum_schedule, make_schedule, make_step_schedule
from v2_train import adapted

V2_SIDE_KEYS = {'projection.A_q', 'projection.A_k', 'projection.A_v', 'reader.Rq', 'reader.Ro', 'fusion.P'}


# ---- core: centered cosine slot read ------------------------------------------------

def test_centered_cosine_matches_hand_formula_and_uses_only_valid_same_row_slots():
    torch.manual_seed(0)
    keys = torch.rand(2, 4, 3) + .1
    valid = torch.tensor([[True, True, True, False], [True, False, True, True]])
    q = torch.randn(2, 3)
    tau = torch.tensor(7.)
    scores = centered_cosine_scores(q, keys, valid, tau)
    for b in range(2):
        mean = keys[b, valid[b]].mean(0)
        for n in range(4):
            expected = tau * F.cosine_similarity(q[b], keys[b, n] - mean, dim=0)
            torch.testing.assert_close(scores[b, n], expected, atol=1e-6, rtol=1e-5)
    changed = keys.clone()
    changed[0, 3] = 100.  # invalid slot of row 0
    changed[1] += 5.      # another row
    torch.testing.assert_close(centered_cosine_scores(q, changed, valid, tau)[0, :3], scores[0, :3])


def test_centered_cos_state_read_softmax_masking_and_temperature_contract():
    torch.manual_seed(1)
    state = MemoryState.empty(2, 3, 'centered_cos_slot')
    k, v = torch.rand(2, 3, 3) + .1, torch.randn(2, 3, 3)
    write = torch.tensor([[True, True, True], [True, False, True]])
    for i in range(3):
        state = state.write(k[:, i], v[:, i], write[:, i])
    q, tau = torch.randn(2, 3), torch.tensor(10.)
    out = state.read(q, temperature=tau)
    weights = centered_cosine_scores(q, state.slot_keys, state.slot_valid, tau).masked_fill(~state.slot_valid, -torch.inf).softmax(-1)
    torch.testing.assert_close(out, torch.einsum('bn,bnv->bv', weights, state.slot_values))
    assert float(weights[1, 1]) == 0
    with pytest.raises(ValueError, match='temperature'):
        state.read(q)
    soft = MemoryState.empty(2, 3, 'soft_slot').write(k[:, 0], v[:, 0])
    with pytest.raises(ValueError, match='only centered'):
        soft.read(q, temperature=tau)
    assert torch.equal(MemoryState.empty(2, 3, 'centered_cos_slot').read(q, temperature=tau), torch.zeros(2, 3))


def test_temperature_parameter_exists_only_for_centered_and_gets_gradient():
    assert set(LatentReader(4).state_dict()) == {'Rq', 'Ro'}
    reader = LatentReader(4, temperature_init=CENTERED_COS_TEMPERATURE_INIT)
    assert float(reader.temperature()) == pytest.approx(10.)
    state = MemoryState.empty(1, 4, 'centered_cos_slot')
    for i in range(3):
        state = state.write(torch.rand(1, 4) + i, torch.randn(1, 4))
    reader(torch.randn(1, 4), state).sum().backward()
    assert reader.log_temperature.grad is not None and float(reader.log_temperature.grad.abs()) > 0


# ---- schedules -----------------------------------------------------------------------

def rows_of(n, prefix='r', size=1):
    return [{'episode_id': f'{prefix}-{i}', 'support_input_ids': [1]*(size+i % 5),
             'question_input_ids': [2]*(1+i % 3), 'query': {'answer_token_id': 7}} for i in range(n)]


def test_step_schedule_is_token_schedule_prefix_and_bounded():
    rows = rows_of(64)
    token_plan = make_schedule(rows, 300)
    step_plan = make_step_schedule(rows, token_plan['steps'])
    for key in ('order', 'batches', 'steps', 'tokens', 'warmup_steps', 'order_sha256'):
        assert step_plan[key] == token_plan[key]
    assert make_step_schedule(rows, 24)['batches'] == [token_plan['order'][i:i+8] for i in range(0, 192, 8)]
    for bad in (0, -1, 25, 1.5):
        with pytest.raises(ValueError):
            make_step_schedule(rows, bad)


def test_curriculum_switches_by_step_and_uses_each_segment_order():
    sets = [(f'e{f}', rows_of(16, f'e{f}', f), n) for f, n in ((1, 2), (4, 3), (8, 1), (16, 4))]
    rows, plan = make_curriculum_schedule(sets)
    assert plan['steps'] == 10 and len(plan['batches']) == 10 and plan['warmup_steps'] == 1
    expected_labels = ['e1']*2 + ['e4']*3 + ['e8'] + ['e16']*4
    for batch, label in zip(plan['batches'], expected_labels):
        assert {rows[i]['episode_id'].split('-')[0] for i in batch} == {label}
    start = 0
    for (label, segment_rows, steps), part in zip(sets, plan['curriculum']):
        own = make_step_schedule(segment_rows, steps)['batches']
        got = [[rows[i]['episode_id'] for i in b] for b in plan['batches'][start:start+steps]]
        assert got == [[segment_rows[i]['episode_id'] for i in b] for b in own]
        assert (part['first_step'], part['last_step']) == (start+1, start+steps)
        start += steps
    assert plan['tokens'] == sum(len(rows[i]['support_input_ids'])+len(rows[i]['question_input_ids'])
                                 for b in plan['batches'] for i in b)
    with pytest.raises(ValueError, match='share'):
        make_curriculum_schedule([('a', rows_of(8), 1), ('b', rows_of(8), 1)])


# ---- training loop: only requires_grad parameters move ----------------------------------

def test_train_optimizes_only_parameters_that_require_grad(monkeypatch, tmp_path):
    import v2_train

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()
            self.side = torch.nn.ModuleDict({'memory': torch.nn.Linear(2, 3, bias=False),
                                             'prior': torch.nn.Linear(2, 3, bias=False)})

        def adapter_parameters(self):
            return self.side.parameters()

        def adapters(self):
            return {14: SimpleNamespace(side=self.side)}

    rows = [{'features': [i/7, (i % 3)/2], 'label': i % 3, 'tokens': 3} for i in range(8)]
    schedule = {'steps': 2, 'warmup_steps': 1, 'batches': [list(range(8))]*2, 'tokens': 48}

    def fake_collate(selected, pad_id, device):
        return SimpleNamespace(x=torch.tensor([r['features'] for r in selected]),
                               labels=torch.tensor([r['label'] for r in selected]),
                               write_mask=torch.ones(len(selected), 1, dtype=torch.bool), input_tokens=3*len(selected))

    seen_options = []

    def fake_forward(model, batch, **options):
        seen_options.append(options)
        logits = model.side['memory'](batch.x) + model.side['prior'](batch.x)
        return (F.cross_entropy(logits, batch.labels), logits,
                {14: {'side': {'reads': 0, 'writes': len(batch.labels)}}}, {14: {'side': {'reads': 1, 'writes': 0}}})

    monkeypatch.setattr(v2_train, 'collate', fake_collate)
    monkeypatch.setattr(v2_train, 'train_forward', fake_forward)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    model = Tiny()
    model.side['prior'].weight.requires_grad_(False)
    frozen = model.side['prior'].weight.detach().clone()
    moving = model.side['memory'].weight.detach().clone()
    result = v2_train.train(model, rows, schedule, 0, .1, 8, tmp_path, [], None,
                            forward_options={'zero_memory': True}, learning_curves=False, log_name='x.jsonl')
    assert torch.equal(model.side['prior'].weight, frozen)
    assert not torch.equal(model.side['memory'].weight, moving)
    assert result['trainable_parameters'] == 6 and result['stored_parameters'] == 12
    assert seen_options == [{'zero_memory': True}]*2
    assert len((tmp_path/'x.jsonl').read_text().splitlines()) == 2


# ---- real-model checks ------------------------------------------------------------------

@pytest.fixture(scope='module')
def base():
    return load_base()


@pytest.fixture(scope='module')
def varied_rows():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    episodes = fixture(tok, n_facts=3) + fixture(tok, n_facts=4)[:1]
    for i, row in enumerate(episodes):
        row['episode_id'] = f'v2r-{i}'
        row['question_input_ids'] = tok.encode('Please '*i, add_special_tokens=False) + row['question_input_ids']
        row['query']['target_fact_index'] = (i+1) % len(row['facts'])
    return episodes, tok.pad_token_id


def randomize_exits(model, std=.03):
    side = model.adapters()[14].side
    with torch.no_grad():
        for parameter in side.fusion.parameters():
            parameter.normal_(std=std)
        if side.prior is not None:
            side.prior.P.normal_(std=std)


def test_default_config_keeps_v2_parameters_and_config():
    cfg = config('direct_hidden', 'soft_slot', 'F_out')
    assert cfg.prior_rank == 0
    data = asdict(cfg)
    data.pop('prior_rank')
    data['layers'] = tuple(data['layers'])
    assert AdapterConfig(**data) == cfg  # a V2 config.json without the field loads unchanged
    with pytest.raises(ValueError):
        AdapterConfig(branch='tokenwise_silu', prior_rank=4)


@torch.no_grad()
def test_default_side_state_and_initial_values_unchanged_by_new_options(base):
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        plain = {k: v.clone() for k, v in model.adapters()[14].side.state_dict().items()}
    assert set(plain) == V2_SIDE_KEYS
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out', 32)) as model:
        prior = model.adapters()[14].side.state_dict()
        assert set(prior) == V2_SIDE_KEYS | {'prior.A', 'prior.P'}
        assert all(torch.equal(plain[k], prior[k]) for k in plain)
        assert not prior['prior.P'].any()
    with adapted(base, cfg=config('direct_hidden', 'centered_cos_slot', 'F_out')) as model:
        cos = model.adapters()[14].side.state_dict()
        assert set(cos) == V2_SIDE_KEYS | {'reader.log_temperature'}
        assert all(torch.equal(plain[k], cos[k]) for k in plain)
        assert float(cos['reader.log_temperature'].exp()) == pytest.approx(10.)


@torch.no_grad()
def test_prior_path_query_only_zero_init_identity_and_reload(base, varied_rows, tmp_path):
    rows, pad_id = varied_rows
    batch = collate(rows, pad_id)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        randomize_exits(model)
        reference_support, _ = build_support(model, batch)
        _, reference, _, _ = train_forward(model, batch)
        fusion = model.adapters()[14].side.fusion.P.clone()
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out', 32)) as model:
        model.eval()
        side = model.adapters()[14].side
        with torch.no_grad():
            side.fusion.P.copy_(fusion)
        _, zero_init, _, _ = train_forward(model, batch)
        assert torch.equal(zero_init, reference)
        side.prior.P.normal_(std=.05)
        support, _ = build_support(model, batch)
        assert torch.equal(support.memory[14].slot_keys, reference_support.memory[14].slot_keys)
        assert torch.equal(support.memory[14].slot_values, reference_support.memory[14].slot_values)
        _, changed, _, _ = train_forward(model, batch)
        assert not torch.equal(changed, reference)
        with pytest.raises(ValueError, match='question forwards'):
            model(batch.question_ids, batch.question_mask, phase='F')
        details, _ = evaluate_batch(model, batch, 'zero')
        assert all(x > 0 for x in details['_prior_to_residual_last'])
        model.export_adapter(tmp_path/'adapter')
        expected = {k: v.clone() for k, v in side.state_dict().items()}
    with adapted(base, checkpoint=tmp_path/'adapter') as reloaded:
        reloaded.eval()
        assert reloaded.config.prior_rank == 32
        state = reloaded.adapters()[14].side.state_dict()
        assert all(torch.equal(expected[k], state[k]) for k in expected) and set(state) == set(expected)
        _, after, _, _ = train_forward(reloaded, batch)
        assert torch.equal(after, changed)


@torch.no_grad()
def test_centered_cos_diagnostic_read_reproduces_model_under_all_controls(base, varied_rows, tmp_path):
    rows, pad_id = varied_rows
    batch = collate(rows, pad_id)
    donors = collate(rows[1:]+rows[:1], pad_id)
    tolerance = json.loads((Path(__file__).parent/'tolerances.json').read_text())
    with adapted(base, cfg=config('direct_hidden', 'centered_cos_slot', 'F_out')) as model:
        model.eval()
        randomize_exits(model)
        model.adapters()[14].side.reader.log_temperature.fill_(math.log(3.))
        for mode in CONTROLS:
            details, logits = evaluate_batch(model, batch, mode, **({'donor_batch': donors} if mode == 'shuffled' else {}))
            assert math.isfinite(details['nll']) and torch.isfinite(logits).all()
            assert details['read.reconstruction_max_abs'] < 1e-4
            if mode == 'normal':
                normal = logits
                assert 0 < details['address.target_weight'] < 1
            if mode == 'zero':
                assert not torch.equal(normal, logits)
        _, separate = evaluate_batch(model, collate(rows[:1], pad_id))
        torch.testing.assert_close(separate, normal[:1], atol=tolerance['bf16_atol'], rtol=tolerance['bf16_rtol'])
        model.export_adapter(tmp_path/'cos')
    with adapted(base, checkpoint=tmp_path/'cos') as reloaded:
        reloaded.eval()
        assert float(reloaded.adapters()[14].side.reader.temperature()) == pytest.approx(3.)
        assert torch.equal(evaluate_batch(reloaded, batch)[1], normal)


def test_address_loss_matches_diagnostic_weight_and_default_forward(base, varied_rows):
    rows, pad_id = varied_rows
    batch = collate(rows, pad_id)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        randomize_exits(model)
        with torch.no_grad():
            model.adapters()[14].side.projection.A_q.mul_(4.)  # make the address non-uniform
            details, eval_logits = evaluate_batch(model, batch)
            plain, logits, _, _ = train_forward(model, batch)
            explicit, logits0, _, _ = train_forward(model, batch, address_weight=0.)
            assert torch.equal(plain, explicit) and torch.equal(logits, logits0)
        target = torch.tensor([r['query']['target_fact_index'] for r in rows], device='cuda')
        last_weights = torch.tensor(details['_last_target_weight'], device='cuda')
        side = model.adapters()[14].side
        log = {}
        loss, logits_a, _, _ = train_forward(model, batch, address_weight=1., log=log)
        assert torch.equal(logits_a.detach(), logits)
        torch.testing.assert_close(torch.tensor(log['address_loss']), -last_weights.log().mean().cpu(), atol=1e-5, rtol=1e-5)
        assert log['answer_ce'] == pytest.approx(float(plain), abs=1e-6)
        assert float(loss) == pytest.approx(log['answer_ce'] + log['address_loss'], abs=1e-5)
        # The extra term reaches only addressing parameters (A_q, Rq, A_k), never values or fusion.
        model.zero_grad(set_to_none=True)
        support, _ = build_support(model, batch)
        memory = support.select(batch.owners).memory
        with capture_projection(side) as seen:
            model(batch.question_ids, batch.question_mask, memory=memory, phase='query')
        (-address_log_weight(seen['q'], memory[14], side.reader, batch.question_mask, target).mean()).backward()
        grads = {n: p.grad for n, p in side.named_parameters()}
        assert all(grads[n] is not None and grads[n].abs().sum() > 0 for n in ('projection.A_q', 'projection.A_k', 'reader.Rq'))
        assert all(grads[n] is None or not grads[n].any() for n in ('projection.A_v', 'reader.Ro', 'fusion.P'))
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def test_slot_columns_follow_valid_slot_order_and_zero_memory_matches_zero_control(base, varied_rows):
    valid = torch.tensor([[True, False, True, True], [False, True, True, False]])
    assert slot_columns(valid, torch.tensor([2, 1])).tolist() == [3, 2]
    with pytest.raises(ValueError):
        slot_columns(valid, torch.tensor([3, 0]))
    rows, pad_id = varied_rows
    batch = collate(rows, pad_id)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out', 32)) as model:
        model.eval()
        randomize_exits(model)
        _, zero_eval = evaluate_batch(model, batch, 'zero')
        _, zero_train, _, _ = train_forward(model, batch, zero_memory=True)
        assert torch.equal(zero_eval, zero_train)


# ---- released data: pinned variants and the E=16 step schedule ---------------------------

def test_pinned_variants_load_and_step_schedule_reproduces_v2_token_schedule():
    from v2r_data import RELEASE_DIR, VARIANTS, load_variant
    from v2_schedule import load_data
    for n in VARIANTS:
        for split, count in (('train', 16384), ('calibration', 2048), ('dev', 1000)):
            metadata, rows = load_variant(Path(RELEASE_DIR)/f'e{n}', n, split)
            assert len(rows) == count and all(len(r['facts']) == n for r in rows)
        with pytest.raises(ValueError):
            load_variant(Path(RELEASE_DIR)/f'e{n}', {1: 4, 4: 8, 8: 1}[n], 'dev')
    v2 = load_data('/work/xg24i002/x10041/bypass-lora/runs/ATT-0039-1/data/train.jsonl', 'train')
    token_plan, step_plan = make_schedule(v2, 4_000_000), make_step_schedule(v2, 6000)
    assert step_plan['batches'] == token_plan['batches'] and step_plan['tokens'] == token_plan['tokens'] == 4000470
    assert step_plan['warmup_steps'] == 300 and step_plan['order_sha256'] == token_plan['order_sha256']
