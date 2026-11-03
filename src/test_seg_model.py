from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from bypass_model import BranchModel
from calibrate import load_base
from seg_model import make_model, forward, KEEP, PARAMETERS, first_forward, state_description
from test_rb_model import attached
from seg_model import config_for


@pytest.fixture(scope='module')
def base():
    return load_base()


def ids():
    return (torch.tensor([[11, 29, 73, 181, 8]], device='cuda'),
            torch.tensor([[71, 9, 18, 41]], device='cuda'))


def nonzero(model):
    with torch.no_grad():
        for layer in model.adapters().values():
            layer.side.fusion.P.normal_(std=.03)


@contextmanager
def model_for(base, arm):
    with patch('seg_model.load_base', return_value=base):
        model = make_model(arm)
    try:
        yield model
    finally:
        if isinstance(model, BranchModel):
            for i, layer in model.adapters().items():
                for h in layer.handles: h.remove()
                base.model.layers[i] = layer.original
        elif arm == 'SEG-04':
            from v1_lora import _get_parent
            for (i, path), target in model._targets.items():
                parent, name = _get_parent(base.model.layers[i], path)
                setattr(parent, name, target.base)
        base.zero_grad(set_to_none=True)


@pytest.mark.parametrize('arm', ['SEG-00', 'SEG-01', 'SEG-03', 'SEG-04', 'SEG-06', 'SEG-07'])
def test_no_channel_with_and_without_discarded_first_is_bitwise_identical(base, arm):
    a, b = ids()
    with model_for(base, arm) as model, torch.no_grad():
        model.eval()
        if isinstance(model, BranchModel): nonzero(model)
        elif arm == 'SEG-04':
            for target in model._targets.values(): target.B.normal_(std=.03)
        direct, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b))
        discarded, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), force_first=True)
        assert torch.equal(direct.logits, discarded.logits)


@pytest.mark.parametrize('arm', ('SEG-02', 'SEG-05'))  # E-0025 R_keep arms; E-0032 in test_seg_w.py
def test_zero_initialization_off_reset_self_donor_and_no_backbone_KV(base, arm, record_property):
    a, b = ids()
    with torch.no_grad(): expected = base(b, use_cache=False).logits
    with model_for(base, arm) as model, torch.no_grad():
        model.eval()
        initial, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b))
        assert torch.equal(initial.logits, expected)
        nonzero(model)
        observed = []
        def capture(module, args, kwargs):
            observed.append({k: v.clone() if torch.is_tensor(v) else v for k, v in kwargs.items()
                             if k in ('attention_mask', 'position_ids', 'past_key_values', 'use_cache')})
        h = model.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            normal, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b))
        finally: h.remove()
        assert len(observed) == 2
        for obs, seq in zip(observed, (a, b)):
            assert obs['past_key_values'] is None and obs['use_cache'] is False
            assert torch.equal(obs['position_ids'], torch.arange(seq.shape[1], device='cuda')[None])
            assert obs['attention_mask'].shape == seq.shape
        off, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), kind='iv_off')
        assert torch.equal(off.logits, expected)
        reset, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), kind='iv_reset')
        assert torch.equal(reset.logits, model(b, use_cache=False).logits)
        first = first_forward(model, a, torch.ones_like(a))
        donor = first.cache.memory
        before = {i: (s.M.clone(), s.c.clone(), s.counts.clone()) for i, s in donor.items()}
        swapped, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), kind='iv_swap', donor_memory=donor)
        assert torch.equal(swapped.logits, normal.logits)
        for i, s in donor.items():
            assert all(torch.equal(x, y) for x, y in zip((s.M, s.c, s.counts), before[i]))
            assert int(s.counts[0]) == a.shape[1]
            assert int(normal.cache.memory[i].counts[0]) == a.shape[1]+b.shape[1]
        desc = state_description(donor)
        expected_bytes = 264192 if arm == 'SEG-05' else 16896
        assert sum(s['M_c_bytes'] for s in desc.values()) == expected_bytes
        assert sum(s['counts_bytes'] for s in desc.values()) == 32
        record_property(arm+'.state_M_c_bytes', expected_bytes)
        repeated, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b))
        assert torch.equal(repeated.logits, normal.logits)


def test_training_graph_reaches_first_segment_all_four_layers_and_frozen_backbone(base, record_property):
    a, b = ids()
    with model_for(base, 'SEG-02') as model:
        model.train()
        optimizer = torch.optim.SGD(model.adapter_parameters(), lr=.1)
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            first = first_forward(model, a, torch.ones_like(a))
            for state in first.cache.memory.values():
                state.M.retain_grad(); state.c.retain_grad()
                assert state.M.grad_fn is not None and state.c.grad_fn is not None
            second = model(b, attention_mask=torch.ones_like(b), initial_memory=first.cache.memory, use_cache=False)
            F.cross_entropy(second.logits[:, -2].float(), b[:, -1]).backward()
            for i, layer in model.adapters().items():
                state = first.cache.memory[i]
                assert state.M.grad is not None and torch.isfinite(state.M.grad).all()
                if step == 2:
                    assert state.M.grad.norm() > 0
                    assert state.c.grad is not None and state.c.grad.norm() > 0
                for name, p in layer.side.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    assert p.grad.norm() > 0 if step == 2 or name.startswith('fusion.') else p.grad.norm() == 0
                record_property(f'layer{i}.step{step}.first_M_grad', state.M.grad.norm().item())
            assert all(p.grad is None for n, p in model.named_parameters() if '.side.' not in n)
            optimizer.step()


def test_pairing_with_RB1_initialization_and_r128_parameter_counts(base):
    from rb_train import make_model as rb_make
    with model_for(base, 'SEG-01') as one:
        initial = [p.detach().clone() for p in one.adapter_parameters()]
    for arm in ('SEG-02', 'SEG-03'):
        with model_for(base, arm) as model:
            assert all(torch.equal(a, b) for a, b in zip(initial, model.adapter_parameters()))
    with patch('rb_train.load_base', return_value=base): rb = rb_make('RB1-01')
    try:
        assert all(torch.equal(a, b) for a, b in zip(initial, rb.adapter_parameters()))
    finally:
        for i, layer in rb.adapters().items():
            for h in layer.handles: h.remove()
            base.model.layers[i] = layer.original
    for arm in ('SEG-05', 'SEG-06', 'SEG-07'):
        with model_for(base, arm) as model:
            assert sum(p.numel() for p in model.adapter_parameters()) == PARAMETERS[arm]


def test_initial_memory_rejects_KV_or_foreign_modes_and_default_path_stays_identical(base):
    a, b = ids()
    with model_for(base, 'SEG-02') as model, torch.no_grad():
        states = first_forward(model, a, torch.ones_like(a)).cache.memory
        for kwargs in ({'use_cache': True}, {'phase': 'query'}, {'memory': states}, {'capture_history': True}):
            with pytest.raises(ValueError): model(b, initial_memory=states, **kwargs)
        with pytest.raises(ValueError): model(b, initial_memory={})
        assert torch.equal(model(b).logits, model(b, initial_memory=None).logits)
