"""Real Qwen3 LoRA checks; run inside the registered GPU job."""
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from calibrate import fixture, load_base
from v1_lora import LoRAConfig, LoRAModel, TARGETS


@pytest.fixture(scope='module')
def base():
    model = load_base()
    assert model.config.hidden_size == 1024 and len(model.model.layers) == 28
    assert model.config.num_attention_heads == 16 and model.config.num_key_value_heads == 8
    yield model


@contextmanager
def adapted(base, config):
    wrapped = LoRAModel(base, config)
    try:
        yield wrapped
    finally:
        for (index, path), adapter in wrapped._targets.items():
            parent = base.model.layers[index]
            parts = path.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], adapter.base)
        base.zero_grad(set_to_none=True)


def test_parameter_counts_and_initialization(base):
    assert LoRAConfig().alpha == LoRAConfig().rank == 10
    with adapted(base, LoRAConfig()) as model:
        params = list(model.adapter_parameters())
        assert len(params) == 4 * len(TARGETS) * 2
        assert sum(p.numel() for p in params) == 901120
        assert all(p.dtype == torch.float32 and p.requires_grad for p in params)
        assert all(not p.requires_grad for name, p in base.named_parameters()
                   if not name.endswith(('.A', '.B')))
        assert all(torch.count_nonzero(adapter.B) == 0 for adapter in model._targets.values())
    with adapted(base, LoRAConfig(layers=tuple(range(28)), rank=2)) as model:
        assert model.config.alpha == 2
        assert sum(p.numel() for p in model.adapter_parameters()) == 1261568


def test_zero_exit_gradients_and_checkpoint_roundtrip(base, tmp_path):
    ids = fixture()
    frozen_params = tuple(base.parameters())
    with torch.no_grad():
        reference = base(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    config = LoRAConfig()
    with adapted(base, config) as model:
        with torch.no_grad():
            initial = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        torch.testing.assert_close(initial, reference, rtol=0, atol=0)

        optimizer = torch.optim.SGD(model.adapter_parameters(), lr=.01)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   ids[:, 1:].reshape(-1))
            loss.backward()
            assert all(p.grad is None for p in frozen_params)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.adapter_parameters())
            if step == 0:
                assert all(adapter.A.grad.count_nonzero() == 0
                           for adapter in model._targets.values())
                assert any(adapter.B.grad.norm() > 0 for adapter in model._targets.values())
            else:
                assert any(adapter.A.grad.norm() > 0 for adapter in model._targets.values())
            optimizer.step()

        with torch.no_grad():
            expected = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        model.export_adapter(tmp_path)

    reloaded_base = load_base()
    reloaded = LoRAModel.load_adapter(reloaded_base, tmp_path)
    assert reloaded.config == config
    with torch.no_grad():
        actual = reloaded(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in reloaded.adapter_parameters())
    assert all(not p.requires_grad for p in reloaded_base.parameters()
               if not any(p is a for a in reloaded.adapter_parameters()))
