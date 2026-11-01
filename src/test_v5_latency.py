from types import SimpleNamespace

import torch
from torch import nn

from v1_lora import LoRALinear
from v5_latency import merge_lora


def test_merge_lora_updates_weight_once_and_removes_wrapper():
    original = nn.Linear(3,2,bias=False)
    adapter = LoRALinear(original,rank=1,alpha=2)
    with torch.no_grad():
        adapter.A.copy_(torch.tensor([[1.,2.,3.]]))
        adapter.B.copy_(torch.tensor([[.1],[.2]]))
    inputs = torch.tensor([[.5,-1.,.2]])
    expected = adapter(inputs)
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.up_proj = adapter
    base = nn.Module()
    base.model = nn.Module()
    base.model.layers = nn.ModuleList([layer])
    model = SimpleNamespace(model=base,_targets={(0,'mlp.up_proj'):adapter})
    merged = merge_lora(model)
    assert merged is base
    assert layer.mlp.up_proj is original
    torch.testing.assert_close(original(inputs),expected,rtol=2e-5,atol=2e-5)
    assert not any(isinstance(m,LoRALinear) for m in merged.modules())
