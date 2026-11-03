"""Consequential E-0031 topology, gradient, state and boundary regressions (PBS)."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from bypass_model import AdapterConfig, BranchModel
from rb1_intervene import original_backbone
from rb_tap_check import initial_state
from rb_train import E0031_ARMS, adapter_config


def tiny(arm):
    torch.manual_seed(239)
    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=32, attention_dropout=0.)
    cfg._attn_implementation = 'eager'
    base = Qwen3ForCausalLM(cfg).eval()
    torch.manual_seed(729)
    return BranchModel(base, replace(adapter_config(arm), layers=(0, 1), rank=4))


@pytest.mark.parametrize('history,tokenwise', [('H-QVU','T0-QVU'), ('U-25','T0-U-25')])
def test_initial_parameters_are_elementwise_paired(history, tokenwise):
    left, right = initial_state(tiny(history)), initial_state(tiny(tokenwise))
    assert left[0] == right[0]
    assert left[1].keys() == right[1].keys()
    assert all(torch.equal(v, right[1][k]) for k, v in left[1].items())


@pytest.mark.parametrize('arm', E0031_ARMS)
def test_zero_exits_gradients_frozen_backbone_and_off(arm):
    model = tiny(arm)
    ids = torch.tensor([[1, 5, 9, 13, 18]])
    original = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    with torch.no_grad(), original_backbone(model) as base:
        expected = base(ids, use_cache=False).logits
    model.eval()
    assert torch.equal(model(ids).logits, expected)
    optimizer = torch.optim.SGD(model.adapter_parameters(), lr=.1)
    for step in (1, 2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids).logits
        F.cross_entropy(logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1)).backward()
        for layer in model.adapters().values():
            for name, p in layer.side.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert p.grad.norm() > 0 if step == 2 or name.startswith('fusion.') else p.grad.norm() == 0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
    assert all(torch.equal(p, original[n]) for n,p in model.named_parameters() if not p.requires_grad)
    with torch.no_grad():
        model.eval()
        assert torch.equal(model(ids, intervention={'kind':'iv_off'}).logits, expected)


def test_same_early_read_is_written_before_q_norm_and_ffn_multiplication():
    model = tiny('H-QVU').eval()
    ids = torch.tensor([[1, 3, 11, 6]])
    layer = model.adapters()[0]
    with torch.no_grad():
        for adapted in model.adapters().values():
            for p in adapted.side.fusion.parameters():
                p.normal_(std=.1)
    records, order, handles = {}, [], []
    def normalized(module, args, result):
        records['n'] = result
        order.append('n')
    def read(module, args, result):
        records['z'] = result[0]
        order.append('read')
    def qnorm(module, args):
        records['qnorm_input'] = args[0]
        order.append('qnorm')
    def mlp(module, args):
        records['u'] = args[0]
        order.append('ffn')
    def down(module, args):
        records['product'] = args[0]
    handles += [layer.original.input_layernorm.register_forward_hook(normalized),
        layer.side.register_forward_hook(read),
        layer.original.self_attn.q_norm.register_forward_pre_hook(qnorm),
        layer.original.mlp.register_forward_pre_hook(mlp),
        layer.original.mlp.down_proj.register_forward_pre_hook(down)]
    try:
        with torch.no_grad():
            output = model(ids)
    finally:
        for handle in handles:
            handle.remove()
    assert order == ['n', 'read', 'qnorm', 'ffn']
    attn, ff, z = layer.original.self_attn, layer.original.mlp, records['z']
    raw_q = F.linear(records['n'], attn.q_proj.weight)
    expected_q = (raw_q + z @ layer.side.fusion.C_q).view(1, 4, 2, 8)
    torch.testing.assert_close(records['qnorm_input'], expected_q, atol=0, rtol=0)
    expected_product = F.silu(F.linear(records['u'], ff.gate_proj.weight)) * (
        F.linear(records['u'], ff.up_proj.weight) + z @ layer.side.fusion.C_u)
    torch.testing.assert_close(records['product'], expected_product, atol=0, rtol=0)
    assert all(v['base'] == {'attention':1,'ffn':1} for v in output.accounting.values())
    # An independent projection probe checks the value hook before GQA expansion.
    with torch.no_grad(), layer.side.fusion.attention_writes(attn, z):
        torch.testing.assert_close(attn.v_proj(records['n']),
            F.linear(records['n'], attn.v_proj.weight) + z @ layer.side.fusion.C_v, atol=0, rtol=0)


@pytest.mark.parametrize('arm', E0031_ARMS)
def test_suffix_causality_padding_and_reload(arm, tmp_path):
    model = tiny(arm).eval()
    with torch.no_grad():
        for layer in model.adapters().values():
            for p in layer.side.fusion.parameters():
                p.normal_(std=.1)
    ids = torch.tensor([[1, 7, 3, 9, 10, 12]])
    changed = ids.clone(); changed[:, 4:] = torch.tensor([[20, 30]])
    with torch.no_grad():
        normal = model(ids).logits
        suffix = model(changed).logits
        torch.testing.assert_close(normal[:, :4], suffix[:, :4], atol=0, rtol=0)
        for mask in (torch.tensor([[0,0,1,1,1,1]]), torch.tensor([[1,1,1,1,0,0]])):
            padded = ids.clone(); padded[~mask.bool()] = 31
            records, handles = [], []
            for layer in model.adapters().values():
                handles.append(layer.side.register_forward_hook(lambda m,a,o: records.append(o[0])))
            try:
                first = model(ids, attention_mask=mask)
                second = model(padded, attention_mask=mask)
            finally:
                for handle in handles:
                    handle.remove()
            assert all(torch.count_nonzero(z[~mask.bool()]) == 0 for z in records)
            torch.testing.assert_close(first.logits[mask.bool()], second.logits[mask.bool()], atol=0, rtol=0)
            for index in model.config.layers:
                a, b = first.cache.memory[index], second.cache.memory[index]
                if a is not None:
                    assert torch.equal(a.M, b.M) and torch.equal(a.c, b.c)
                    assert torch.equal(a.counts, mask.sum(-1))
        model.export_adapter(tmp_path/'adapter')
        base = tiny(arm)
        with original_backbone(base) as backbone:
            loaded = BranchModel.load_adapter(backbone, tmp_path/'adapter').eval()
            torch.testing.assert_close(loaded(ids).logits, normal, atol=0, rtol=0)


def test_cyclic_or_nonmemory_qvu_configuration_is_rejected():
    with pytest.raises(ValueError, match='direct-hidden'):
        AdapterConfig(fusion='C_qvu')
    with pytest.raises(ValueError, match='direct-hidden'):
        AdapterConfig(fusion='C_qvu', tap='direct_hidden', branch='tokenwise_silu')
