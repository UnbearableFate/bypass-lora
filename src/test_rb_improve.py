"""E-0029 rule 1 regressions; synthetic and real-operator checks run in PBS."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from bypass_core import LatentReader, elu_plus_one, process_sequence, rmsnorm_no_affine
from bypass_model import BranchModel
from rb_improve_init import (GradientCollector, basis_orthogonality_error, full_value_prefix, grouped_query_projection,
                             mixed_gradient, pair_qk, scaled_subspace)
from rb_improve_read import read_statistics
from rb_improve_train import backbone_only, init_check
from rb_tap_check import initial_state
from rb_train import E0029_ARMS, PARAMETERS, adapter_config
from v1_lora import LoRAConfig, UpLoRAModel


def tiny_base():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=32, attention_dropout=0.)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config).eval()


def wrapped(arm):
    base = tiny_base()
    torch.manual_seed(19)
    return BranchModel(base, replace(adapter_config(arm), layers=(0, 1), rank=4))


def test_normalized_qk_tap_captures_exact_layernorm_and_v_projection():
    model = wrapped('Q-N')
    captured, handles, checks = {}, [], []
    for index, layer in model.adapters().items():
        def n_hook(module, args, output, index=index):
            captured[index, 'n'] = output
        def v_hook(module, args, output, index=index):
            captured[index, 'v'] = output
        def projection(module, args, index=index):
            checks.append(torch.equal(args[0], captured[index, 'n']) and
                          torch.equal(args[1], captured[index, 'n']) and
                          torch.equal(args[2], captured[index, 'v']))
        handles.extend([layer.original.input_layernorm.register_forward_hook(n_hook),
            layer.original.self_attn.v_proj.register_forward_hook(v_hook),
            layer.side.projection.register_forward_pre_hook(projection)])
    try:
        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert checks == [True, True]
    assert initial_state(wrapped('Q-N'))[0] == initial_state(wrapped('X-01'))[0]


@pytest.mark.parametrize('arm', ['P-T', 'P-N'])
def test_pair_changes_only_Aq_and_keeps_independent_parameters(arm):
    model = wrapped(arm)
    _, reference = initial_state(model)
    pair_qk(model)
    assert init_check(model, arm, reference)['passed']
    for layer in model.adapters().values():
        projection = layer.side.projection
        expected = projection.A_k if arm == 'P-N' else grouped_query_projection(projection.A_k, 2, 1, 16)
        assert torch.equal(projection.A_q, expected)
        assert projection.A_q is not projection.A_k
        before = projection.A_k.detach().clone()
        with torch.no_grad():
            projection.A_q.add_(1)
        assert torch.equal(projection.A_k, before)


def test_gqa_head_mapping_has_two_adjacent_query_heads_per_kv_head():
    A_k = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    actual = grouped_query_projection(A_k, 4, 2, 2)
    expected = torch.tensor([[1., 2.], [3., 4.], [1., 2.], [3., 4.],
                             [5., 6.], [7., 8.], [5., 6.], [7., 8.]]) / 2 ** .5
    assert torch.equal(actual, expected)
    q = torch.arange(8.).reshape(1, -1)
    grouped = (q.reshape(1, 2, 2, 2).sum(2) / 2 ** .5).reshape(1, 4)
    torch.testing.assert_close(q @ actual, grouped @ A_k)


@pytest.mark.parametrize('chunk', [1, 3, 64])
def test_full_value_prefix_matches_explicit_causal_weights_and_latent_reader(chunk):
    torch.manual_seed(61)
    q, k, values = torch.randn(2, 7, 4), torch.randn(2, 7, 4), torch.randn(2, 7, 6)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1, 0]], dtype=torch.bool)
    reader = LatentReader(4)
    reader.Rq.data.normal_()
    p = full_value_prefix(q, k, values, reader, valid, chunk)
    rho = elu_plus_one(rmsnorm_no_affine(q) @ reader.Rq)
    keys = elu_plus_one(rmsnorm_no_affine(k))
    expected = torch.zeros_like(p)
    for b in range(2):
        for t in range(7):
            if valid[b, t]:
                weights = rho[b, t] @ keys[b, :t+1].T * valid[b, :t+1]
                expected[b, t] = weights @ values[b, :t+1] / (weights.sum() + reader.eps_read)
    torch.testing.assert_close(p, expected, rtol=2e-6, atol=2e-6)
    A_v = torch.randn(6, 4)
    actual = process_sequence(q, k, values @ A_v, reader, valid=valid, impl='prefix').outputs
    torch.testing.assert_close(actual, reader.eta * (p @ A_v @ reader.Ro), rtol=3e-5, atol=5e-7)


def test_mixed_gradient_includes_indirect_prompt_positions_and_excludes_padding():
    p = torch.tensor([[[1., 2.], [3., 4.], [900., 800.]]])
    d = torch.tensor([[[5., 6.], [7., 8.], [700., 600.]]])
    valid = torch.tensor([[True, True, False]])
    B = torch.zeros(2, 2, requires_grad=True)
    ((p @ B) * d * valid[..., None]).sum().backward()
    assert torch.equal(mixed_gradient(p, d, valid), B.grad)


@pytest.mark.parametrize('arm', ['G-T', 'G-QN', 'G-N'])
def test_collected_gradient_matches_virtual_full_rank_autograd_per_example(arm):
    model = wrapped(arm)
    collector = GradientCollector(model)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    collector.valid = valid
    # Only the last position is directly supervised; earlier positions affect it through attention.
    F.cross_entropy(model(input_ids=ids, use_cache=False).logits[:, -2].float(), ids[:, -1]).div(2).backward()
    actual = {index: stats['G'].clone() for index, stats in collector.stats.items()}
    collector.close()
    model.zero_grad(set_to_none=True)
    matrices, pending, handles = {}, {}, []
    for index, layer in model.adapters().items():
        projection = layer.side.projection
        matrices[index] = torch.zeros(projection.A_v.shape[0], 24, requires_grad=True)
        def capture(module, args, output, index=index, reader=layer.side.reader):
            pending[index] = full_value_prefix(output[0], output[1], args[2], reader, valid)
        def dense(module, args, output, index=index):
            return output + pending[index] @ matrices[index]
        handles.extend([projection.register_forward_hook(capture),
                        layer.original.mlp.up_proj.register_forward_hook(dense)])
    try:
        F.cross_entropy(model(input_ids=ids, use_cache=False).logits[:, -2].float(), ids[:, -1]).div(2).backward()
    finally:
        for handle in handles:
            handle.remove()
    for index in matrices:
        torch.testing.assert_close(actual[index], matrices[index].grad, rtol=1e-5, atol=1e-7)
        assert torch.isfinite(actual[index]).all() and actual[index].norm() > 0
    assert all(p.grad is None for name, p in model.named_parameters() if not p.requires_grad)


def test_svd_basis_is_orthogonal_and_scaled_readout_matches_random_RMS():
    torch.manual_seed(31)
    p, d = torch.randn(29, 6), torch.randn(29, 8)
    reader, rank = LatentReader(3), 3
    random_A = torch.randn(6, rank)
    random_Z = reader.eta * p @ random_A
    value, info = scaled_subspace(p.T @ d, p.T @ p, random_Z.square().sum().item(), reader, rank)
    basis = value / info['scale']
    torch.testing.assert_close(basis.T @ basis, torch.eye(rank), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close((reader.eta * p @ value).square().mean(), random_Z.square().mean(), rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA precision regression')
def test_orthogonality_check_measures_stored_columns_even_when_TF32_is_enabled():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        basis = torch.linalg.qr(torch.randn(1024, 32, device='cuda')).Q
        assert basis_orthogonality_error(basis) < 1e-5
        basis[:, 0].mul_(1.001)
        # The unchanged1e-4 gate must still reject an actually non-unit column.
        assert basis_orthogonality_error(basis) > 1e-3
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize('rank,expected', [(56, 917504), (48, 786432)])
def test_up_only_lora_identity_target_counts_and_checkpoint_roundtrip(rank, expected, tmp_path):
    assert 4 * rank * (1024 + 3072) == expected == PARAMETERS[f'L-{rank}']
    base = tiny_base()
    ids = torch.tensor([[1, 2, 3, 4]])
    reference = base(ids, use_cache=False).logits.detach()
    model = UpLoRAModel(base, LoRAConfig(layers=(0, 1), rank=rank, alpha=rank))
    assert set(model._targets) == {(0, 'mlp.up_proj'), (1, 'mlp.up_proj')}
    torch.testing.assert_close(model(ids, use_cache=False).logits, reference, rtol=0, atol=0)
    with backbone_only(model) as original:
        torch.testing.assert_close(original(ids, use_cache=False).logits, reference, rtol=0, atol=0)
    for adapter in model._targets.values():
        assert adapter.scale == 1 and adapter.B.count_nonzero() == 0
        adapter.B.data.normal_(std=.01)
    expected_logits = model(ids, use_cache=False).logits.detach()
    model.export_adapter(tmp_path)
    loaded = UpLoRAModel.load_adapter(tiny_base(), tmp_path)
    torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected_logits, rtol=0, atol=0)


def test_read_weights_are_normalized_and_prefix_bins_count_valid_tokens():
    reader = LatentReader(3)
    q = torch.zeros(1, 4, 3)
    valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    stats = read_statistics(q, q, reader, valid)
    torch.testing.assert_close(stats['self_weight'][0, :3], torch.tensor([1., .5, 1/3]))
    torch.testing.assert_close(stats['maximum_weight'][0, :3], torch.tensor([1., .5, 1/3]))
    torch.testing.assert_close(stats['n_eff'][0, :3], torch.tensor([1., 2., 3.]))


def test_arm_registry_has_only_the_prespecified_eight_variants():
    assert E0029_ARMS == ('G-T', 'G-QN', 'G-N', 'Q-N', 'P-T', 'P-N', 'L-56', 'L-48')
    assert adapter_config('Q-N').tap == adapter_config('G-QN').tap == 'normalized_qk'
    for arm in E0029_ARMS[:6]:
        cfg = adapter_config(arm)
        assert cfg.layers == (6, 13, 20, 27) and cfg.steps == 1 and cfg.rank == 32
        assert cfg.fusion == 'C_up' and cfg.eta == .25 and cfg.mode == 'inclusive'
