import pytest
import torch

from bypass_core import (
    LatentReader,
    MemoryState,
    elu_plus_one,
    process_sequence,
    rmsnorm_no_affine,
)


def _record_error(record_property, name, actual, expected):
    delta = (actual - expected).abs()
    record_property(f"{name}.max_abs_err", float(delta.max().item()))
    relative = delta / expected.abs().clamp_min(1e-8)
    record_property(f"{name}.max_rel_err", float(relative.max().item()))


@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_kernel_matrix_equals_slot_reference_outputs_and_qkv_gradients(rank, steps, record_property):
    torch.manual_seed(11 + rank)
    batch, slots = 3, 5
    q0 = torch.randn(batch, rank)
    k0 = torch.rand(batch, slots, rank) + 0.1
    v0 = torch.randn(batch, slots, rank)
    probe = torch.randn(batch, rank)
    reader = LatentReader(rank)
    outputs, grads = [], []
    for backend in ("kernel_matrix", "kernel_slot_reference"):
        q = q0.clone().requires_grad_()
        k = k0.clone().requires_grad_()
        v = v0.clone().requires_grad_()
        state = MemoryState.empty(batch, rank, backend)
        for index in range(slots):
            state = state.write(k[:, index], v[:, index])
        out = reader(q, state, steps=steps)
        loss = (out * probe).sum()
        grad = torch.autograd.grad(loss, (q, k, v))
        outputs.append(out.detach())
        grads.append(tuple(g.detach() for g in grad))
    _record_error(record_property, "kernel_output", outputs[0], outputs[1])
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-5, rtol=2e-5)
    for name, a, b in zip(("q_grad", "k_grad", "v_grad"), grads[0], grads[1]):
        _record_error(record_property, f"kernel_{name}", a, b)
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_reader_fixed_query_has_exact_linear_magnitude(rank, steps, record_property):
    torch.manual_seed(4)
    state = MemoryState.empty(2, rank)
    state = state.write(torch.rand(2, rank) + 0.1, torch.randn(2, rank))
    reader = LatentReader(rank)
    qbar = torch.randn(2, rank)
    with torch.no_grad():
        fixed_q = elu_plus_one(rmsnorm_no_affine(qbar) @ reader.Rq)
        a = state.read(fixed_q)
        expected = steps * reader.eta * reader.read_gain * (a @ reader.Ro)
        actual = reader(qbar, state, steps=steps, fixed_query=True)
        full = reader(qbar, state, steps=steps, fixed_query=True, readout="full")
    _record_error(record_property, "fixed_query_increment", actual, expected)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(full, qbar + expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", ["kernel_matrix", "kernel_slot_reference", "soft_slot"])
def test_state_reset_padding_counts_and_slot_ownership(backend):
    rank = 32
    state = MemoryState.empty(2, rank, backend)
    k, v = torch.rand(2, rank), torch.randn(2, rank)
    state = state.write(k, v, torch.tensor([True, False]))
    assert state.counts.tolist() == [1, 0]
    if backend == "kernel_matrix":
        assert state.slot_keys is None
    else:
        assert state.slot_keys is not None
        assert state.slot_valid.tolist() == [[True], [False]]

    beams = state.reorder(torch.tensor([0, 0, 1]))
    updated = beams.write(torch.ones(3, rank), torch.ones(3, rank),
                          torch.tensor([True, False, False]))
    assert updated.counts.tolist() == [2, 1, 0]
    assert beams.counts.tolist() == [1, 1, 0]
    assert state.counts.tolist() == [1, 0]
    cleared = updated.reset(torch.tensor([False, True, False]))
    assert cleared.counts.tolist() == [2, 0, 0]
    if cleared.slot_valid is not None:
        assert not cleared.slot_valid[1].any()
        assert not cleared.slot_keys[1].any()
        assert not cleared.slot_values[1].any()
        fully_cleared = updated.reset()
        assert fully_cleared.slot_count == 0
        assert fully_cleared.slot_valid.shape == (3, 0)
        assert not fully_cleared.counts.any()
        assert not fully_cleared.M.any()
        assert not fully_cleared.c.any()


@pytest.mark.parametrize("mode", ["inclusive", "strict"])
def test_sequence_padding_causality_and_accounting(mode):
    torch.manual_seed(9)
    batch, time, rank = 2, 4, 32
    q, k, v = (torch.randn(batch, time, rank) for _ in range(3))
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    reset = torch.zeros_like(valid)
    reset[1, 2] = True
    writer = valid.clone()
    reader = LatentReader(rank)
    result = process_sequence(q, k, v, reader, valid=valid, reset=reset,
                              write=writer, steps=2, mode=mode)
    assert result.accounting == {"valid_tokens": 6, "writes": 6, "reads": 12, "reader_steps": 2}
    assert result.state.counts.tolist() == [3, 2]
    assert torch.equal(result.outputs[~valid], torch.zeros_like(result.outputs[~valid]))

    changed = v.clone()
    changed[:, 3] += 1000
    rerun = process_sequence(q, k, changed, reader, valid=valid, reset=reset,
                             write=writer, steps=2, mode=mode)
    torch.testing.assert_close(result.outputs[:, :3], rerun.outputs[:, :3], atol=2e-5, rtol=2e-5)


def test_strict_current_token_cannot_read_its_own_value():
    rank = 32
    q = torch.randn(1, 1, rank)
    k = torch.randn(1, 1, rank)
    v1 = torch.randn(1, 1, rank)
    v2 = v1 + 20
    reader = LatentReader(rank)
    strict1 = process_sequence(q, k, v1, reader, mode="strict")
    strict2 = process_sequence(q, k, v2, reader, mode="strict")
    inclusive1 = process_sequence(q, k, v1, reader, mode="inclusive")
    inclusive2 = process_sequence(q, k, v2, reader, mode="inclusive")
    torch.testing.assert_close(strict1.outputs, strict2.outputs, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(inclusive1.outputs, inclusive2.outputs, atol=2e-5, rtol=2e-5)


def test_write_only_then_read_only_memory_is_unchanged():
    rank = 32
    reader = LatentReader(rank)
    q = torch.randn(1, 2, rank)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for backend in ("kernel_matrix", "kernel_slot_reference", "soft_slot"):
        support = MemoryState.empty(1, rank, backend).write(
            torch.rand(1, rank), torch.randn(1, rank))
        slot_count = support.slot_count
        result = process_sequence(q, k, v, reader, state=support,
                                  write=torch.zeros(1, 2, dtype=torch.bool), steps=2)
        assert result.state.counts.tolist() == [1]
        torch.testing.assert_close(result.state.M, support.M)
        assert result.state.slot_count == slot_count
        assert result.accounting["writes"] == 0
        assert result.accounting["reads"] == 4

        padded = process_sequence(q, k, v, reader, state=support,
                                  valid=torch.zeros(1, 2, dtype=torch.bool), steps=2)
        assert padded.state.slot_count == slot_count
        assert padded.accounting["writes"] == 0
        assert padded.accounting["reads"] == 0
        torch.testing.assert_close(padded.state.M, support.M)

        support_result = process_sequence(
            q, k, v, reader, state=MemoryState.empty(1, rank, backend),
            write=torch.ones(1, 2, dtype=torch.bool),
            read=torch.zeros(1, 2, dtype=torch.bool), steps=2)
        assert support_result.state.counts.tolist() == [2]
        assert support_result.accounting["writes"] == 2
        assert support_result.accounting["reads"] == 0
        assert support_result.state.slot_count == (None if backend == "kernel_matrix" else 2)
        torch.testing.assert_close(support_result.outputs, torch.zeros_like(support_result.outputs))


def test_negative_controls_detect_future_leak_repeated_write_and_padding_write():
    # Future leak: an intentionally broken all-sequence aggregate changes the prefix.
    k = torch.tensor([[[1.0], [2.0]]])
    v = torch.tensor([[[3.0], [4.0]]])
    future_changed = v.clone()
    future_changed[:, 1] = 40
    broken_prefix = (k * v).sum(dim=1)[:, 0]
    broken_changed_prefix = (k * future_changed).sum(dim=1)[:, 0]
    with pytest.raises(AssertionError):
        torch.testing.assert_close(broken_prefix, broken_changed_prefix)

    # Repeated write: an intentionally broken S-step writer accumulates S copies.
    one = MemoryState.empty(1, 1).write(torch.ones(1, 1), torch.ones(1, 1))
    repeated = one.write(torch.ones(1, 1), torch.ones(1, 1)).write(
        torch.ones(1, 1), torch.ones(1, 1))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(one.M, repeated.M)

    # Padding write: intentionally ignore valid=False; correct and broken states differ.
    correct = MemoryState.empty(1, 1).write(torch.ones(1, 1), torch.ones(1, 1),
                                            torch.tensor([False]))
    broken = MemoryState.empty(1, 1).write(torch.ones(1, 1), torch.ones(1, 1))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(correct.M, broken.M)
