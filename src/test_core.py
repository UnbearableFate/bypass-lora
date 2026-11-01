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
@pytest.mark.parametrize("slots", [1, 5, 17])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_kernel_matrix_equals_slot_reference_outputs_and_qkv_gradients(
        rank, slots, steps, record_property):
    torch.manual_seed(11 + rank + slots)
    batch = 3
    q0 = torch.randn(batch, rank)
    k0 = torch.randn(batch, slots, rank)
    v0 = torch.randn(batch, slots, rank)
    write_mask = torch.rand(batch, slots) > 0.35
    write_mask[0] = False  # Include an empty-memory row.
    if slots > 1:
        write_mask[1, 0] = True
        write_mask[1, -1] = True
    probe = torch.randn(batch, rank)
    reader = LatentReader(rank)
    outputs, grads = [], []
    for backend in ("kernel_matrix", "kernel_slot_reference"):
        q = q0.clone().requires_grad_()
        k = k0.clone().requires_grad_()
        v = v0.clone().requires_grad_()
        state = MemoryState.empty(batch, rank, backend)
        for index in range(slots):
            if slots > 1 and index == slots // 2:
                reset_rows = torch.tensor([False, True, False])
                state = state.reset(reset_rows)
            state = state.write(k[:, index], v[:, index], write_mask[:, index])
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


def _assert_prefix_invariant(run, values):
    changed = values.clone()
    changed[:, 3] += 1000
    torch.testing.assert_close(run(values)[:, :3], run(changed)[:, :3], atol=2e-5, rtol=2e-5)


def _assert_write_contract(result, valid):
    per_row = valid.sum(dim=1)
    assert result.accounting["writes"] == int(per_row.sum().item())
    assert torch.equal(result.state.counts, per_row)


def _assert_read_contract(result, expected):
    assert result.accounting["reads"] == expected


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


def test_negative_control_full_sequence_future_leak_uses_prefix_assertion():
    torch.manual_seed(39)
    q, k, v = (torch.randn(2, 4, 32) for _ in range(3))
    valid = torch.ones(2, 4, dtype=torch.bool)
    reader = LatentReader(32)

    def causal_run(values):
        return process_sequence(q, k, values, reader, valid=valid).outputs

    _assert_prefix_invariant(causal_run, v)

    def full_sequence_leak(values):
        support = process_sequence(q, k, values, reader, valid=valid,
                                   read=torch.zeros_like(valid))
        query = process_sequence(q, k, values, reader, state=support.state,
                                 valid=valid, write=torch.zeros_like(valid))
        return query.outputs

    with pytest.raises(AssertionError):
        _assert_prefix_invariant(full_sequence_leak, v)


@pytest.mark.parametrize("steps", [2, 4])
def test_negative_control_repeated_write_uses_write_count_assertion(monkeypatch, steps):
    torch.manual_seed(2)
    q, k, v = (torch.randn(2, 4, 32) for _ in range(3))
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    original_write = MemoryState.write

    def run():
        return process_sequence(q, k, v, LatentReader(32), valid=valid, steps=steps)

    _assert_write_contract(run(), valid)

    def repeated_write(self, key, value, write_valid=None):
        for _ in range(steps):
            self = original_write(self, key, value, write_valid)
        return self

    monkeypatch.setattr(MemoryState, "write", repeated_write)
    with pytest.raises(AssertionError):
        _assert_write_contract(run(), valid)


def test_negative_control_padding_write_uses_write_count_assertion(monkeypatch):
    torch.manual_seed(3)
    q, k, v = (torch.randn(2, 4, 32) for _ in range(3))
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    original_write = MemoryState.write

    def run():
        return process_sequence(q, k, v, LatentReader(32), valid=valid)

    _assert_write_contract(run(), valid)
    monkeypatch.setattr(MemoryState, "write",
                        lambda self, key, value, write_valid=None:
                        original_write(self, key, value, None))
    with pytest.raises(AssertionError):
        _assert_write_contract(run(), valid)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_read_accounting_uses_performed_reader_calls(monkeypatch, steps):
    torch.manual_seed(19)
    q, k, v = (torch.randn(2, 4, 32) for _ in range(3))
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    reader = LatentReader(32)
    result = process_sequence(q, k, v, reader, valid=valid, steps=steps)
    _assert_read_contract(result, 6 * steps)
    assert reader.last_read_calls == steps

    original_forward = LatentReader.forward

    def skip_step(self, *args, **kwargs):
        if len(args) >= 3:
            args = (args[0], args[1], max(0, args[2] - 1), *args[3:])
        else:
            kwargs["steps"] = max(0, kwargs.get("steps", 1) - 1)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(LatentReader, "forward", skip_step)
    skipped = process_sequence(q, k, v, reader, valid=valid, steps=steps)
    with pytest.raises(AssertionError):
        _assert_read_contract(skipped, 6 * steps)
    assert skipped.accounting["reads"] == 6 * max(0, steps - 1)


@pytest.mark.parametrize("mode", ["inclusive", "strict"])
def test_key_normalization_uses_reader_epsilon(mode):
    rank = 32
    q = torch.zeros(1, 1, rank)
    k = torch.full_like(q, 1e-3)
    v = torch.ones_like(q)
    default = process_sequence(q, k, v, LatentReader(rank), mode=mode)
    configured = process_sequence(q, k, v, LatentReader(rank, eps_norm=1e-2), mode=mode)
    assert not torch.allclose(default.state.M, configured.state.M, atol=2e-5, rtol=2e-5)


def test_kernel_identity_negative_control():
    torch.manual_seed(55)
    state = MemoryState.empty(2,32)
    for _ in range(4):
        state = state.write(torch.rand(2,32)+.1,torch.randn(2,32))
    q = torch.rand(2,32)+.1
    correct = state.read(q)
    # Deliberately drop normalization; the identity assertion must reject it.
    broken = torch.einsum('br,brv->bv',q,state.M)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(broken,correct,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('steps',[1,2,4])
def test_depth_does_not_multiply_write_count(steps):
    torch.manual_seed(2)
    q,k,v = (torch.randn(2,4,32) for _ in range(3))
    valid = torch.tensor([[True,True,False,True],[True,False,True,True]])
    result = process_sequence(q,k,v,LatentReader(32),valid=valid,steps=steps)
    _assert_write_contract(result, valid)
    _assert_read_contract(result, 6*steps)
