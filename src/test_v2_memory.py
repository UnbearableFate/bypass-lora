"""M-line protocol regressions, run in the registered GPU allocation."""
from dataclasses import replace
import math

import pytest
import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION, load_base
from v2_memory import (collate, config, train_forward, build_support, evaluate_batch,
                       read_vectors, read_metrics, control_snapshot, Snapshot,
                       memory_from_slots, independent_probes)
from v2_run import fixture, adapted
from bypass_core import LatentReader


@pytest.fixture(scope='module')
def base():
    return load_base()


@pytest.fixture(scope='module')
def rows():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    return fixture(tok), tok.pad_token_id


@pytest.mark.parametrize('backend', ['kernel_matrix', 'soft_slot'])
def test_hand_computed_read_metrics(backend):
    keys = torch.tensor([[[1., 1.], [1., 1.]]])
    vals = torch.tensor([[[2., 0.], [0., 2.]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    state = memory_from_slots(keys, vals, valid, backend)
    snap = Snapshot({14: state}, keys, vals, valid)
    reader = LatentReader(2)
    target = torch.tensor([0])
    raw, w, scores, den = read_vectors(torch.zeros(1, 1, 2), snap, reader, backend, target)
    torch.testing.assert_close(raw, torch.ones(1, 1, 2), atol=1e-6, rtol=1e-6)
    metrics = read_metrics(raw, w, scores, den, vals[:, 0], target, torch.ones(1, 1, dtype=torch.bool), valid)
    assert metrics['address.top1'] == 1
    assert metrics['address.margin'] == 0
    assert metrics['address.target_weight'] == pytest.approx(.5, abs=1e-6)
    assert metrics['address.entropy'] == pytest.approx(math.log(2))
    assert metrics['address.n_eff'] == pytest.approx(2)
    assert metrics['read.normalized_error'] == pytest.approx(.5, abs=1e-6)
    assert metrics['read.direction_error'] == pytest.approx(1-1/math.sqrt(2), abs=1e-6)


@pytest.mark.parametrize('backend', ['kernel_matrix', 'soft_slot'])
def test_controls_and_oracles(backend):
    keys = torch.tensor([[[4., 1.], [1., 4.]], [[2., 1.], [1., 2.]]])
    vals = torch.tensor([[[2., 0.], [0., 3.]], [[5., 0.], [0., 7.]]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    state = memory_from_slots(keys, vals, valid, backend)
    snap = Snapshot({14: state}, keys, vals, valid)
    reader = LatentReader(2)
    q = torch.tensor([[[2., -2.]], [[2., -2.]]])
    target = torch.tensor([0, 0])
    raw, *_ = read_vectors(q, snap, reader, backend, target)
    for mode in ('zero', 'shuffled', 'permuted'):
        altered = control_snapshot(snap, mode)
        actual, *_ = read_vectors(q, altered, reader, backend, target)
        assert not torch.allclose(actual, raw)
        query = torch.nn.functional.elu(q[:, 0] * torch.rsqrt(q[:, 0].square().mean(-1, keepdim=True)+1e-6))+1
        torch.testing.assert_close(actual[:, 0], altered.memory[14].read(query))
    oracle, *_ = read_vectors(q, snap, reader, backend, target, 'oracle_value')
    torch.testing.assert_close(oracle[:, 0], vals[:, 0])
    oracle, *_ = read_vectors(q, snap, reader, backend, target, 'oracle_key')
    torch.testing.assert_close(oracle[:, 0], state.read(keys[:, 0]))
    torch.testing.assert_close(snap.values, vals)
    torch.testing.assert_close(snap.memory[14].M, state.M)


def test_boundary_and_owner_validation(rows):
    episodes, pad_id = rows
    batch = collate(episodes, pad_id, device='cpu', owners=[1, 0, 1])
    assert batch.owners.tolist() == [1, 0, 1]
    assert batch.write_mask.sum(-1).tolist() == [2, 2]
    bad = [{**episodes[0], 'facts': [episodes[0]['facts'][0]]*2}, episodes[1]]
    with pytest.raises(ValueError, match='distinct'):
        collate(bad, pad_id, device='cpu')
    with pytest.raises(ValueError, match='owner'):
        collate(episodes, pad_id, device='cpu', owners=[2])


@pytest.mark.parametrize('tap', ['direct_hidden', 'qknorm_pre_rope'])
@pytest.mark.parametrize('backend', ['kernel_matrix', 'soft_slot'])
@pytest.mark.parametrize('fusion', ['F_out', 'C_dual'])
def test_m_line_protocol(base, rows, tap, backend, fusion):
    episodes, pad_id = rows
    batch = collate(episodes, pad_id)
    with torch.no_grad():
        reference = base(batch.question_ids, attention_mask=batch.question_mask, use_cache=False).logits
    with adapted(base, config(tap, backend, fusion)) as model:
        model.eval()
        with torch.no_grad():
            _, logits, support, query = train_forward(model, batch)
            expected = reference[:, -1].float()  # fixture questions have equal length
            assert torch.equal(logits, expected)
            assert support[14]['side']['reads'] == 0
            assert support[14]['side']['writes'] == 4
            assert query[14]['side']['writes'] == 0
            assert query[14]['side']['reads'] == int(batch.question_mask.sum())
        optimizer = torch.optim.SGD(model.adapter_parameters(), lr=.01)
        model.train()
        for step in range(2):
            model.zero_grad(set_to_none=True)
            loss, _, _, _ = train_forward(model, batch)
            loss.backward()
            side = model.adapters()[14].side
            assert any(p.grad is not None and p.grad.norm() > 0 for p in side.fusion.parameters())
            if step == 0:
                assert all(p.grad is None or p.grad.norm() == 0 for p in side.projection.parameters())
            else:
                assert side.projection.A_v.grad.norm() > 0
            assert all(p.grad is None for n,p in model.model.named_parameters() if '.side.' not in n)
            optimizer.step()
        with pytest.raises(ValueError, match='eval'):
            evaluate_batch(model, batch, 'oracle_value')
        model.eval()
        with torch.no_grad():
            snapshot, _ = build_support(model, batch, observe=True)
            old = snapshot.memory[14].M.clone()
            first = model(batch.question_ids, batch.question_mask, memory=snapshot.memory, phase='query')
            second = model(batch.question_ids, batch.question_mask, memory=snapshot.memory, phase='query')
            assert torch.equal(first.logits, second.logits)
            assert torch.equal(old, snapshot.memory[14].M)
            assert first.cache.backbone is None
            assert first.cache.memory[14].counts.tolist() == [2, 2]
            _, normal_logits = evaluate_batch(model, batch)
            reordered = collate(episodes, pad_id, owners=[1, 0, 1])
            diag, reordered_logits = evaluate_batch(model, reordered)
            torch.testing.assert_close(reordered_logits, normal_logits[[1, 0, 1]], atol=2e-5, rtol=2e-5)
            assert diag['read.reconstruction_max_abs'] < 2e-5
            for mode in ('oracle_key', 'oracle_value', 'zero', 'shuffled', 'permuted'):
                diag, _ = evaluate_batch(model, batch, mode)
                assert math.isfinite(diag['nll'])
                if mode == 'oracle_value':
                    assert diag['read.normalized_error'] == 0
        # Oracle target metadata is irrelevant to deployed training forward.
        mutated = [{**e, 'query': {**e['query'], 'target_fact_index': -100}} for e in episodes]
        with torch.no_grad():
            _, a, _, _ = train_forward(model, batch)
            _, b, _, _ = train_forward(model, collate(mutated, pad_id))
            assert torch.equal(a, b)


def test_independent_probes():
    x = torch.tensor([[1., 0.], [0., 1.], [2., 0.], [0., 2.]])
    # Read coordinates are swapped: independent fitting recovers the labels,
    # applying the clean head to reads fails (distribution shift).
    y = torch.tensor([10, 20, 10, 20])
    result = independent_probes(x, x.flip(-1), y, x[:2], x[:2].flip(-1), y[:2])
    assert result['clean_to_clean'] == 1
    assert result['read_to_read'] == 1
    assert result['clean_to_read'] == 0
