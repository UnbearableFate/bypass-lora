"""E-0016 real-model and masked-slot regressions; execute only inside PBS."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from bypass_core import MemoryState
from calibrate import MODEL, REVISION, load_base
from v2_memory import Snapshot, build_support, collate, config, train_forward as old_forward
from v2_memory import answer_logits
from v2_run import fixture
from v2_train import adapted
from v2r16_environment import selected_profile
from v2r16_memory import (ADDRESS, DECODER, _query, compact_snapshot, evaluate_batch,
                          freeze_for_address, single_slot, train_forward)
from v2r16_objective import slot_marginal


def test_selected_profile_matches_allocated_uuid_not_any_good_slice_on_host():
    listing = ('GPU 0: GH200 (UUID: GPU-host)\n'
               '  MIG 2g.24gb Device 0: (UUID: MIG-good)\n'
               '  MIG 1g.24gb Device 1: (UUID: MIG-small)\n')
    assert selected_profile(listing, 'MIG-small') == ('1g.24gb', 'MIG-small')
    assert selected_profile(listing, 'MIG-good') == ('2g.24gb', 'MIG-good')
    with pytest.raises(RuntimeError, match='unique MIG'):
        selected_profile(listing, '0')
    with pytest.raises(RuntimeError, match='absent or ambiguous'):
        selected_profile(listing, 'MIG-missing')


def test_single_slot_tracks_sparse_native_columns_without_mutating_snapshot():
    state = MemoryState.empty(2, 3, 'soft_slot')
    for i in range(4):
        state = state.write(torch.full((2, 3), float(i+1)), torch.full((2, 3), float(10+i)),
                            torch.tensor([i in (0, 2, 3), i in (1, 3)]))
    snapshot = compact_snapshot(Snapshot({14: state}))
    result = single_slot(snapshot, torch.tensor([1, 0]))
    assert result.memory[14].slot_valid.tolist() == [[False, False, True, False], [False, True, False, False]]
    assert result.valid.tolist() == [[False, True, False], [True, False, False]]
    assert result.memory[14].counts.tolist() == [1, 1]
    assert state.counts.tolist() == [3, 2]
    assert torch.equal(result.values, snapshot.values)


@pytest.fixture(scope='module')
def real_inputs():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    rows = fixture(tokenizer, n_facts=4)[:2]
    for i, row in enumerate(rows):
        row['episode_id'] = f'v2r16-fixture-{i}'
        row['query']['target_fact_index'] = i+1
    return load_base(), collate(rows, tokenizer.pad_token_id), rows, tokenizer.pad_token_id


def exits(model):
    with torch.no_grad():
        model.adapters()[14].side.fusion.P.normal_(std=.025)


def test_default_wrapper_preserves_exact_outputs_and_gradients(real_inputs):
    base, batch, _, _ = real_inputs
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.model.eval()
        exits(model)
        reference_loss, reference_logits, _, _ = old_forward(model, batch)
        reference_loss.backward()
        grads = {name: p.grad.clone() for name, p in model.adapters()[14].side.named_parameters()}
        model.zero_grad(set_to_none=True)
        loss, logits, _, _ = train_forward(model, batch)
        loss.backward()
        assert torch.equal(reference_loss, loss) and torch.equal(reference_logits, logits)
        assert all(torch.equal(p.grad, grads[name]) for name, p in model.adapters()[14].side.named_parameters())


def test_d2a_has_only_nonzero_address_gradients_and_frozen_decoder(real_inputs):
    base, batch, _, _ = real_inputs
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.model.eval()
        exits(model)
        assert set(freeze_for_address(model)) == ADDRESS
        frozen = {n: p.detach().clone() for n, p in model.adapters()[14].side.named_parameters() if n in DECODER}
        loss, _, _, _ = train_forward(model, batch)
        loss.backward()
        parameters = dict(model.adapters()[14].side.named_parameters())
        assert all(parameters[n].grad is not None and bool(parameters[n].grad.any()) for n in ADDRESS)
        assert all(parameters[n].grad is None for n in DECODER)
        torch.optim.AdamW([p for p in parameters.values() if p.requires_grad], lr=.003, weight_decay=0.).step()
        assert all(torch.equal(parameters[n], tensor) for n, tensor in frozen.items())


@torch.no_grad()
def test_hard_read_matches_selected_value_and_other_positions_have_no_increment(real_inputs):
    base, batch, _, _ = real_inputs
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        exits(model)
        support, _ = build_support(model, batch)
        snapshot = compact_snapshot(support.select(batch.owners))
        hard, cache = _query(model, batch.question_ids, batch.question_mask, snapshot, strategy='hard')
        row = torch.arange(len(batch.labels), device=batch.labels.device)
        chosen = cache['weights'][row, cache['last']].argmax(-1)
        forced, forced_cache = _query(model, batch.question_ids, batch.question_mask, snapshot,
                                     strategy='forced', forced_values=snapshot.values[row, chosen])
        assert torch.equal(hard.logits, forced.logits)
        last_mask = torch.zeros_like(batch.question_mask).scatter(1, cache['last'][:, None], True)
        assert not bool(cache['z'][~last_mask].any())
        assert not bool(forced_cache['z'][~last_mask].any())
        soft, soft_cache = _query(model, batch.question_ids, batch.question_mask, snapshot, strategy='soft')
        assert not bool(soft_cache['z'][~last_mask].any())


def test_marginal_uses_support_once_has_no_target_metadata_and_reaches_both_paths(real_inputs):
    base, batch, _, _ = real_inputs
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.model.eval()
        exits(model)
        calls = []
        handle = model.register_forward_pre_hook(lambda m, args, kwargs: calls.append((kwargs.get('phase'), len(args[0]))),
                                                 with_kwargs=True)
        try:
            loss, logits, _, _ = train_forward(model, replace(batch, rows=[{} for _ in batch.rows]), protocol='marginal')
        finally:
            handle.remove()
        assert calls == [('support', 2), ('query', 8)]
        assert torch.isfinite(loss) and logits.shape[0] == 2
        loss.backward()
        parameters = dict(model.adapters()[14].side.named_parameters())
        assert all(parameters[n].grad is not None and bool(parameters[n].grad.any()) for n in ADDRESS | DECODER)


@torch.no_grad()
def test_batched_marginal_matches_explicit_slot_forward_enumeration(real_inputs):
    base, batch, _, _ = real_inputs
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        exits(model)
        actual_loss, actual_logp, _, _ = train_forward(model, batch, protocol='marginal')
        support, _ = build_support(model, batch)
        snapshot = compact_snapshot(support.select(batch.owners))
        n = snapshot.valid.shape[1]
        owners = torch.arange(len(batch.labels), device=batch.labels.device).repeat_interleave(n)
        expanded = snapshot.select(owners)
        ids, mask = batch.question_ids[owners], batch.question_mask[owners]
        per_slot = []
        for slot in range(n):
            values = snapshot.values[:, slot].repeat_interleave(n, 0)
            output, cache = _query(model, ids, mask, expanded, strategy='forced', forced_values=values)
            per_slot.append(answer_logits(output, mask)[::n])
        logw = cache['log_weights'][torch.arange(len(owners), device=owners.device), cache['last']][::n]
        expected_loss, expected_logp = slot_marginal(logw, torch.stack(per_slot, 1), batch.labels)
        tolerance = json.loads((Path(__file__).parent/'tolerances.json').read_text())
        torch.testing.assert_close(actual_loss, expected_loss, atol=tolerance['fp32_atol'], rtol=tolerance['fp32_rtol'])
        torch.testing.assert_close(actual_logp, expected_logp, atol=tolerance['fp32_atol'], rtol=tolerance['fp32_rtol'])


@torch.no_grad()
def test_teacher_normal_wrong_and_donor_same_index_masks(real_inputs, monkeypatch):
    import v2r16_memory
    base, batch, rows, pad = real_inputs
    targets = [r['query']['target_fact_index'] for r in rows]
    selected, original = [], v2r16_memory.single_slot
    def record(snapshot, target):
        selected.append(target.tolist())
        return original(snapshot, target)
    monkeypatch.setattr(v2r16_memory, 'single_slot', record)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        exits(model)
        evaluate_batch(model, batch, 'tf_normal')
        evaluate_batch(model, batch, 'tf_wrong')
        evaluate_batch(model, batch, 'tf_shuffled', collate(rows[::-1], pad))
    assert selected == [targets, [(t+1) % 4 for t in targets], targets]


@torch.no_grad()
def test_question_hooks_are_removed_after_failure(real_inputs):
    base, batch, _, _ = real_inputs
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        side = model.adapters()[14].side
        support, _ = build_support(model, batch)
        snapshot = compact_snapshot(support.select(batch.owners))
        before = (len(side._forward_hooks), len(side._forward_pre_hooks), len(side.projection._forward_hooks))
        with pytest.raises(ValueError, match='forced per-slot'):
            _query(model, batch.question_ids, batch.question_mask, snapshot, strategy='forced', forced_values=torch.zeros(1, 1))
        assert before == (len(side._forward_hooks), len(side._forward_pre_hooks), len(side.projection._forward_hooks))
