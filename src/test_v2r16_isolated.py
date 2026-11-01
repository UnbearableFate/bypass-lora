"""E-0016 D4/D5 isolated-encoding write path. Execute in a GPU allocation."""
import json
import time

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION, load_base
from v2_data import encode
from v2_isolated import annotate_isolated, fact_text
from v2_memory import (ISOLATED_PAD_ID, answer_logits, build_support, collate, config, isolated_support,
                       isolated_writes)
from v2_train import adapted
import test_v2r_diagnostics as e0015

PARAMETERS = {'projection.A_q', 'projection.A_k', 'projection.A_v', 'reader.Rq', 'reader.Ro', 'fusion.P'}


@pytest.fixture(scope='module')
def base():
    return load_base()


@pytest.fixture(scope='module')
def tok():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)


def isolated_forward(model, batch):
    """Answer CE with the isolated write (the D4/D5 training computation)."""
    support, accounting = isolated_support(model, batch)
    output = model(batch.question_ids, batch.question_mask, memory=support.select(batch.owners).memory, phase='query')
    logits = answer_logits(output, batch.question_mask)
    return F.cross_entropy(logits, batch.labels), logits, accounting, output.accounting


def test_annotation_is_the_e1_format(tok):
    rows = e0015.rows_with(tok, 4)
    for row in rows:
        before = [dict(item) for item in row['_isolated']]
        assert annotate_isolated(tok, [row]) == 0 and row['_isolated'] == before
        for fact, item in zip(row['facts'], row['_isolated']):
            assert item['ids'] == encode(tok, fact_text(fact)) and item['delimiter'] == len(item['ids']) - 1
    bad = dict(rows[0], support_text=rows[0]['support_text'] + ' ')
    with pytest.raises(ValueError, match='concatenated fact format'):
        annotate_isolated(tok, [bad])


@torch.no_grad()
def test_isolated_support_is_the_e0015_isolated_encoding(base, tok):
    rows = e0015.rows_with(tok, 4)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        e0015.randomize_exits(model)
        batch = collate(rows, tok.pad_token_id)
        snapshot, accounting = isolated_support(model, batch)
        iso = isolated_writes(model, rows, ISOLATED_PAD_ID, 'cuda')
        # Same sequences, same slot order: the stored slots equal E-0015's isolated writes.
        assert torch.equal(snapshot.valid, iso['valid'])
        assert torch.equal(snapshot.keys, iso['keys']) and torch.equal(snapshot.values, iso['values'])
        state = snapshot.memory[14]
        assert state.backend == 'soft_slot' and torch.equal(state.slot_valid, iso['valid'])
        assert torch.equal(state.slot_keys, iso['keys']) and torch.equal(state.slot_values, iso['values'])
        assert torch.equal(state.counts, torch.full((4,), 4, device='cuda'))
        # One support write per fact and no support-side reads, as in build_support.
        assert accounting[14]['side']['writes'] == int(batch.write_mask.sum()) == 16
        assert accounting[14]['side']['reads'] == 0
        contextual, _ = build_support(model, batch, observe=True)
        assert torch.equal(snapshot.valid, contextual.valid)
        with pytest.raises(ValueError, match='one question per support owner'):
            isolated_support(model, collate(rows, tok.pad_token_id, owners=[1, 0, 2, 3]))


@torch.no_grad()
def test_each_slot_depends_only_on_its_own_fact(base, tok):
    rows = e0015.rows_with(tok, 4)
    changed = [dict(r) for r in rows]
    # Replace the tail of fact 1 of row 0 by another single-token name of the same length.
    facts = [{k: f[k] for k in ('head', 'relation', 'tail')} for f in rows[0]['facts']]
    old = facts[1]['tail']
    facts[1]['tail'] = next(n for n in e0015.NAMES if n != old and n not in (f['head'] for f in facts)
                            and len(encode(tok, ' ' + n)) == 1
                            and len(encode(tok, fact_text(dict(facts[1], tail=n)))) == len(encode(tok, fact_text(facts[1]))))
    changed[0] = e0015.make_row(tok, 0, facts, rows[0]['query']['target_fact_index'], prefix=0)
    e0015.diag.annotate(tok, changed, 4)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        a, _ = isolated_support(model, collate(rows, tok.pad_token_id))
        b, _ = isolated_support(model, collate(changed, tok.pad_token_id))
        assert not torch.equal(a.values[0, 1], b.values[0, 1])
        for j in (0, 2, 3):
            assert torch.equal(a.values[0, j], b.values[0, j]) and torch.equal(a.keys[0, j], b.keys[0, j])
        assert torch.equal(a.values[1:], b.values[1:])
        # The contextual write of the later facts sees the change.
        ca, _ = build_support(model, collate(rows, tok.pad_token_id), observe=True)
        cb, _ = build_support(model, collate(changed, tok.pad_token_id), observe=True)
        assert not torch.equal(ca.values[0, 2], cb.values[0, 2])


def test_training_gradients_flow_through_isolated_writes(base, tok):
    rows = e0015.rows_with(tok, 4)
    batch = collate(rows, tok.pad_token_id)
    mask = batch.question_mask
    last = mask.long().sum(-1) - 1
    with torch.no_grad():  # Base answer logits, before the adapter is installed (as run_arm)
        reference = base(batch.question_ids, attention_mask=mask,
                         position_ids=(mask.long().cumsum(-1) - 1).clamp_min(0), use_cache=False).logits
        reference = reference[torch.arange(len(last), device=last.device), last].float()
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        side = model.adapters()[14].side
        model.eval()
        with torch.no_grad():
            # Zero-initialized exit: the isolated write leaves the Base answer logits unchanged.
            _, logits, _, _ = isolated_forward(model, batch)
            assert torch.equal(logits, reference)
        e0015.randomize_exits(model)
        model.train()
        model.model.eval()
        loss, _, support_counts, query_counts = isolated_forward(model, batch)
        loss.backward()
        assert support_counts[14]['side']['reads'] == 0 and query_counts[14]['side']['writes'] == 0
        grads = {n: p.grad for n, p in side.named_parameters()}
        assert set(grads) == PARAMETERS
        assert all(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g) for g in grads.values())
        assert all(p.grad is None for n, p in model.model.named_parameters() if '.side.' not in n)
        # The answer CE reaches the stored isolated slots (keys and values), whose only
        # producer is the single-fact support pass.
        model.zero_grad(set_to_none=True)
        support, _ = isolated_support(model, batch)
        output = model(batch.question_ids, batch.question_mask, memory=support.select(batch.owners).memory,
                       phase='query')
        ce = F.cross_entropy(answer_logits(output, batch.question_mask), batch.labels)
        dk, dv = torch.autograd.grad(ce, (support.memory[14].slot_keys, support.memory[14].slot_values))
        assert torch.count_nonzero(dk) and torch.count_nonzero(dv)


def test_a_few_optimizer_steps_lower_the_isolated_loss_and_time_them(base, tok):
    rows = e0015.rows_with(tok, 4)
    batch = collate(rows, tok.pad_token_id)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.train()
        model.model.eval()
        optimizer = torch.optim.AdamW(list(model.adapter_parameters()), lr=3e-3, weight_decay=0.)
        losses, seconds = [], []
        for _ in range(12):
            torch.cuda.synchronize()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, *_ = isolated_forward(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.adapter_parameters()), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            seconds.append(time.monotonic() - tick)
            losses.append(float(loss))
        print('isolated losses', [round(x, 3) for x in losses], 'step seconds', [round(x, 3) for x in seconds])
        assert losses[-1] < losses[0]


# ---- D4/D5 runner pieces -------------------------------------------------------------

def summary(em, nll, weight=None, last_weight=None, correct=0):
    return {'em': em, 'nll': nll, 'count': 1000, 'correct_count': correct,
            'diagnostics': {} if weight is None else {'address.target_weight': weight},
            'last_position': {'target_weight_mean': last_weight}}


def test_isolated_gate_uses_the_e0010_h3_thresholds():
    import v2r16_isolated as runner
    passing = {'normal': summary(.30, 3.0, .2, .3, 300), 'zero': summary(.01, 9.0), 'shuffled': summary(.19, 4.0)}
    result = runner.gate(passing)
    assert result['passed'] and abs(result['em_margin'] - .11) < 1e-12
    assert result['correct'] == {'normal': 300, 'zero': 0, 'shuffled': 0}
    assert result['target_weight_last_position'] == .3
    assert not runner.gate(dict(passing, shuffled=summary(.21, 4.0)))['passed']      # margin < 10 pp
    assert not runner.gate(dict(passing, shuffled=summary(.01, 2.9)))['passed']      # NLL not below
    assert not runner.gate(dict(passing, normal=summary(.30, 3.0, .124)))['passed']  # all-position weight


def test_r1_provenance_accepts_the_recorded_run_and_rejects_changes(tmp_path):
    import shutil
    import v2r16_isolated as runner
    cfg = config('direct_hidden', 'soft_slot', 'F_out')
    record = runner.r1_provenance(runner.R1_RUN, cfg)
    assert set(record['checkpoint_files']) == {'adapter.pt', 'config.json'}
    assert record['adapter_only'] and record['optimizer_state_loaded'] is False
    copy = tmp_path/'r1'
    (copy/runner.ARM_DIR).mkdir(parents=True)
    shutil.copytree(runner.R1_RUN/runner.ARM_DIR/'adapter', copy/runner.ARM_DIR/'adapter')
    shutil.copy(runner.R1_RUN/'metrics.json', copy/'metrics.json')
    manifest = json.loads((runner.R1_RUN/'manifest.json').read_text())
    (copy/'manifest.json').write_text(json.dumps(dict(manifest, smoke_steps=20)))
    with pytest.raises(ValueError, match='completed formal'):
        runner.r1_provenance(copy, cfg)
    (copy/'manifest.json').write_text(json.dumps(manifest))
    assert runner.r1_provenance(copy, cfg)['checkpoint_files'] == record['checkpoint_files']
    (copy/runner.ARM_DIR/'adapter'/'optimizer.pt').write_bytes(b'x')
    with pytest.raises(ValueError, match='only adapter'):
        runner.r1_provenance(copy, cfg)
