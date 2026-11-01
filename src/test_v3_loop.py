"""E-0018: the S>1 accumulating loop and T2 on the isolated-write M line.

The core-reader tests run anywhere; the model tests need a GPU allocation. The released-data
tests read T2 train/dev and T1 dev only.
"""
import json
import shutil
import sys

import pytest
import torch
import torch.nn.functional as F

from bypass_core import LatentReader, MemoryState
import v3_loop as loop

D5 = loop.RUNS/'ATT-0159-1'/'run'
D5_SHA256 = 'c4f3f9f8c24a55a89142cd251e845a39ac4c235fb22aa244afb8e7cc3425b794'
PARAMETERS = {'projection.A_q', 'projection.A_k', 'projection.A_v', 'reader.Rq', 'reader.Ro', 'fusion.P'}


# ---- core recurrence (plan §3.5) -------------------------------------------------------------

def reference_loop(qbar, keys, values, valid, Rq, Ro, steps, eta=.25, gain=1., eps=1e-6):
    """Independent plan §3.5 recurrence with an explicit SoftmaxSlot read (any dtype)."""
    z0 = z = qbar
    for _ in range(steps):
        normed = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
        query = F.elu(normed @ Rq) + 1
        scores = (query[:, None, :] * keys).sum(-1) / keys.shape[-1] ** .5
        weights = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        z = z + eta * ((gain * (weights[..., None] * values).sum(1)) @ Ro)
    return z - z0


def small_case(dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    b, n, r = 3, 5, 8
    make = lambda *shape, scale=1.: (torch.randn(*shape, generator=generator) * scale).to(dtype)
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[2, 3:] = False
    return {'qbar': make(b, r), 'keys': F.elu(make(b, n, r)) + 1, 'values': make(b, n, r),
            'valid': valid, 'Rq': torch.eye(r, dtype=dtype) + make(r, r, scale=.3),
            'Ro': torch.eye(r, dtype=dtype) + make(r, r, scale=.3)}


def soft_slot_state(keys, values, valid):
    state = MemoryState.empty(len(keys), keys.shape[-1], 'soft_slot', keys.device)
    for j in range(keys.shape[1]):
        state = state.write(keys[:, j], values[:, j], valid[:, j])
    return state


@pytest.mark.parametrize('steps', (1, 2, 4, 8))
def test_core_reader_is_the_plan_accumulating_loop(steps):
    case = small_case()
    reader = LatentReader(8)
    with torch.no_grad():
        reader.Rq.copy_(case['Rq'])
        reader.Ro.copy_(case['Ro'])
    state = soft_slot_state(case['keys'], case['values'], case['valid'])
    out = reader(case['qbar'], state, steps=steps)
    assert reader.last_read_calls == steps
    double = {k: v.double() if v.is_floating_point() else v for k, v in case.items()}
    expected = reference_loop(double['qbar'], double['keys'], double['values'], double['valid'],
                              double['Rq'], double['Ro'], steps)
    assert torch.allclose(out.double(), expected, rtol=1e-5, atol=1e-6)
    if steps > 1:  # more reads change the increment
        assert not torch.allclose(out, reader(case['qbar'], state, steps=1))


def test_loop_gradients_match_an_independent_double_reference():
    case = small_case(torch.float64, seed=1)
    names = ('qbar', 'keys', 'values', 'Rq', 'Ro')
    inputs = tuple(case[n].clone().requires_grad_(True) for n in names)
    weight = torch.randn(3, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    reference = lambda q, k, v, rq, ro: (reference_loop(q, k, v, case['valid'], rq, ro, 4) * weight).sum()
    assert torch.autograd.gradcheck(reference, inputs)
    expected = torch.autograd.grad(reference(*inputs), inputs)
    reader = LatentReader(8)
    q, k, v = (case[n].float().clone().requires_grad_(True) for n in ('qbar', 'keys', 'values'))
    with torch.no_grad():
        reader.Rq.copy_(case['Rq'])
        reader.Ro.copy_(case['Ro'])
    out = reader(q, soft_slot_state(k, v, case['valid']), steps=4)
    actual = torch.autograd.grad((out * weight.float()).sum(), (q, k, v, reader.Rq, reader.Ro))
    for name, a, e in zip(names, actual, expected):
        assert torch.allclose(a.double(), e, rtol=1e-4, atol=1e-5), name


# ---- summaries, gate, provenance and CLI (no model) ---------------------------------------------

def record(hop, correct, weights, steps=2, argmax=0, path=(0, 1, 2), tokens=10):
    return {'episode_id': f'e{hop}{correct}{argmax}{weights}', 'hop': hop, 'support_order': 'natural',
            'target': 7, 'prediction': 7 if correct else 8, 'correct': correct, 'nll': .5, 'question_tokens': tokens,
            'steps': steps, 'final_state_norm_last': 1., 'increment_norm_last': .1, 'increment_norm_all': .1,
            **{n: [1.]*steps for n in ('n_eff_all', 'n_eff_last', 'query_cosine_all', 'query_cosine_last',
                                         'state_norm_all', 'state_norm_last', 'read_norm_all', 'read_norm_last')},
            'argmax_last': [argmax]*steps, 'path_weight_all': [list(weights[:hop])]*steps,
            'path_weight_last': [list(weights[:hop])]*steps, '_path': list(path[:hop])}


def test_path_weights_average_only_rows_that_reach_the_hop():
    records = [record(1, True, (.9,)), record(2, False, (.5, .3), argmax=1), record(3, True, (.2, .4, .6), argmax=2)]
    summary = loop.summarize_loop(records, prior_token=7)
    step = summary['all']['per_step'][1]
    assert step['path_weight_last'] == pytest.approx([(.9+.5+.2)/3, (.3+.4)/2, .6])
    assert step['argmax_is_path_fact'] == pytest.approx([1/3, 1/2, 1.])
    assert summary['hop2']['per_step'][0]['path_weight_all'] == pytest.approx([.5, .3])
    assert summary['all']['em'] == pytest.approx(2/3) and summary['all']['prior_em'] == 1.
    assert summary['hop3.natural']['count'] == 1 and 'per_step' not in summary['hop3.natural']
    donor = [dict(r, path_weight_all=None, path_weight_last=None) for r in records]
    assert 'path_weight_all' not in loop.summarize_loop(donor)['all']['per_step'][0]


def reviewed_record(episode, correct, weight=.2):
    return {'episode_id': episode, 'target': 7, 'prediction': 7 if correct else 8, 'correct': correct, 'nll': 1.,
            'question_tokens': 10, 'all_target_weight': weight, 'last_target_weight': weight, 'all_n_eff': 2.,
            'last_n_eff': 2., 'last_argmax': 0}


def test_hop1_gate_is_the_exact_h3_gate_on_hop1_rows():
    rows = [{'episode_id': f'h{h}-{i}', 'axes': {'hop': h}} for h in (1, 2) for i in range(10)]
    hop1 = [r['episode_id'] for r in rows if r['axes']['hop'] == 1]
    reviewed = {
        'normal': [reviewed_record(e, e in hop1[:3]) for e in (r['episode_id'] for r in rows)],
        'zero': [dict(reviewed_record(e, e in hop1[:2]), nll=2.) for e in (r['episode_id'] for r in rows)],
        'shuffled': [dict(reviewed_record(e, False), nll=2., all_target_weight=None, last_target_weight=None)
                     for e in (r['episode_id'] for r in rows)]}
    by_hop = loop.reviewed_by_hop(reviewed, rows, 7)
    gate = loop.hop1_gate(by_hop)
    # 3/10 - 2/10 is exactly 10 pp: passes with exact fractions.
    assert gate['passed'] and gate['em_margin_fraction'] == '1/10' and gate['counts']['normal'] == 3
    reviewed['normal'] = [dict(r, all_target_weight=.124) for r in reviewed['normal']]
    assert not loop.hop1_gate(loop.reviewed_by_hop(reviewed, rows, 7))['passed']


def test_compare_to_reviewed_is_exact():
    mine = {'episode_id': 'a', 'prediction': 7, 'correct': True, 'nll': 1., 'path_weight_all': [[.2]],
            'path_weight_last': [[.2]], 'n_eff_all': [2.], 'n_eff_last': [2.], 'argmax_last': [0]}
    assert loop.compare_to_reviewed([mine], [reviewed_record('a', True)])['identical']
    nudged = dict(mine, nll=1.0000001)
    assert loop.compare_to_reviewed([nudged], [reviewed_record('a', True)])['differing_examples_by_field'] == {'nll': 1}
    with pytest.raises(AssertionError):
        loop.compare_to_reviewed([mine], [reviewed_record('b', True)])


def test_adapter_provenance_pins_the_recorded_digest(tmp_path):
    assert loop.adapter_provenance(D5, D5_SHA256)['seed'] == 0
    for bad in ('0'*64, D5_SHA256.upper(), None):
        with pytest.raises(ValueError):
            loop.adapter_provenance(D5, bad)
    copy = tmp_path/'run'
    shutil.copytree(D5/'adapter', copy/'adapter')
    manifest = json.loads((D5/'manifest.json').read_text())
    (copy/'manifest.json').write_text(json.dumps(manifest))
    assert loop.adapter_provenance(copy, D5_SHA256)['checkpoint_files'] == manifest['checkpoint_files']
    (copy/'manifest.json').write_text(json.dumps(dict(manifest, smoke_steps=20)))
    with pytest.raises(ValueError, match='formal'):
        loop.adapter_provenance(copy, D5_SHA256)
    (copy/'manifest.json').write_text(json.dumps(manifest))
    (copy/'adapter'/'extra.pt').write_bytes(b'x')
    with pytest.raises(ValueError, match='manifest'):
        loop.adapter_provenance(copy, D5_SHA256)


@pytest.mark.parametrize('extra', (['--s-eval', '2,4'], ['--s-eval', '1,3'], ['--s-eval', '1,4,2'],
                                   ['--t1-check'], ['--eval-limit', '1']))
def test_cli_rejects_other_protocols(monkeypatch, tmp_path, extra):
    run = loop.RUNS/'ATT-0173-1'/'run'
    monkeypatch.setattr(sys, 'argv', ['v3_loop.py', 'eval', '--adapter-run', str(run), '--adapter-sha256', 'a'*64,
                                      '--output', str(tmp_path/'out')] + extra)
    with pytest.raises(SystemExit):
        loop.main()


def test_cli_rejects_other_adapters_and_timing_settings(monkeypatch, tmp_path):
    common = ['--adapter-sha256', 'a'*64, '--output', str(tmp_path/'out')]
    for argv in (['eval', '--adapter-run', str(loop.RUNS/'ATT-0122-1'/'run')],
                 ['timing', '--adapter-run', str(D5), '--timing-s', '8'],
                 ['timing', '--adapter-run', str(D5), '--timing-steps', '500']):
        monkeypatch.setattr(sys, 'argv', ['v3_loop.py'] + argv + common)
        with pytest.raises(SystemExit):
            loop.main()


# ---- released data ------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)


@pytest.fixture(scope='module')
def t2_dev(tok):
    metadata, rows, differing = loop.load_t2(tok, 'dev')
    return metadata, rows, differing


def test_t2_dev_paths_and_sealed_slices(tok, t2_dev):
    metadata, rows, differing = t2_dev
    assert len(rows) == 1000 and metadata['phase'] == 'V3'
    assert [sum(r['axes']['hop'] == h for r in rows) for h in (1, 2, 3)] == [334, 333, 333]
    assert all(len(r['_isolated']) == 16 for r in rows)
    print('T2 dev facts whose context tokens differ from the isolated encoding:', differing)
    for split in ('confirmation', 'ood_h4', 'ood_h6', 'ood_h8', 'ood_unseen'):
        with pytest.raises(ValueError):
            loop.load_t2(tok, split)
    broken = json.loads(json.dumps({k: v for k, v in rows[1].items() if k != '_isolated'}))
    broken['path']['fact_indices'] = broken['path']['fact_indices'][::-1]
    with pytest.raises(ValueError):
        loop.check_path(broken)
    paths = loop.path_matrix(rows[:6], 'cpu')
    assert paths[0, :rows[0]['axes']['hop']].tolist() == rows[0]['path']['fact_indices']
    assert (paths[0, rows[0]['axes']['hop']:] == -1).all()


# ---- model (GPU) ---------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def base():
    from calibrate import load_base
    return load_base()


def gpu_batch(tok, rows):
    from v2_memory import collate
    return collate(rows, tok.pad_token_id, 'cuda')


def native_logits(model, batch):
    from v2_memory import answer_logits, isolated_support
    support, _ = isolated_support(model, batch)
    output = model(batch.question_ids, batch.question_mask, memory=support.select(batch.owners).memory, phase='query')
    return answer_logits(output, batch.question_mask)


def side_gradients(model, batch):
    from v2_memory import isolated_support
    from v2r16_memory import train_forward
    model.zero_grad(set_to_none=True)
    model.train()
    model.model.eval()
    loss, *_ = train_forward(model, batch, support_fn=isolated_support)
    loss.backward()
    model.eval()
    return {n: p.grad.clone() for n, p in model.adapters()[14].side.named_parameters()}


def test_default_off_is_master_and_s_changes_only_the_step_count(base, tok, t2_dev):
    from v2_train import adapted
    batch = gpu_batch(tok, t2_dev[1][:16])
    with adapted(base, checkpoint=D5/'adapter') as model:
        model.eval()
        config, side_config = model.config, model.adapters()[14].side.config
        parameters = {n: p.detach().clone() for n, p in model.adapters()[14].side.named_parameters()}
        with torch.no_grad():
            master = native_logits(model, batch)
            with loop.loop_steps(model, 1):
                assert torch.equal(native_logits(model, batch), master)
            with loop.loop_steps(model, 4):
                assert model.config.steps == 4 and model.adapters()[14].side.config.steps == 4
                four = native_logits(model, batch)
                with pytest.raises(ValueError):
                    with loop.loop_steps(model, 2):
                        pass
            assert model.config is config and model.adapters()[14].side.config is side_config
            assert torch.equal(native_logits(model, batch), master) and not torch.equal(four, master)
        assert all(torch.equal(p, parameters[n]) for n, p in model.adapters()[14].side.named_parameters())
        for bad in (0, 1.0, True):
            with pytest.raises(ValueError):
                with loop.loop_steps(model, bad):
                    pass
        # Gradients: S=1 is master exactly; S=2 reaches all six parameter groups, finite.
        reference = side_gradients(model, batch)
        with loop.loop_steps(model, 1):
            same = side_gradients(model, batch)
        assert all(torch.equal(reference[n], same[n]) for n in reference)
        with loop.loop_steps(model, 2):
            two = side_gradients(model, batch)
        assert set(two) == PARAMETERS
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in two.values())
        assert any(not torch.equal(two[n], reference[n]) for n in two)
        assert all(p.grad is None for n, p in model.model.named_parameters() if '.side.' not in n)
        model.zero_grad(set_to_none=True)


def test_s1_loop_evaluator_equals_the_reviewed_evaluator(base, tok, t2_dev, tmp_path):
    from v2_isolated import annotate_isolated
    from v2_memory import isolated_support
    from v2_schedule import load_data
    from v2_train import adapted
    from v2r16_memory import evaluate
    t1 = load_data(loop.T1_DATA/'dev.jsonl', 'dev')[:48]
    annotate_isolated(tok, t1)
    with adapted(base, checkpoint=D5/'adapter') as model:
        for name, rows in (('t2', t2_dev[1][:48]), ('t1', t1)):
            _, reviewed = evaluate(model, rows, tok.pad_token_id, tmp_path/name, 'dev', loop.LOOP_CONTROLS, None,
                                   support_fn=isolated_support)
            mine, reconstruction = loop.evaluate_loop(model, rows, tok.pad_token_id, 1)
            print(name, 'S=1 reconstruction', reconstruction)
            for mode in loop.LOOP_CONTROLS:
                comparison = loop.compare_to_reviewed(mine[mode], reviewed[mode])
                assert comparison['identical'], (name, mode, comparison)


def test_s_step_records_and_fixed_memory(base, tok, t2_dev):
    from v2_train import adapted
    rows = t2_dev[1][:32]
    with adapted(base, checkpoint=D5/'adapter') as model:
        records = {}
        for steps in (1, 2, 4):
            records[steps], reconstruction = loop.evaluate_loop(model, rows, tok.pad_token_id, steps)
            print('S', steps, 'reconstruction', reconstruction)
            assert max(r['max_relative_to_state'] for r in reconstruction.values()) <= loop.RECONSTRUCTION_TOLERANCE
        for steps in (2, 4):
            normal = records[steps]['normal']
            for r, row in zip(normal, rows):
                assert len(r['path_weight_all']) == steps and all(len(w) == row['axes']['hop'] for w in r['path_weight_all'])
                assert r['query_cosine_last'][0] == pytest.approx(1., abs=1e-6)
                # Step 1 of an S-step evaluation is the S=1 read of the same memory.
                one = next(x for x in records[1]['normal'] if x['episode_id'] == r['episode_id'])
                assert r['path_weight_all'][0] == one['path_weight_all'][0] and r['argmax_last'][0] == one['argmax_last'][0]
                assert r['state_norm_last'][0] == one['state_norm_last'][0]
            assert records[steps]['shuffled'][0]['path_weight_all'] is None
            # Empty memory reads zero at every step: the zero control does not depend on S.
            assert [(r['prediction'], r['nll']) for r in records[steps]['zero']] == \
                   [(r['prediction'], r['nll']) for r in records[1]['zero']]
            assert all(n == 0. for r in records[steps]['zero'] for n in r['read_norm_last'])
        assert any(a['nll'] != b['nll'] for a, b in zip(records[1]['normal'], records[4]['normal']))
