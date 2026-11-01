"""E-0022: T2 training path and the plan §9.2 arm variants on the isolated-write M line.

Core tests run anywhere; model tests need a GPU allocation and use the D5 seed-0 adapter as a
stand-in common start. The released-data tests read T2 train/dev only.
"""
from collections import Counter
from dataclasses import asdict, replace
from fractions import Fraction
import json
import sys

import pytest
import torch
import torch.nn.functional as F

from bypass_core import LatentReader, MemoryState
import v3_loop as loop
import v3_train as v3

D5 = loop.RUNS/'ATT-0159-1'/'run'
PARAMETERS = {'projection.A_q', 'projection.A_k', 'projection.A_v', 'reader.Rq', 'reader.Ro', 'fusion.P'}


# ---- schedules, arms, MACs ------------------------------------------------------------------

def fake_rows(count, hop=None):
    return [{'episode_id': f'e{i}', 'support_input_ids': [1]*(i % 5+1), 'question_input_ids': [2]*(i % 3+2),
             'axes': {'hop': hop or i % 3+1}} for i in range(count)]


def test_schedule_follows_the_contract_seed_rule():
    import random
    from v1_data import derive_seed
    rows = fake_rows(100)
    plan = v3.v3_schedule(rows, 'start', 30)
    assert plan == v3.v3_schedule(rows, 'start', 30)
    assert plan['steps'] == 30 and all(len(b) == 8 for b in plan['batches']) and plan['epochs'] == 3
    base = derive_seed(20261029, 'concat_loop_v2', 'V3', 'start', 0)
    expected = []
    for epoch in range(3):
        indices = list(range(100))
        random.Random(base + epoch).shuffle(indices)
        expected.extend(indices)
    assert [i for b in plan['batches'] for i in b] == expected[:240]
    assert plan['order_seed'] == base
    train = v3.v3_schedule(rows, 'train', 30)
    assert train['order_sha256'] != plan['order_sha256']
    assert train['order_seed'] == derive_seed(20261029, 'concat_loop_v2', 'V3', 'train', 0)
    full = v3.v3_schedule(fake_rows(6144), 'start')
    assert full['warmup_steps'] == 300 and full['steps'] == 6000 and full['epochs'] == 8
    smoke = v3.trim_smoke(full, fake_rows(6144), 5)
    assert smoke['steps'] == 5 and smoke['warmup_steps'] == 1 and smoke['batches'] == full['batches'][:5]
    for bad in ('dev', 'V3'):
        with pytest.raises(ValueError):
            v3.v3_schedule(rows, bad, 3)


def test_arms_are_the_plan_matrix_and_default_off():
    from v2_memory import config
    start = config('direct_hidden', 'soft_slot', 'F_out')
    configs = {arm: v3.arm_config(start, arm) for arm in v3.ARMS}
    assert configs['V3-01'] == start and v3.arm_variant('V3-01') == v3.Variant()
    assert {a: c.steps for a, c in configs.items()} == {
        'V3-01': 1, 'V3-02': 2, 'V3-03': 4, 'V3-04': 4, 'V3-05': 4, 'V3-06': 4, 'V3-07': 1, 'V3-08': 4,
        'V3-09': 4, 'V3-10': 4}
    assert configs['V3-04'].eta == .0625 and configs['V3-09'].update == 'anchored'
    assert configs['V3-10'].readout == 'full' and configs['V3-07'].rank == 152
    for arm in ('V3-02', 'V3-03', 'V3-05', 'V3-06', 'V3-08'):
        assert replace(configs[arm], steps=1) == start
    assert v3.arm_variant('V3-05') == v3.Variant(fixed_query=True)
    assert v3.arm_variant('V3-06') == v3.Variant(kernel='tokenwise')
    assert v3.arm_variant('V3-08') == v3.Variant(untied_steps=4)
    assert all(v3.arm_variant(a) == v3.Variant() for a in ('V3-01', 'V3-02', 'V3-03', 'V3-04', 'V3-07', 'V3-09', 'V3-10'))


def test_v3_07_width_is_mac_matched_to_v3_03():
    assert v3.macs_per_question_token('V3-03')['question_token'] == 671744
    assert v3.macs_per_question_token('V3-07')['question_token'] == 673664
    assert v3.macs_per_question_token('V3-07')['support_projection_per_token'] / \
        v3.macs_per_question_token('V3-03')['support_projection_per_token'] == pytest.approx(1.1875)
    # Executed counts: the fixed query projects once (RV-0058); the tokenwise kernel has no slot terms.
    assert v3.macs_per_question_token('V3-05')['question_token'] == 4096*128 + 128**2 + 4*(128**2 + 2*16*128) == 622592
    assert v3.macs_per_question_token('V3-06')['question_token'] == 4096*128 + 4*2*128**2
    for arm in ('V3-01', 'V3-02', 'V3-04', 'V3-08', 'V3-09', 'V3-10'):
        r, steps = 128, v3.ARMS[arm]['steps']
        assert v3.macs_per_question_token(arm)['question_token'] == 4096*r + steps*(2*r*r + 32*r)
    # 152 is the nearest integer width to the S4 MAC (151.6).
    target = 671744
    assert min(range(128, 200), key=lambda r: abs(4096*r + 2*r*r + 32*r - target)) == 152


# ---- core variants ---------------------------------------------------------------------------------

def small_state(seed=0, b=3, n=5, r=8):
    generator = torch.Generator().manual_seed(seed)
    keys = F.elu(torch.randn(b, n, r, generator=generator)) + 1
    values = torch.randn(b, n, r, generator=generator)
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[2, 3:] = False
    state = MemoryState.empty(b, r, 'soft_slot')
    for j in range(n):
        state = state.write(keys[:, j], values[:, j], valid[:, j])
    qbar = torch.randn(b, r, generator=generator)
    return state, qbar, generator


def perturbed_reader(r, generator, cls=LatentReader):
    reader = LatentReader(r)
    with torch.no_grad():
        reader.Rq.add_(torch.randn(r, r, generator=generator)*.3)
        reader.Ro.add_(torch.randn(r, r, generator=generator)*.3)
    reader.__class__ = cls
    return reader


@pytest.mark.parametrize('update,readout', [('accumulate', 'increment'), ('anchored', 'increment'),
                                            ('accumulate', 'full')])
def test_untied_copies_equal_the_shared_loop_and_train_apart(update, readout):
    state, qbar, generator = small_state()
    reader = perturbed_reader(8, generator)
    shared = reader(qbar, state, 4, update, readout)
    reader.__class__ = v3.UntiedReader
    reader.untied_Rq = torch.nn.ParameterList([torch.nn.Parameter(reader.Rq.detach().clone()) for _ in range(3)])
    reader.untied_Ro = torch.nn.ParameterList([torch.nn.Parameter(reader.Ro.detach().clone()) for _ in range(3)])
    untied = reader(qbar, state, 4, update, readout)
    assert torch.equal(untied, shared) and reader.last_read_calls == 4
    untied.square().sum().backward()
    grads = [reader.Rq.grad] + [p.grad for p in reader.untied_Rq]
    assert all(g is not None and torch.count_nonzero(g) for g in grads)
    assert not torch.equal(grads[0], grads[1])
    for bad in ({'steps': 5}, {'fixed_query': True}):
        with pytest.raises(ValueError):
            reader(qbar, state, **{'steps': 2, **bad})
    with torch.no_grad():
        reader.untied_Rq[0].add_(1.)
    assert torch.equal(reader(qbar, state, 1), LatentReader.forward(reader, qbar, state, 1))
    assert not torch.equal(reader(qbar, state, 2), LatentReader.forward(reader, qbar, state, 2))


def test_fixed_query_reader_is_the_shared_option_and_its_magnitude():
    state, qbar, generator = small_state(1)
    reader = perturbed_reader(8, generator)
    expected = reader(qbar, state, 4, fixed_query=True)
    one = reader(qbar, state, 1)
    reader.__class__ = v3.FixedQueryReader
    assert torch.equal(reader(qbar, state, 4), expected)
    # MSG-0549: Z = S * eta * a^(0) Ro for accumulate/increment.
    assert torch.allclose(reader(qbar, state, 4), 4*one, rtol=1e-5, atol=1e-6)


def test_tokenwise_kernel_matches_a_double_reference():
    generator = torch.Generator().manual_seed(2)
    b, t, r = 2, 5, 8
    q, k, v = (torch.randn(b, t, r, generator=generator, dtype=torch.float64) for _ in range(3))
    reader = perturbed_reader(8, generator).double()

    def reference(steps):
        nk = k / torch.sqrt(k.square().mean(-1, keepdim=True) + 1e-6)
        z = q
        for _ in range(steps):
            nz = z / torch.sqrt(z.square().mean(-1, keepdim=True) + 1e-6)
            pre = nz @ reader.Rq + nk
            z = z + .25 * ((pre * torch.sigmoid(pre) * v) @ reader.Ro)
        return z - q

    for steps in (1, 2, 4, 8):
        assert torch.allclose(v3.tokenwise_loop(q, k, v, reader, steps), reference(steps), rtol=1e-12, atol=1e-12)


def test_widen_embeds_the_start_and_initializes_new_dims():
    generator = torch.Generator().manual_seed(3)
    d = 16
    weights = {'projection.A_q': torch.randn(d, 8, generator=generator), 'projection.A_k': torch.randn(d, 8, generator=generator),
               'projection.A_v': torch.randn(d, 8, generator=generator), 'reader.Rq': torch.randn(8, 8, generator=generator),
               'reader.Ro': torch.randn(8, 8, generator=generator), 'fusion.P': torch.randn(8, 12, generator=generator)}
    wide = v3.widen(weights, 11, (d, d, d))
    again = v3.widen(weights, 11, (d, d, d))
    assert set(again) == set(weights) and all(torch.equal(wide[n], again[n]) for n in wide)
    for name in ('projection.A_q', 'projection.A_k', 'projection.A_v'):
        assert wide[name].shape == (d, 11) and torch.equal(wide[name][:, :8], weights[name])
        assert torch.count_nonzero(wide[name][:, 8:]) == 3*d
    assert not torch.equal(wide['projection.A_q'][:, 8:], wide['projection.A_k'][:, 8:])
    for name in ('reader.Rq', 'reader.Ro'):
        assert torch.equal(wide[name], torch.block_diag(weights[name], torch.eye(3)))
    assert torch.equal(wide['fusion.P'][:8], weights['fusion.P']) and not wide['fusion.P'][8:].any()
    other = v3.widen(weights, 11, (d, d, d), seed_parts=('other',))
    assert not torch.equal(other['projection.A_q'], wide['projection.A_q'])
    with pytest.raises(ValueError):
        v3.widen(weights, 8, (d, d, d))


def test_install_variant_rejects_combinations_and_bad_configs():
    class Side:
        def __init__(self, cfg):
            self.config, self.reader = cfg, LatentReader(4)
    from v2_memory import config
    cfg = replace(config('direct_hidden', 'soft_slot', 'F_out'), steps=4)
    with pytest.raises(ValueError):
        v3.install_variant(Side(cfg), v3.Variant(fixed_query=True, untied_steps=4))
    with pytest.raises(ValueError):
        v3.install_variant(Side(cfg), v3.Variant(untied_steps=2))
    with pytest.raises(ValueError):
        v3.install_variant(Side(replace(cfg, update='anchored')), v3.Variant(kernel='tokenwise'))


def test_align_is_right_aligned_and_clamped():
    donor = torch.arange(2*4).float().reshape(2, 4, 1)
    donor_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    recipient = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    aligned = v3._align(donor, donor_mask, recipient).squeeze(-1)
    # row 0: donor length 3, recipient 4 -> positions -1,0,1,2 clamped; row 1: donor 4, recipient 2 -> 2,3.
    assert aligned[0, :4].tolist() == [0., 0., 1., 2.] and aligned[1, :2].tolist() == [6., 7.]


def test_matched_donors_share_hop_and_question_length():
    rows = fake_rows(9)
    donors = v3.matched_donors(rows)
    for i, j in enumerate(donors):
        if j is not None:
            assert j != i and rows[j]['axes']['hop'] == rows[i]['axes']['hop']
            assert len(rows[j]['question_input_ids']) == len(rows[i]['question_input_ids'])
    assert v3.matched_donors(fake_rows(1)) == [None]


# ---- rules and bootstrap -------------------------------------------------------------------------

def counts(hop1, hop2, hop3, n=(334, 333, 333)):
    return {f'hop{h}': {'correct_count': c, 'count': m} for h, c, m in zip((1, 2, 3), (hop1, hop2, hop3), n)}


def matrix(**overrides):
    base = {arm: counts(200, 100, 100) for arm in v3.ARMS}
    base.update(overrides)
    return base


def test_decide_gl_m_and_s_m_star_rules():
    good = counts(200, 120, 120)  # +6.006 pp primary
    decision = v3.decide(matrix(**{'V3-03': good}))
    assert decision['S_M*'] == 'V3-03' and decision['GL_M'] is True and 'GL_M holds' in decision['outcome']
    # Rule 3: within 0.5 pp the S2 arm is chosen; (e) is deferred at S2.
    near = v3.decide(matrix(**{'V3-02': counts(200, 120, 120), 'V3-03': counts(200, 121, 120)}))
    assert near['S_M*'] == 'V3-02' and near['checks']['e']['passed'] is None and near['GL_M'] is True
    # (e) fails when the fixed-query control is as good as S4.
    e_fail = v3.decide(matrix(**{'V3-03': good, 'V3-05': good}))
    assert e_fail['GL_M'] is False and e_fail['checks']['e']['note'] == 'no evidence of an iterated query yet'
    # (b) hop1 drop below -1 pp.
    b_fail = v3.decide(matrix(**{'V3-03': counts(195, 120, 120)}))
    assert b_fail['checks']['b']['passed'] is False and 'descriptive' in b_fail['outcome']
    # (c) does not bind when V3-07 is below V3-01; it binds otherwise.
    loose = v3.decide(matrix(**{'V3-03': good, 'V3-07': counts(200, 90, 90)}))
    assert loose['checks']['c']['binding'] is False and loose['checks']['c']['passed'] is True and loose['GL_M']
    tight = v3.decide(matrix(**{'V3-03': good, 'V3-07': counts(200, 115, 115)}))
    assert tight['checks']['c']['binding'] is True and tight['GL_M'] is False
    # Rule 7: no gain over V3-01.
    assert 'challenges' in v3.decide(matrix())['outcome']
    # Thresholds include equality on exact fractions: exactly +3 pp passes.
    n = 300
    exact = {arm: counts(150, 30, 30, (300, n, n)) for arm in v3.ARMS}
    exact['V3-03'] = counts(150, 39, 39, (300, n, n))
    assert v3.decide(exact)['checks']['a']['difference_pp'] == 3 and v3.decide(exact)['GL_M'] is True


def test_bootstrap_is_paired_shared_and_deterministic():
    generator = torch.Generator().manual_seed(4)
    correct = {arm: {h: torch.randint(2, (40,), generator=generator).tolist() for h in (1, 2, 3)} for arm in ('A', 'B')}
    correct['C'] = correct['A']
    first = v3.bootstrap(correct, [('A', 'B'), ('C', 'A')], resamples=200)
    assert first == v3.bootstrap(correct, [('A', 'B'), ('C', 'A')], resamples=200)
    assert all(v == {'low': 0., 'high': 0.} for v in first['C - A'].values())
    assert set(first['A - B']) == {'hop1', 'hop2', 'hop3', 'primary'}
    with pytest.raises(ValueError):
        v3.bootstrap({'A': correct['A'], 'B': {1: [1], 2: [1], 3: [1]}}, [('A', 'B')], resamples=10)


# ---- provenance and CLI ----------------------------------------------------------------------------

def write_start(tmp_path, **changes):
    from v2r16_train import checkpoint_records
    run = tmp_path/'start'
    (run/'adapter').mkdir(parents=True)
    for name in ('adapter.pt', 'config.json'):
        (run/'adapter'/name).write_bytes((D5/'adapter'/name).read_bytes())
    (run/'adapter'/'variant.json').write_text(json.dumps(asdict(v3.Variant())))
    manifest = {'status': 'complete', 'validator': {'passed': True}, 'arm': 'V3-S0', 'eligible_as_evidence': True,
                'contract': v3.CONTRACT, 'metrics': {'hop1_gate': {'passed': True}},
                'checkpoint_files': checkpoint_records(run/'adapter')}
    manifest.update(changes)
    (run/'manifest.json').write_text(json.dumps(manifest))
    return run


def test_start_provenance_requires_the_passed_gate_and_the_pin(tmp_path):
    sha = 'c4f3f9f8c24a55a89142cd251e845a39ac4c235fb22aa244afb8e7cc3425b794'  # the D5 file copied as a stand-in
    assert v3.start_provenance(write_start(tmp_path), sha)['expected_adapter_sha256'] == sha
    for changes in ({'metrics': {'hop1_gate': {'passed': False}}}, {'eligible_as_evidence': False},
                    {'contract': 'E-0022@1'}, {'arm': 'V3-01'}):
        with pytest.raises(ValueError):
            v3.start_provenance(write_start(tmp_path/str(len(str(changes))), **changes), sha)
    with pytest.raises(ValueError):
        v3.start_provenance(write_start(tmp_path/'x'), 'a'*64)
    with pytest.raises(ValueError):
        v3.start_provenance(write_start(tmp_path/'y'), None)


@pytest.mark.parametrize('argv', [
    ['arm', '--arm', 'V3-03'],
    ['start', '--arm', 'V3-03'],
    ['start', '--smoke-steps', '3'],
    ['analyze', '--eval-limit', '4'],
    ['intervene', '--arm-run', 'x'],
    ['arm', '--arm', 'V3-11', '--start-run', 'x', '--start-sha256', 'a'*64],
])
def test_cli_rejects_incomplete_or_mixed_options(monkeypatch, tmp_path, argv):
    monkeypatch.setattr(sys, 'argv', ['v3_train.py', *argv, '--output', str(tmp_path/'out')])
    with pytest.raises(SystemExit):
        v3.main()


# ---- model (GPU) ---------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)


@pytest.fixture(scope='module')
def dev(tok):
    rows = loop.load_t2(tok, 'dev')[1]
    return [r for h in (1, 2, 3) for r in [x for x in rows if x['axes']['hop'] == h][:8]]


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
    loss, _, support_counts, query_counts = train_forward(model, batch, support_fn=isolated_support)
    loss.backward()
    model.eval()
    assert support_counts[14]['side']['reads'] == 0 and query_counts[14]['side']['writes'] == 0
    return {n: p.grad.clone() for n, p in model.adapters()[14].side.named_parameters()}


def test_v3_01_is_the_stock_start_and_variants_are_scoped(base, tok, dev):
    from v2_train import adapted
    batch = gpu_batch(tok, dev[:16])
    with adapted(base, checkpoint=D5/'adapter') as model, torch.no_grad():
        model.eval()
        stock = native_logits(model, batch)
    with v3.arm_model(base, 'V3-01', D5/'adapter') as model, torch.no_grad():
        model.eval()
        assert type(model.adapters()[14].side.reader) is LatentReader
        assert torch.equal(native_logits(model, batch), stock)
    # The base model is restored after every arm.
    assert all(not hasattr(layer, 'side') for layer in base.model.layers)


@pytest.mark.parametrize('arm', tuple(v3.ARMS))
def test_every_arm_trains_evaluates_and_reloads(base, tok, dev, arm, tmp_path):
    batch = gpu_batch(tok, dev[:8])
    with v3.arm_model(base, arm, D5/'adapter') as model:
        side = model.adapters()[14].side
        steps = v3.ARMS[arm]['steps']
        assert model.config.steps == steps and side.reader.eta == v3.ARMS[arm].get('eta', .25)
        for parameter in model.adapter_parameters():
            parameter.requires_grad_(True)
        grads = side_gradients(model, batch)
        expected = set(PARAMETERS)
        if arm == 'V3-08':
            expected |= {f'reader.untied_{m}.{s}' for m in ('Rq', 'Ro') for s in range(3)}
        assert set(grads) == expected
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in grads.values()), arm
        assert all(p.grad is None for n, p in model.model.named_parameters() if '.side.' not in n)
        model.zero_grad(set_to_none=True)
        records, reconstruction = v3.evaluate_arm(model, dev, tok.pad_token_id, steps)
        print(arm, 'reconstruction', reconstruction)
        for r in records['normal']:
            assert len(r['state_norm_last']) == steps and r['query_cosine_last'][0] == pytest.approx(1., abs=1e-6)
            assert r['query_cosine_prev_all'][0] is None and len(r['query_cosine_prev_last']) == steps
            if arm == 'V3-05':  # the fixed query does not move
                assert all(c == pytest.approx(1., abs=1e-6) for c in r['query_cosine_prev_last'][1:])
        if arm == 'V3-06':
            # No memory is read: every control gives the normal predictions.
            for m in records:
                assert [(x['prediction'], x['nll']) for x in records[m]] == \
                       [(x['prediction'], x['nll']) for x in records['normal']], m
            assert records['normal'][0]['path_weight_all'] is None
        else:
            assert all(len(r['path_weight_all']) == steps for r in records['normal'])
        scan = v3.UNTIED_S_EVAL if arm == 'V3-08' else v3.S_EVAL
        for s in scan:
            v3.evaluate_arm(model, dev[:4], tok.pad_token_id, s, ('normal',))
        if arm == 'V3-08':
            with pytest.raises(ValueError):
                v3.evaluate_arm(model, dev[:4], tok.pad_token_id, 8, ('normal',))
        assert model.config.steps == steps
        v3.export_arm(model, tmp_path/'adapter')
    with v3.load_arm(base, tmp_path/'adapter') as reloaded:
        assert v3.side_variant(reloaded.adapters()[14].side) == v3.arm_variant(arm)
        again, _ = v3.evaluate_arm(reloaded, dev, tok.pad_token_id, steps)
        assert again == records


def test_v3_05_is_the_exact_magnitude_control(base, tok, dev):
    """V3-05 equals the stock reader called with LatentReader's own fixed_query=True at S4,
    and differs from the shared S4 loop (V3-03) on the same weights."""
    from v2_train import adapted
    batch = gpu_batch(tok, dev[:8])
    with v3.arm_model(base, 'V3-05', D5/'adapter') as model, torch.no_grad():
        model.eval()
        fixed = native_logits(model, batch)
    with v3.arm_model(base, 'V3-03', D5/'adapter') as model, torch.no_grad():
        model.eval()
        shared = native_logits(model, batch)
    with adapted(base, checkpoint=D5/'adapter') as model, torch.no_grad():
        model.eval()
        side = model.adapters()[14].side
        calls = []
        original = side.reader.forward

        def forward(qbar, state, steps=1, update='accumulate', readout='increment', fixed_query=False):
            calls.append((steps, fixed_query))
            return original(qbar, state, 4, update, readout, True)
        side.reader.forward = forward
        try:
            forced = native_logits(model, batch)
        finally:
            del side.reader.forward
    assert calls and all(c == (1, False) for c in calls)  # SideBranch passes no fixed_query
    assert torch.equal(forced, fixed) and not torch.equal(fixed, shared)


def test_initial_checks_and_reviewed_identity(base, tok, dev, tmp_path):
    from v2_memory import collate
    with v3.arm_model(base, 'V3-08', D5/'adapter') as model:
        result, _ = v3._initial_checks('V3-08', model, dev, tok.pad_token_id)
        assert result['untied_vs_shared_s4']['identical'] is True
    from v2_train import adapted
    with adapted(base, checkpoint=D5/'adapter') as start:
        start.eval()
        batch = collate(dev[:16], tok.pad_token_id, 'cuda')
        with torch.no_grad():
            start_logits, _, _ = v3._capture(start, batch, loop._snapshots(start, batch, 'normal', None))
    with v3.arm_model(base, 'V3-07', D5/'adapter') as model:
        result, logits = v3._initial_checks('V3-07', model, dev, tok.pad_token_id)
        gap = float((logits-start_logits).abs().max())
        print('V3-07 step-0 gap', gap, result)
        assert 0 < gap < float('inf') and 0 <= result['widened_step0']['dev_hop1_em'] <= 1
        # The widened S=1 arm equals the reviewed evaluator example by example.
        from v2_memory import isolated_support
        from v2r16_memory import evaluate
        _, reviewed = evaluate(model, dev, tok.pad_token_id, tmp_path/'reviewed', 'dev',
                               ('normal', 'zero', 'shuffled', 'permuted'), None, support_fn=isolated_support)
        mine, _ = v3.evaluate_arm(model, dev, tok.pad_token_id, 1)
        for mode in mine:
            loop.attach_paths(mine[mode], dev)
            assert loop.compare_to_reviewed(mine[mode], reviewed[mode])['identical'], mode


def test_interventions_none_is_native_and_replacements_act(base, tok, dev):
    with v3.arm_model(base, 'V3-03', D5/'adapter') as model:
        summary, records, checks = v3.intervene(model, dev, tok.pad_token_id, 4)
    assert checks['none_vs_native_prediction_mismatches'] == 0
    assert set(summary) == {'1', '2', '3'} and set(summary['1']) == {'none', 'random', 'unrelated', 'matched'}
    for s in (1, 2, 3):
        assert len(records[s]['none']) == len(dev) == len(records[s]['random'])
        assert len(records[s]['matched']) == len(dev) - checks['rows_without_matched_donor']
        assert any(a['nll'] != b['nll'] for a, b in zip(records[s]['none'], records[s]['random']))
    # The same seed gives the same random replacement.
    with v3.arm_model(base, 'V3-03', D5/'adapter') as model:
        again, _, _ = v3.intervene(model, dev, tok.pad_token_id, 4)
    assert again == summary


def test_t2_hop1_training_rows_and_schedules(tok):
    _, rows, _ = loop.load_t2(tok, 'train')
    hop1 = [r for r in rows if r['axes']['hop'] == 1]
    start, train = v3.v3_schedule(hop1, 'start'), v3.v3_schedule(rows, 'train')
    print('hop1 train rows', len(hop1), 'start epochs', start['epochs'], 'train epochs', train['epochs'],
          'tokens/step', start['tokens']/6000, train['tokens']/6000)
    assert len(rows) == 18432 and all(hop1[i]['axes']['hop'] == 1 for b in start['batches'] for i in b)
    assert start['steps'] == train['steps'] == 6000 and start['warmup_steps'] == train['warmup_steps'] == 300
    # The analysis reference is the same released data and the same shared schedule.
    reference = v3.analysis_reference()
    assert reference['order_sha256'] == train['order_sha256'] and reference['tokens'] == train['tokens']
    assert len(reference['dev']) == 1000 and [sum(h == x for _, h, _ in reference['dev']) for x in (1, 2, 3)] == [334, 333, 333]


def write_ready(tmp_path, start, **changes):
    run = tmp_path/'ready'
    run.mkdir(parents=True)
    start_manifest = json.loads((start/'manifest.json').read_text())
    manifest = {'status': 'complete', 'validator': {'passed': True}, 'contract': 'E-0023@1', 'command': 'ready',
                'eligible_as_evidence': True, 'smoke_only': False,
                'start': {'checkpoint_files': start_manifest['checkpoint_files']},
                'metrics': {'readiness_passed': True, 'readiness_criteria': {'hop1_em_margin': True}}}
    manifest.update(changes)
    (run/'manifest.json').write_text(json.dumps(manifest))
    return run


def test_old_gate_failure_is_a_formal_start_only_with_a_passed_readiness_run(tmp_path):
    """E-0023: ATT-0196-1 failed the old all-position gate; rule R on calibration releases it."""
    sha = 'c4f3f9f8c24a55a89142cd251e845a39ac4c235fb22aa244afb8e7cc3425b794'
    start = write_start(tmp_path, contract='E-0022@2', metrics={'hop1_gate': {'passed': False}})
    with pytest.raises(ValueError):
        v3.start_provenance(start, sha)
    record = v3.start_provenance(start, sha, ready_run=write_ready(tmp_path/'ok', start))
    assert record['formal_start'] is True and record['hop1_gate_passed'] is False
    assert record['readiness']['criteria'] == {'hop1_em_margin': True}
    other = {'adapter.pt': {'sha256': 'b'*64, 'bytes': 1}}
    for i, changes in enumerate(({'metrics': {'readiness_passed': False}}, {'eligible_as_evidence': False},
                                 {'smoke_only': True}, {'contract': 'E-0022@2'}, {'command': 'arm'},
                                 {'status': 'failed'}, {'start': {'checkpoint_files': other}})):
        with pytest.raises(ValueError):
            v3.start_provenance(start, sha, ready_run=write_ready(tmp_path/f'bad{i}', start, **changes))
    # A smoke V3-S0 start is not made formal by a readiness run.
    smoke = write_start(tmp_path/'smoke', eligible_as_evidence=False, smoke_only=True)
    assert v3.start_provenance(smoke, sha, smoke=True, ready_run=write_ready(tmp_path/'s', smoke))['formal_start'] is False


def test_smoke_start_is_accepted_only_by_smoke_jobs(tmp_path):
    sha = 'c4f3f9f8c24a55a89142cd251e845a39ac4c235fb22aa244afb8e7cc3425b794'
    smoke = write_start(tmp_path, eligible_as_evidence=False, smoke_only=True, metrics={'hop1_gate': {'passed': False}})
    with pytest.raises(ValueError):
        v3.start_provenance(smoke, sha)
    assert v3.start_provenance(smoke, sha, smoke=True)['formal_start'] is False


SYN_START = {'expected_adapter_sha256': 'a'*64, 'formal_start': True, 'run': '/synthetic/start',
             'checkpoint_files': {'adapter.pt': {'sha256': 'a'*64, 'bytes': 1}}}


def synthetic_reference(n=4):
    return {'metadata_sha256': 'm'*64, 'train_sha256': 't'*64, 'dev_sha256': 'd'*64, 'order_sha256': 'o'*64,
            'tokens': 4800000, 'order_seed': 1,
            'dev': [(f'h{h}-{i}', h, 100+i) for h in (1, 2, 3) for i in range(n)]}


def refresh_manifest(run):
    from v2r16_train import file_record
    manifest = json.loads((run/'manifest.json').read_text())
    manifest['artifacts'] = [{'path': str(p.resolve()), 'role': str(p.relative_to(run)), **file_record(p)}
                             for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    (run/'manifest.json').write_text(json.dumps(manifest))


def synthetic_arm(directory, arm, hop_correct, reference):
    """A complete formal arm run on the reference dev set; the first hop_correct[h-1] rows of hop h are correct."""
    from v2_memory import config
    steps = v3.ARMS[arm]['steps']
    (directory/f'eval-s{steps}').mkdir(parents=True)
    records, seen = [], Counter()
    for episode, hop, target in reference['dev']:
        right = seen[hop] < hop_correct[hop-1]
        seen[hop] += 1
        prediction = target if right else target+1000
        records.append({'episode_id': episode, 'hop': hop, 'target': target, 'prediction': prediction,
                        'correct': prediction == target, 'steps': steps, 'nll': 1.})
    (directory/f'eval-s{steps}'/'dev.normal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    hops = {}
    for h in (1, 2, 3):
        group = [r for r in records if r['hop'] == h]
        hops[f'hop{h}'] = {'count': len(group), 'correct_count': sum(r['correct'] for r in group),
                           'em': sum(r['correct'] for r in group)/len(group), 'nll': 1., 'prior_em': 0.}
    metrics = {
        'config': asdict(v3.arm_config(config('direct_hidden', 'soft_slot', 'F_out'), arm)),
        'variant': asdict(v3.arm_variant(arm)), 'start': dict(SYN_START),
        'schedule': {'steps': 6000, 'stage': 'train', 'order_sha256': reference['order_sha256'], 'tokens': reference['tokens']},
        'train': {'steps': 6000, 'tokens': reference['tokens'], 'stored_parameters': 1, 'stable_step_seconds': .2},
        'data': {'metadata_sha256': reference['metadata_sha256'], 'dev_sha256': reference['dev_sha256'],
                 'train_sha256': reference['train_sha256'], 'dev_rows': len(reference['dev'])},
        'checkpoint_reload': {'full_records_identical': True, 'steps': steps},
        'evaluation': {str(steps): {'normal': hops, 'zero': hops}}, 'node_hours': .1,
        'macs': v3.macs_per_question_token(arm), 'reads_per_question_token': steps}
    (directory/'metrics.json').write_text(json.dumps(metrics))
    (directory/'manifest.json').write_text(json.dumps({
        'status': 'complete', 'validator': {'passed': True}, 'eligible_as_evidence': True, 'smoke_only': False,
        'smoke_steps': 0, 'eval_limit': 0, 'contract': v3.CONTRACT, 'command': 'arm', 'arm': arm}))
    refresh_manifest(directory)
    return directory


def synthetic_matrix(tmp_path, reference, star=(2, 3, 3)):
    return [synthetic_arm(tmp_path/arm, arm, star if arm == 'V3-03' else (2, 1, 1), reference) for arm in v3.ARMS]


def test_analyze_validated_records_apply_the_rules(tmp_path):
    reference = synthetic_reference()
    result = v3.analyze_runs(synthetic_matrix(tmp_path, reference), reference, SYN_START)
    assert result['decision']['S_M*'] == 'V3-03' and result['decision']['GL_M'] is True
    assert result['decision']['primary_pp']['V3-03'] == 75. and result['decision']['primary_pp']['V3-01'] == 25.
    assert 'V3-03 - V3-01' in result['bootstrap']['intervals_pp'] and set(result['arms']) == set(v3.ARMS)
    runs = synthetic_matrix(tmp_path/'again', reference)
    with pytest.raises(ValueError):
        v3.analyze_runs(runs[:9], reference, SYN_START)
    with pytest.raises(ValueError):
        v3.analyze_runs(runs + runs[:1], reference, SYN_START)


def corrupt_metrics(run, change):
    metrics = json.loads((run/'metrics.json').read_text())
    change(metrics)
    (run/'metrics.json').write_text(json.dumps(metrics))
    refresh_manifest(run)


def corrupt_records(run, change):
    path = next(run.glob('eval-s*'))/'dev.normal.jsonl'
    records = [json.loads(line) for line in path.open()]
    records = change(records)
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    refresh_manifest(run)


def hop(metrics, h):
    return next(iter(metrics['evaluation'].values()))['normal'][f'hop{h}']


CORRUPTIONS = {
    # RV-0058 counterexample: only the stored denominators change.
    'denominator': lambda r: corrupt_metrics(r, lambda m: [hop(m, h).update(count=3) for h in (2, 3)]),
    'correct_count': lambda r: corrupt_metrics(r, lambda m: hop(m, 2).update(correct_count=4)),
    'stored_em': lambda r: corrupt_metrics(r, lambda m: hop(m, 3).update(em=1.)),
    'start_digest': lambda r: corrupt_metrics(r, lambda m: m['start'].update(expected_adapter_sha256='b'*64)),
    'nonformal_start': lambda r: corrupt_metrics(r, lambda m: m['start'].update(formal_start=False)),
    'order': lambda r: corrupt_metrics(r, lambda m: m['schedule'].update(order_sha256='x'*64)),
    'steps': lambda r: corrupt_metrics(r, lambda m: m['train'].update(steps=5999)),
    'smoke_schedule': lambda r: corrupt_metrics(r, lambda m: m['schedule'].update(smoke_steps=3)),
    'dev_digest': lambda r: corrupt_metrics(r, lambda m: m['data'].update(dev_sha256='y'*64)),
    'config': lambda r: corrupt_metrics(r, lambda m: m['config'].update(eta=.5)),
    'reload': lambda r: corrupt_metrics(r, lambda m: m['checkpoint_reload'].update(full_records_identical=False)),
    'identity': lambda r: corrupt_records(r, lambda rs: [dict(rs[0], episode_id='other'), *rs[1:]]),
    'hop': lambda r: corrupt_records(r, lambda rs: [dict(rs[0], hop=2), *rs[1:]]),
    'target': lambda r: corrupt_records(r, lambda rs: [dict(rs[0], target=rs[0]['target']+1), *rs[1:]]),
    'correct_flag': lambda r: corrupt_records(r, lambda rs: rs[:-1] + [dict(rs[-1], correct=True)]),
    'truncated': lambda r: corrupt_records(r, lambda rs: rs[:-1]),
    'unrecorded_edit': lambda r: (r/'metrics.json').write_text((r/'metrics.json').read_text() + ' '),
}


@pytest.mark.parametrize('name', sorted(CORRUPTIONS))
def test_analyze_rejects_each_corruption(tmp_path, name):
    reference = synthetic_reference()
    runs = synthetic_matrix(tmp_path, reference)
    CORRUPTIONS[name](runs[2])  # V3-03
    with pytest.raises(ValueError):
        v3.analyze_runs(runs, reference, SYN_START)


def test_analyze_rejects_runs_consistent_with_each_other_but_not_the_released_dev(tmp_path):
    reference = synthetic_reference()
    short = dict(reference, dev=reference['dev'][:-1])
    runs = synthetic_matrix(tmp_path, short)  # every arm agrees on a truncated dev set
    with pytest.raises(ValueError):
        v3.analyze_runs(runs, reference, SYN_START)
    wrong = dict(reference, dev=[(e, h, t+1) for e, h, t in reference['dev']])
    with pytest.raises(ValueError):
        v3.analyze_runs(synthetic_matrix(tmp_path/'wrong', wrong), reference, SYN_START)


def test_query_cosines_first_and_adjacent():
    q1 = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    q2 = torch.tensor([[[1., 1.], [0., 1.], [1., 0.]]])
    q3 = torch.tensor([[[0., 1.], [-1., 1.], [1., 0.]]])
    mask = torch.tensor([[True, True, False]])  # the third position is padding
    result = v3.query_cosines([q1, q2, q3], mask)
    assert result[0]['query_cosine_prev_all'] is None and result[0]['query_cosine_prev_last'] is None
    assert result[0]['query_cosine_all'].item() == pytest.approx(1.)
    c = 2 ** -.5
    assert result[2]['query_cosine_all'].item() == pytest.approx((0. + -c)/2)   # q3 vs q1
    assert result[2]['query_cosine_prev_all'].item() == pytest.approx((c + c)/2)  # q3 vs q2
    assert result[2]['query_cosine_prev_last'].item() == pytest.approx(c) and result[2]['query_cosine_last'].item() == pytest.approx(-c)
    fixed = v3.query_cosines([q1, q1, q1], mask)
    assert all(e['query_cosine_prev_all'].item() == pytest.approx(1.) for e in fixed[1:])


def test_summary_weights_adjacent_cosine_by_question_tokens():
    def record(tokens, prev):
        return {'episode_id': str(tokens), 'hop': 1, 'support_order': 'natural', 'target': 1, 'prediction': 1,
                'correct': True, 'nll': 0., 'question_tokens': tokens, 'steps': 2, 'output_norm_last': 1.,
                'query_cosine_all': [1., .5], 'query_cosine_last': [1., .5], 'query_cosine_prev_all': [None, prev],
                'query_cosine_prev_last': [None, prev], 'state_norm_all': [1., 1.], 'state_norm_last': [1., 1.],
                'read_norm_all': [1., 1.], 'read_norm_last': [1., 1.], 'n_eff_all': None, 'n_eff_last': None,
                'argmax_last': None, 'path_weight_all': None, 'path_weight_last': None}
    summary = v3.summarize_arm([record(1, 0.), record(3, 1.)])['all']['per_step']
    assert summary[0]['query_cosine_prev_all'] is None
    assert summary[1]['query_cosine_prev_all'] == pytest.approx(.75) and summary[1]['query_cosine_prev_last'] == pytest.approx(.5)
