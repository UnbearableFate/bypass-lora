"""Explicit causal-slot checks for normalized RB read diagnostics."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from bypass_core import LatentReader, kernel_prefix_states, read_kernel_prefix, process_prefix
from bypass_model import AdapterConfig
from rb_history import run_history
from rb_read_diagnostics import prefix_read_metrics, ReadSummary


def test_uniform_reads_have_expected_entropy_effective_count_and_current_weight():
    query = torch.ones(1, 5, 3)
    valid = torch.tensor([[True, True, True, False, False]])
    keys = query*valid[..., None]
    got = prefix_read_metrics(query, keys, valid, chunk_size=2)
    torch.testing.assert_close(got['n_eff'], torch.tensor([[1., 2., 3., 0., 0.]]))
    torch.testing.assert_close(got['entropy'][0, :3], torch.arange(1, 4).float().log())
    torch.testing.assert_close(got['current_weight'][0, :3], 1/torch.arange(1, 4).float())
    torch.testing.assert_close(got['relative_n_eff'], valid.float())


@pytest.mark.parametrize('kind', [None, 'iv_window1', 'iv_window8', 'iv_strict', 'iv_permute', 'iv_swap'])
def test_read_statistics_equal_independent_explicit_distribution(kind):
    torch.manual_seed(17)
    keys = torch.rand(2, 11, 4)+.1
    query = torch.rand_like(keys)+.1
    valid = torch.arange(11)[None] < torch.tensor([[9], [11]])
    donor = {'keys': torch.rand(2, 7, 4)+.1,
             'valid': torch.arange(7)[None] < torch.tensor([[7], [5]])}
    got = prefix_read_metrics(query, keys, valid, kind=kind, donor=donor, chunk_size=3)
    for b in range(2):
        for t in range(11):
            if not valid[b, t]:
                assert all(value[b, t] == 0 for value in got.values())
                continue
            source = donor['keys'][b] if kind == 'iv_swap' else keys[b]
            mask = donor['valid'][b] if kind == 'iv_swap' else valid[b]
            indices = [i for i in range(len(source)) if mask[i] and
                       (i < t if kind in ('iv_strict', 'iv_swap') else i <= t) and
                       (kind != 'iv_window1' or i == t) and
                       (kind != 'iv_window8' or i > t-8)]
            selected = [source[i] for i in indices]
            if kind == 'iv_swap':
                selected.append(keys[b, t])
            if not selected:
                assert all(value[b, t] == 0 for value in got.values())
                continue
            scores = torch.stack(selected) @ query[b, t]
            prob = scores/scores.sum()
            neff = 1/prob.square().sum()
            own = 0. if kind == 'iv_strict' else prob[-1]
            torch.testing.assert_close(got['current_weight'][b, t], torch.as_tensor(own))
            torch.testing.assert_close(got['n_eff'][b, t], neff)
            torch.testing.assert_close(got['relative_n_eff'][b, t], neff/len(selected))
            torch.testing.assert_close(got['entropy'][b, t], -(prob*prob.log()).sum())
            # Position-0 shares index the history's first token (the donor's under swap).
            first = [j for j, i in enumerate(indices) if i < 3]
            share0 = prob[indices.index(0)] if 0 in indices else torch.tensor(0.)
            torch.testing.assert_close(got['share0'][b, t], share0)
            torch.testing.assert_close(got['share0_2'][b, t], prob[first].sum())
            argmax0 = 0 in indices and int(prob.argmax()) == indices.index(0)
            assert got['argmax0'][b, t] == float(argmax0)
            torch.testing.assert_close(got['uniform0'][b, t], torch.tensor((0 in indices)/len(selected)))
            torch.testing.assert_close(got['uniform0_2'][b, t], torch.tensor(len(first)/len(selected)))


def test_observer_sees_native_first_and_last_queries_without_changing_outputs_or_gradients():
    torch.manual_seed(91)
    q, k, v = [torch.randn(1, 7, 4, requires_grad=True) for _ in range(3)]
    reader = LatentReader(4)
    valid = torch.ones(1, 7, dtype=torch.bool)
    keys, _, matrices, masses, counts, _ = kernel_prefix_states(k, v, reader, valid)
    kwargs = dict(steps=4, update='accumulate', readout='increment')
    normal, _ = read_kernel_prefix(q, reader, matrices, masses, counts, valid, **kwargs)
    seen = []
    observed, _ = read_kernel_prefix(q, reader, matrices, masses, counts, valid,
        observe_read=lambda query, step: seen.append((step, query.detach().clone())), **kwargs)
    assert torch.equal(observed, normal)
    assert [step for step, _ in seen] == [1, 2, 3, 4]
    assert not torch.equal(seen[0][1], seen[-1][1])
    params = (q, k, v, reader.Rq, reader.Ro)
    ga = torch.autograd.grad(normal.sum(), params, retain_graph=True)
    gb = torch.autograd.grad(observed.sum(), params)
    assert all(torch.equal(a, b) for a, b in zip(ga, gb))
    cfg = AdapterConfig(branch='memory', rank=4, steps=4, impl='prefix')
    with torch.no_grad():
        traced = run_history(q, k, v, reader, valid, cfg, {'trace_reads': True})
        plain = run_history(q, k, v, reader, valid, cfg, {})
    assert torch.equal(plain.outputs, traced.outputs)
    assert [r['step'] for r in traced.diagnostics['read_metrics']] == [1, 4]
    fixed = replace(cfg, fixed_query=True)
    with torch.no_grad():
        records = run_history(q, k, v, reader, valid, fixed, {'trace_reads': True}).diagnostics['read_metrics']
    assert torch.equal(records[0]['n_eff'], records[1]['n_eff'])


def test_gold_answer_query_scope_excludes_padding_and_uses_preceding_position():
    valid = torch.tensor([[1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, 7, -100]])
    record = {'step': 1, 'n_eff': torch.tensor([[1., 2., 3., 99.]]),
              'readable_positions': torch.tensor([[1., 2., 3., 0.]])}
    summary = ReadSummary()
    summary.add('piqa', {6: {'read_metrics': [record]}}, valid, labels)
    report = {r['scope']: r for r in summary.finish()}
    assert report['all_valid_input']['count'] == 3
    assert report['all_valid_input']['means']['n_eff'] == 2
    assert report['answer_prediction']['count'] == 1
    assert report['answer_prediction']['means']['readable_positions'] == 2


def test_read_summary_reports_nonempty_means_without_changing_original_means():
    valid = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    labels = torch.full((1, 3), -100)
    record = {
        'step': 1,
        'current_weight': torch.tensor([[0., 0., 0.]]),
        'n_eff': torch.tensor([[0., 1., 2.]]),
        'entropy': torch.tensor([[0., 0., .69314718]]),
        'relative_n_eff': torch.tensor([[0., 1., 1.]]),
        'readable_positions': torch.tensor([[0., 1., 2.]]),
    }
    summary = ReadSummary()
    summary.add('piqa', {6: {'read_metrics': [record]}}, valid, labels)
    result = summary.finish()[0]
    assert result['count'] == 3
    assert result['empty_reads'] == 1
    assert result['means']['n_eff'] == 1
    assert result['nonempty_means']['n_eff'] == 1.5
    assert result['means']['current_weight'] == 0
    assert result['nonempty_means']['current_weight'] == 0


@pytest.mark.parametrize('steps,mode,fixed', [(1, 'inclusive', False), (4, 'strict', False), (4, 'inclusive', True)])
def test_default_prefix_outputs_and_training_gradients_bitwise_match_reviewed_pre_observer_source(steps, mode, fixed):
    path = Path('/work/xg24i002/x10041/bypass-lora/snapshots/6df9e479b229feb3d5edfac588b06264612c7839/src/bypass_core.py')
    spec = importlib.util.spec_from_file_location('_rb1_pre_observer_core', path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    torch.manual_seed(331)
    inputs = [torch.randn(2, 17, 8).requires_grad_() for _ in range(3)]
    legacy_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    current_reader, old_reader = LatentReader(8), old.LatentReader(8)
    old_reader.load_state_dict(current_reader.state_dict())
    valid = torch.arange(17)[None] < torch.tensor([[17], [13]])
    kwargs = {'valid': valid, 'steps': steps, 'mode': mode, 'fixed_query': fixed}
    current = process_prefix(*inputs, current_reader, **kwargs)
    previous = old.process_prefix(*legacy_inputs, old_reader, **kwargs)
    assert torch.equal(current.outputs, previous.outputs)
    assert torch.equal(current.state.M, previous.state.M)
    assert torch.equal(current.state.c, previous.state.c)
    assert torch.equal(current.state.counts, previous.state.counts)
    ga = torch.autograd.grad(current.outputs.square().sum(), (*inputs, *current_reader.parameters()))
    gb = torch.autograd.grad(previous.outputs.square().sum(), (*legacy_inputs, *old_reader.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(ga, gb))


def test_sink_shares_on_a_constructed_first_token_sink():
    query = torch.ones(1, 6, 2)
    keys = torch.tensor([[[9., 9.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.]]])
    valid = torch.ones(1, 6, dtype=torch.bool)
    got = prefix_read_metrics(query, keys, valid)
    # At t=5 the scores are 18,1,1,1,1,1 over six positions.
    torch.testing.assert_close(got['share0'][0, 5], torch.tensor(18/23))
    torch.testing.assert_close(got['share0_2'][0, 5], torch.tensor(20/23))
    assert got['argmax0'][0].tolist() == [1.]*6
    torch.testing.assert_close(got['uniform0'][0], 1/torch.arange(1., 7.))
    window = prefix_read_metrics(query, keys, valid, kind='iv_window1')
    assert window['share0'][0, 1:].abs().max() == 0 and window['argmax0'][0, 1:].sum() == 0


@pytest.mark.parametrize('kind', [None, 'iv_strict', 'iv_window8', 'iv_permute'])
def test_read_vector_is_the_actual_normalized_read_before_the_exit(kind):
    torch.manual_seed(5)
    q, k, v = [torch.randn(2, 12, 4) for _ in range(3)]
    reader = LatentReader(4)
    with torch.no_grad():
        reader.Rq.copy_(torch.randn(4, 4))
        reader.Ro.copy_(torch.randn(4, 4))
    valid = torch.arange(12)[None] < torch.tensor([[12], [9]])
    cfg = AdapterConfig(branch='memory', rank=4, steps=1, impl='prefix')
    options = {'trace_reads': True}
    if kind:
        options['kind'] = kind
    if kind == 'iv_permute':
        options.update(phase='R1', example_ids=['a', 'b'])
        valid = torch.ones(2, 12, dtype=torch.bool)
    with torch.no_grad():
        result = run_history(q, k, v, reader, valid, cfg, options)
        plain = run_history(q, k, v, reader, valid, cfg, {key: value for key, value in options.items()
                                                          if key != 'trace_reads'})
    assert torch.equal(result.outputs, plain.outputs)
    read = result.diagnostics['read_metrics'][0]['read_vector']
    # S1 increment readout: z - z0 = eta * read @ Ro.
    torch.testing.assert_close(result.outputs, reader.eta*(read @ reader.Ro), rtol=1e-5, atol=1e-6)
    assert not read[~valid].any()
    if kind in (None, 'iv_strict', 'iv_window8'):
        # Independent slot form: kernel-weighted mean of the eligible raw values.
        from bypass_core import elu_plus_one, rmsnorm_no_affine
        query = elu_plus_one(rmsnorm_no_affine(q, reader.eps_norm) @ reader.Rq)
        key = elu_plus_one(rmsnorm_no_affine(k, reader.eps_norm))*valid[..., None]
        for b in range(2):
            for t in range(int(valid[b].sum())):
                lo = t-7 if kind == 'iv_window8' else 0
                hi = t if kind == 'iv_strict' else t+1
                n = torch.arange(max(lo, 0), hi)
                if not n.numel():
                    continue
                score = key[b, n] @ query[b, t]
                expected = (score[:, None]*v[b, n]).sum(0)/(score.sum()+reader.eps_read)
                torch.testing.assert_close(read[b, t], expected, rtol=1e-4, atol=1e-5)


def test_read_summary_never_averages_read_vectors():
    valid = torch.ones(1, 2, dtype=torch.bool)
    record = {'step': 1, 'n_eff': torch.tensor([[1., 2.]]), 'readable_positions': torch.tensor([[1., 2.]]),
              'read_vector': torch.ones(1, 2, 4)}
    summary = ReadSummary()
    summary.add('piqa', {6: {'read_metrics': [record]}}, valid, torch.full((1, 2), -100))
    assert all('read_vector' not in row['means'] for row in summary.finish())
