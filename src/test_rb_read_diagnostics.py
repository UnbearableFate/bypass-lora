"""Explicit causal-slot checks for normalized RB read diagnostics."""
from dataclasses import replace

import pytest
import torch

from bypass_core import LatentReader, kernel_prefix_states, read_kernel_prefix
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
    cfg = AdapterConfig(rank=4, steps=4, impl='prefix')
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
