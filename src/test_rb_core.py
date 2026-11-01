"""RB0 FP32 thresholds fixed before execution (coordinator MSG-0543).

Per tensor: ||actual-reference||inf <= 1e-5 + 1e-4*||reference||inf.
Both the absolute infinity norm and normalized relative error are reported.
No componentwise ratio or post-result tolerance change is used.
"""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from bypass_core import (LatentReader, MemoryState, elu_plus_one, rmsnorm_no_affine,
                         process_sequence, process_prefix, process_kernel_tokenwise)
from bypass_model import AdapterConfig
from rb_history import run_history, donor_indices

ATOL, RTOL = 1e-5, 1e-4


def close(actual, reference, record_property=None, name='error'):
    absolute = (actual-reference).abs().max().item() if actual.numel() else 0.
    scale = reference.abs().max().item() if reference.numel() else 0.
    relative = absolute/max(scale, 1e-8)
    if record_property:
        record_property(name+'.max_abs', absolute)
        record_property(name+'.relative_inf', relative)
    assert absolute <= ATOL+RTOL*scale, (name, absolute, relative, scale)


@pytest.mark.parametrize('length', [17, 64])
@pytest.mark.parametrize('rank', [8, 32])
@pytest.mark.parametrize('mode', ['inclusive', 'strict'])
@pytest.mark.parametrize('steps', [1, 2, 4])
@pytest.mark.parametrize('fixed_query', [False, True])
@pytest.mark.parametrize('update', ['accumulate', 'anchored'])
@pytest.mark.parametrize('readout', ['increment', 'full'])
def test_prefix_reference_slot_outputs_and_all_gradients(length, rank, mode, steps,
        fixed_query, update, readout, record_property):
    torch.manual_seed(19000+length+rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    originals = [torch.randn(3, length, rank) for _ in range(3)]
    probe = torch.randn_like(originals[0])
    valid = torch.ones(3, length, dtype=torch.bool)
    valid[0, :3], valid[1, -5:], valid[2] = False, False, False
    write, read = valid.clone(), valid.clone()
    write[0, 7], read[1, 8] = False, False
    prior_k, prior_v = torch.rand(3, rank), torch.randn(3, rank)
    template = LatentReader(rank)
    with torch.no_grad():
        template.Rq.add_(torch.randn_like(template.Rq)*.1)
        template.Ro.add_(torch.randn_like(template.Ro)*.1)
    results, gradients = [], []
    for implementation, backend in [('reference','kernel_matrix'), ('prefix','kernel_matrix'),
                                     ('reference','kernel_slot_reference')]:
        reader = LatentReader(rank)
        reader.load_state_dict(template.state_dict())
        q, k, v = [x.clone().requires_grad_() for x in originals]
        state = MemoryState.empty(3, rank, backend).write(prior_k, prior_v, torch.tensor([True, True, False]))
        result = process_sequence(q, k, v, reader, state=state, valid=valid, write=write,
            read=read, mode=mode, steps=steps, fixed_query=fixed_query, update=update,
            readout=readout, backend=backend, impl=implementation)
        grad = torch.autograd.grad((result.outputs*probe).sum(), (q,k,v,reader.Rq,reader.Ro))
        results.append(result)
        gradients.append(grad)
        assert result.accounting['writes'] == write.sum()
        assert result.accounting['reads'] == read.sum()*steps
        assert torch.equal(result.outputs[~read], torch.zeros_like(result.outputs[~read]))
    for index, label in [(1,'prefix'), (2,'slots')]:
        close(results[index].outputs, results[0].outputs, record_property, label+'.output')
        close(results[index].state.M, results[0].state.M, record_property, label+'.M')
        close(results[index].state.c, results[0].state.c, record_property, label+'.c')
        assert torch.equal(results[index].state.counts, results[0].state.counts)
        for name, actual, reference in zip(('q','k','v','Rq','Ro'), gradients[index], gradients[0]):
            close(actual, reference, record_property, label+'.grad.'+name)


@pytest.mark.parametrize('mode', ['inclusive','strict'])
def test_prefix_chunks_reordering_reset_and_no_host_scalars(mode):
    torch.manual_seed(191)
    q,k,v = [torch.randn(3,17,8) for _ in range(3)]
    valid = torch.ones(3,17,dtype=torch.bool)
    valid[1,-4:], valid[2] = False, False
    reader = LatentReader(8)
    whole = process_prefix(q,k,v,reader,valid=valid,steps=4,mode=mode)
    state, pieces = None, []
    for a,b in [(0,7),(7,12),(12,17)]:
        result = process_prefix(q[:,a:b],k[:,a:b],v[:,a:b],reader,state=state,
                                valid=valid[:,a:b],steps=4,mode=mode)
        state = result.state
        pieces.append(result.outputs)
    close(torch.cat(pieces,1),whole.outputs)
    close(state.M,whole.state.M)
    order = torch.tensor([1,0,0,2])
    reordered = process_prefix(q.index_select(0,order),k.index_select(0,order),
        v.index_select(0,order),reader,valid=valid.index_select(0,order),steps=4,mode=mode)
    close(reordered.outputs,whole.outputs.index_select(0,order))
    assert reordered.state.M.data_ptr() != whole.state.M.data_ptr()
    with pytest.raises(ValueError,match='reset'):
        process_prefix(q,k,v,reader,valid=valid,reset=torch.zeros_like(valid))
    with patch.object(torch.Tensor,'item',side_effect=AssertionError('host scalar')):
        process_prefix(q,k,v,reader,valid=valid,steps=4,mode=mode)


def test_t0_definition_and_history_independence():
    torch.manual_seed(192)
    q,k,v = [torch.randn(2,7,8) for _ in range(3)]
    valid = torch.ones(2,7,dtype=torch.bool)
    valid[1, -2:] = False
    reader = LatentReader(8)
    z = q
    for _ in range(4):
        a = torch.nn.functional.silu(rmsnorm_no_affine(z)@reader.Rq+rmsnorm_no_affine(k))*v
        z = z + reader.eta*(a@reader.Ro)
    actual = process_kernel_tokenwise(q,k,v,reader,valid,4)
    close(actual,(z-q)*valid[...,None])
    changed = [x.clone() for x in (q,k,v)]
    for x in changed:
        x[:,:5] += 123
    close(process_kernel_tokenwise(*changed,reader,valid,4)[:,5:], actual[:,5:])


def history_inputs():
    torch.manual_seed(193)
    q,k,v = [torch.randn(2,17,8) for _ in range(3)]
    valid = torch.ones(2,17,dtype=torch.bool)
    valid[0,:2], valid[1,-3:] = False, False
    return q,k,v,valid,LatentReader(8)


def test_history_identities_strict_and_current_weights():
    q,k,v,valid,reader = history_inputs()
    cfg = AdapterConfig(branch='memory',rank=8,impl='prefix',fixed_query=True,steps=4,fusion='F_out')
    normal = run_history(q,k,v,reader,valid,cfg,{'capture_history':True,'trace_reads':True})
    for options in ({'kind':'iv_window','window':64},
                    {'kind':'iv_swap','donor':normal.diagnostics['history']},
                    {'kind':'iv_permute','phase':'R1','example_ids':['a','b'],'identity_permutation':True}):
        close(run_history(q,k,v,reader,valid,cfg,options).outputs,normal.outputs)
    off = run_history(q,k,v,reader,valid,cfg,{'kind':'iv_off'})
    assert torch.count_nonzero(off.outputs) == 0
    strict = run_history(q,k,v,reader,valid,cfg,{'kind':'iv_strict'})
    assert torch.count_nonzero(strict.outputs[0,:3]) == 0
    assert torch.count_nonzero(strict.outputs[1,0]) == 0
    weights = normal.diagnostics['current_weight']
    assert weights.shape == (2,17,4)
    close(weights[...,1:], weights[...,:1].expand_as(weights[...,1:]))
    assert ((weights >= 0)&(weights <= 1)).all()
    assert torch.count_nonzero(weights[~valid]) == 0
    one = run_history(q,k,v,reader,valid,cfg,{'kind':'iv_window1'})
    # One inclusive write reduces the normalized kernel read to v (epsilon retained).
    tiny = process_prefix(q[:,5:6],k[:,5:6],v[:,5:6],reader,valid=valid[:,5:6],steps=4,fixed_query=True)
    close(one.outputs[:,5:6],tiny.outputs)


@pytest.mark.parametrize('kind',['iv_window1','iv_window8','iv_strict','iv_swap','iv_permute','iv_off'])
def test_every_intervention_causal_at_feature_boundary(kind):
    q,k,v,valid,reader = history_inputs()
    cfg = AdapterConfig(branch='memory',rank=8,impl='prefix',fusion='F_out',steps=4)
    donor = run_history(q.flip(0),k.flip(0),v.flip(0),reader,valid.flip(0),cfg,{'capture_history':True})
    options = {'kind':kind,'phase':'R1','example_ids':['a','b'],'donor':donor.diagnostics['history']}
    actual = run_history(q,k,v,reader,valid,cfg,options).outputs
    changed = [x.clone() for x in (q,k,v)]
    for x in changed:
        x[:,10:] += 99
    rerun = run_history(*changed,reader,valid,cfg,options).outputs
    close(rerun[:,:10],actual[:,:10])


def test_wrong_implementations_are_rejected():
    q,k,v,valid,reader = history_inputs()
    normal = process_prefix(q,k,v,reader,valid=valid)
    # Full-sequence K^T V leaks a changed future into earlier positions.
    from bypass_core import kernel_prefix_states,read_kernel_prefix
    keys,values,matrices,masses,counts,_ = kernel_prefix_states(k,v,reader,valid)
    bad,_ = read_kernel_prefix(q,reader,matrices[:,-1:].expand_as(matrices),
        masses[:,-1:].expand_as(masses),counts[:,-1:].expand_as(counts),valid)
    with pytest.raises(AssertionError):
        close(bad[:,:8],normal.outputs[:,:8])
    duplicate = replace(normal,accounting={**normal.accounting,'writes':valid.sum()*2})
    with pytest.raises(AssertionError):
        assert duplicate.accounting['writes'] == valid.sum()
    padded = process_prefix(q,k,v,reader,valid=torch.ones_like(valid))
    with pytest.raises(AssertionError):
        assert torch.equal(padded.state.counts,valid.sum(1))


def test_donor_mapping_is_within_task_derangement():
    rows = [{'task':t,'id':i} for t in ('piqa','boolq') for i in range(7)]
    indices = donor_indices(rows,'R1')
    assert indices == donor_indices(rows,'R1')
    assert sorted(indices) == list(range(14))
    assert all(i != d and rows[i]['task'] == rows[d]['task'] for i,d in enumerate(indices))
