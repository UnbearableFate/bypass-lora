"""Real, revision-pinned Qwen3 tests; run only inside the registered GPU job."""
from contextlib import contextmanager
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bypass_model import AdapterConfig, BranchModel, FUSIONS, Fusion
from calibrate import load_base, fixture

TOL = json.loads(Path(__file__).with_name('tolerances.json').read_text())


@pytest.fixture(scope='module')
def base():
    model = load_base()
    assert torch.cuda.get_device_name() == TOL['hardware']
    assert model.config.hidden_size == 1024 and len(model.model.layers) == 28
    assert model.config.num_attention_heads == 16 and model.config.num_key_value_heads == 8
    yield model


@contextmanager
def adapted(base, config):
    torch.manual_seed(42)
    wrapped = BranchModel(base, config)
    try:
        yield wrapped
    finally:
        for i, layer in wrapped.adapters().items():
            for handle in layer.handles:
                handle.remove()
            base.model.layers[i] = layer.original
        base.zero_grad(set_to_none=True)


def measure(actual, expected, record_property, name, atol=None):
    diff = (actual.float() - expected.float()).abs()
    record_property(name + '.max_abs_err', diff.max().item())
    record_property(name + '.max_rel_err', (diff / expected.float().abs().clamp_min(1e-6)).max().item())
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0,
                               atol=TOL['bf16_atol'] if atol is None else atol)


@pytest.mark.parametrize('fusion', FUSIONS)
@pytest.mark.parametrize('branch,layers', [('tokenwise_silu', (6,13,20,27)),
                                          ('tokenwise_linear', (6,13,20,27)),
                                          ('memory', (14,))])
def test_function_preservation(base, fusion, branch, layers, record_property):
    ids = fixture()
    mask = torch.ones_like(ids)
    with torch.no_grad():
        reference = base(ids, attention_mask=mask, use_cache=False).logits
        with adapted(base, AdapterConfig(layers=layers, branch=branch, fusion=fusion)) as model:
            actual = model(ids, attention_mask=mask)
            # Same-shape repeat discrepancy was zero: stricter than general cache tolerance.
            measure(actual.logits, reference, record_property, 'zero_exit', atol=2e-5)
            for counts in actual.accounting.values():
                assert counts['base'] == {'attention': 1, 'ffn': 1}


@pytest.mark.parametrize('branch', ['tokenwise_silu', 'tokenwise_linear', 'memory'])
def test_two_step_grad_start_frozen_input_jacobian(base, branch, record_property):
    ids = fixture()
    # A lower adapter forces the gradient to traverse 21 frozen upper blocks.
    with adapted(base, AdapterConfig(layers=(6,), branch=branch, steps=2)) as model:
        side = model.adapters()[6].side
        originals = {n: p for n,p in base.named_parameters() if '.side.' not in n}
        assert not any(p.requires_grad for p in originals.values())
        # GPU clone is ~1.2 GB and checks every frozen parameter, not a sample.
        before = {n: p.detach().clone() for n,p in originals.items()}
        optimizer = torch.optim.SGD(model.adapter_parameters(), lr=.1)
        for step in (1,2):
            model.zero_grad(set_to_none=True)
            logits = model(ids).logits.float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
            loss.backward()
            for name, parameter in side.named_parameters():
                norm = 0. if parameter.grad is None else parameter.grad.norm().item()
                record_property(f'grad.step{step}.{name}', norm)
                if name.startswith('fusion.') or step == 2:
                    assert norm > 0 and torch.isfinite(parameter.grad).all(), (step, name, norm)
                else:
                    assert norm == 0, (step, name, norm)
            assert all(p.grad is None for p in originals.values())
            optimizer.step()
        max_change = max((p - before[n]).abs().max().item() for n,p in originals.items())
        record_property('backbone.max_parameter_change', max_change)
        assert max_change == 0
        del before


@pytest.mark.parametrize('rank', [32,128])
def test_fusion_identity(rank, record_property):
    torch.manual_seed(91)
    d, m = 48, 64
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(d,m,bias=False)
            self.up_proj = nn.Linear(d,m,bias=False)
            self.down_proj = nn.Linear(m,d,bias=False)
        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
    mlp = MLP().requires_grad_(False)
    p = (torch.randn(rank,d)*.01).requires_grad_()
    u,h,z = torch.randn(2,4,d),torch.randn(2,4,d),torch.randn(2,4,rank)
    pre = h + mlp(u + z @ p)
    # The SAME P participates in both graphs; C matrices are not independent parameters.
    cg, cu = p @ mlp.gate_proj.weight.T, p @ mlp.up_proj.weight.T
    dual = h + mlp.down_proj(F.silu(mlp.gate_proj(u)+z@cg)*(mlp.up_proj(u)+z@cu))
    target = torch.randn_like(pre)
    gp = torch.autograd.grad((pre*target).sum(), p)[0]
    gd = torch.autograd.grad((dual*target).sum(), p)[0]
    measure(dual,pre,record_property,'fusion.output',atol=TOL['fp32_atol'])
    torch.testing.assert_close(gd,gp,atol=TOL['fp32_atol'],rtol=TOL['fp32_rtol'])
    record_property('fusion.gradient.max_abs_err',(gd-gp).abs().max().item())
    # Also test the actual Fusion path with the constrained matrices.
    fusion = Fusion('C_dual',rank,d,m)
    del fusion.C_g, fusion.C_u
    p2 = p.detach().clone().requires_grad_()
    fusion.C_g, fusion.C_u = p2 @ mlp.gate_proj.weight.T, p2 @ mlp.up_proj.weight.T
    actual = fusion(h,u,z,mlp,nn.Identity())
    measure(actual,pre,record_property,'fusion.implementation',atol=TOL['fp32_atol'])
    actual_grad = torch.autograd.grad((actual*target).sum(),p2)[0]
    pre_fusion = Fusion('F_pre',rank,d,m)
    del pre_fusion.P
    pre_fusion.P = p2
    pre_actual = pre_fusion(h,u,z,mlp,nn.Identity())
    pre_grad = torch.autograd.grad((pre_actual*target).sum(),p2)[0]
    torch.testing.assert_close(actual_grad,pre_grad,atol=TOL['fp32_atol'],rtol=TOL['fp32_rtol'])
    record_property('fusion.implementation_gradient.max_abs_err',(actual_grad-pre_grad).abs().max().item())


def nonzero_exits(model):
    with torch.no_grad():
        for layer in model.adapters().values():
            for p in layer.side.fusion.parameters():
                p.normal_(std=.1)


def chunked(model,ids,mask,chunk,phase='F',memory=None):
    cache = None
    parts = []
    for start in range(0,ids.shape[1],chunk):
        end = min(start+chunk,ids.shape[1])
        result = model(ids[:,start:end],attention_mask=mask[:,:end],cache=cache,
                       memory=memory if cache is None else None,phase=phase,use_cache=True)
        parts.append(result.logits)
        cache = result.cache
    return torch.cat(parts,1),cache


@pytest.mark.parametrize('mode', ['inclusive','strict'])
@pytest.mark.parametrize('rank,steps', [(32,1),(32,2),(128,4)])
def test_cache_causality_padding(base,mode,rank,steps,record_property):
    ids = fixture()
    mask = torch.ones_like(ids)
    mask[0,:2] = 0
    mask[1,-2:] = 0
    with torch.no_grad():
        pos = (mask.cumsum(-1)-1).clamp_min(0)
        base_logits = base(ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
    with adapted(base,AdapterConfig(layers=(14,),branch='memory',rank=rank,steps=steps,mode=mode)) as model, torch.no_grad():
        nonzero_exits(model)
        full = model(ids,attention_mask=mask)
        effect = (full.logits[mask.bool()].float()-base_logits[mask.bool()].float()).abs().max().item()
        record_property('side_effect.max_abs',effect)
        assert effect > 2*TOL['bf16_atol']
        for chunk in (3,1):
            actual,cache = chunked(model,ids,mask,chunk)
            measure(actual[mask.bool()],full.logits[mask.bool()],record_property,f'cache.chunk{chunk}')
            torch.testing.assert_close(cache.memory[14].counts,mask.sum(-1))
        changed = ids.clone()
        changed[:,5:] = 330
        suffix = model(changed,attention_mask=mask)
        prefix_valid = mask[:, :5].bool()
        measure(suffix.logits[:,:5][prefix_valid],full.logits[:,:5][prefix_valid],
                record_property,'causal',atol=2e-5)
        assert full.accounting[14]['side']['writes'] == int(mask.sum())
        assert full.accounting[14]['side']['reads'] == steps*int(mask.sum())
        assert full.accounting[14]['base'] == {'attention':1,'ffn':1}


def test_base_padding_causal_diagnostic(base, record_property):
    # Fully masked padding query rows are not semantic token outputs. The pinned
    # eager backend uses finite-min masking followed by softmax for these rows.
    ids = fixture()
    mask = torch.ones_like(ids)
    mask[0,:2] = 0
    mask[1,-2:] = 0
    changed = ids.clone()
    changed[:,5:] = 330
    pos = (mask.cumsum(-1)-1).clamp_min(0)
    with torch.no_grad():
        a = base(ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
        b = base(changed,attention_mask=mask,position_ids=pos,use_cache=False).logits
    prefix_valid = mask[:,:5].bool()
    measure(b[:,:5][prefix_valid],a[:,:5][prefix_valid],record_property,
            'base.valid_prefix',atol=2e-5)
    padded_diff = (b[:,:5][~prefix_valid].float()-a[:,:5][~prefix_valid].float()).abs().max().item()
    record_property('base.padded_prefix.max_abs_err',padded_diff)
    assert padded_diff > 1, 'Revisit the padding diagnosis if the pinned Base behavior changes'


@pytest.mark.parametrize('backend', ['kernel_matrix','kernel_slot_reference','soft_slot'])
def test_m_line_access_ownership(base,backend,record_property):
    ids = fixture()
    with adapted(base,AdapterConfig(layers=(14,),branch='memory',backend=backend,steps=2)) as model, torch.no_grad():
        nonzero_exits(model)
        support = model(ids,phase='support',write_mask=torch.ones_like(ids,dtype=torch.bool))
        assert support.cache.backbone is None
        assert support.accounting[14]['side']['reads'] == 0
        assert support.accounting[14]['side']['writes'] == ids.numel()
        snapshot = support.cache.memory
        captured = []
        handle = base.register_forward_pre_hook(lambda module,args,kw: captured.append(kw.get('past_key_values')),with_kwargs=True)
        try:
            query = model(ids[:,:4],memory=snapshot,phase='query')
        finally:
            handle.remove()
        assert captured == [None]
        again = model(ids[:,:4],memory=snapshot,phase='query')
        measure(again.logits,query.logits,record_property,'shared_support',atol=0)
        assert query.accounting[14]['side']['writes'] == 0
        assert query.accounting[14]['side']['reads'] == 2*ids[:,:4].numel()
        assert torch.equal(snapshot[14].M,query.cache.memory[14].M)
        assert snapshot[14].slot_count == query.cache.memory[14].slot_count
        swapped = support.cache.reorder(torch.tensor([1,0],device=ids.device))
        swapped_query = model(ids[:,:4],memory=swapped.memory,phase='query')
        effect = (swapped_query.logits.float()-query.logits.float()).abs().max().item()
        record_property('support_owner_effect.max_abs',effect)
        assert effect > TOL['bf16_atol']
        for size in (1,3):
            logits,_ = chunked(model,ids[:,:4],torch.ones_like(ids[:,:4]),size,'query',snapshot)
            measure(logits,query.logits,record_property,f'query.cache{size}')
        idx = torch.tensor([1,0,1],device=ids.device)
        reordered = support.cache.reorder(idx)
        batch = model(ids[idx,:4],memory=reordered.memory,phase='query')
        measure(batch.logits,query.logits[idx],record_property,'owner_reorder')
        with pytest.raises(ValueError,match='cross protocol'):
            model(ids,cache=support.cache,phase='query')


@pytest.mark.parametrize('tap', ['qknorm_pre_rope','raw_qkv','direct_hidden'])
def test_feature_tap_once(base,tap):
    with adapted(base,AdapterConfig(layers=(14,),branch='memory',tap=tap)) as model:
        layer = model.adapters()[14]
        counts = {'q':0,'k':0,'v':0}
        handles=[]
        for key in counts:
            def hook(module,args,output,key=key):
                counts[key]+=1
            handles.append(getattr(layer.original.self_attn,key+'_proj').register_forward_hook(hook))
        try:
            model(fixture())
        finally:
            for h in handles: h.remove()
        assert counts == {'q':1,'k':1,'v':1}
        dims = (1024,1024,1024) if tap=='direct_hidden' else (2048,1024,1024)
        assert tuple(p.shape[0] for p in (layer.side.projection.A_q,layer.side.projection.A_k,layer.side.projection.A_v)) == dims


def test_export_reload(base,tmp_path,record_property):
    config = AdapterConfig(layers=(14,),branch='memory',backend='soft_slot',rank=128,
                           steps=4,mode='strict',update='anchored',readout='full',fusion='F_pre')
    with adapted(base,config) as model,torch.no_grad():
        nonzero_exits(model)
        expected = model(fixture()).logits
        model.export_adapter(tmp_path)
    reloaded = BranchModel.load_adapter(base,tmp_path)
    try:
        assert reloaded.config == config
        with torch.no_grad():
            actual = reloaded(fixture()).logits
        measure(actual,expected,record_property,'reload',atol=0)
    finally:
        for i,layer in reloaded.adapters().items():
            for h in layer.handles: h.remove()
            base.model.layers[i] = layer.original


@pytest.mark.parametrize('backend',['kernel_matrix','kernel_slot_reference','soft_slot'])
def test_support_delimiter_write_scope_and_capacity(base,backend):
    ids=fixture()
    mask=torch.ones_like(ids,dtype=torch.bool)
    mask[:,:2]=False
    delimiters=torch.zeros_like(mask)
    delimiters[:,[3,7]]=True
    with adapted(base,AdapterConfig(layers=(14,),branch='memory',backend=backend)) as model,torch.no_grad():
        support=model(ids,attention_mask=mask,phase='support',write_mask=delimiters)
        state=support.cache.memory[14]
        assert support.accounting[14]['side']['writes']==4
        assert state.counts.tolist()==[2,2]
        assert state.slot_count==(None if backend=='kernel_matrix' else 2)
        for phase in ('query','F'):
            with pytest.raises(ValueError,match='only accepted'):
                model(ids,phase=phase,memory=support.cache.memory if phase=='query' else None,write_mask=delimiters)
        with pytest.raises(ValueError,match='explicit write_mask'):
            model(ids,phase='support')
        invalid=delimiters.clone(); invalid[:,0]=True
        with pytest.raises(ValueError,match='subset'):
            model(ids,attention_mask=mask,phase='support',write_mask=invalid)
    if backend!='kernel_matrix':
        with adapted(base,AdapterConfig(layers=(14,),branch='memory',backend=backend,slot_capacity=1)) as model:
            with pytest.raises(ValueError,match='capacity exceeded'):
                model(ids,phase='support',write_mask=delimiters)


def test_reject_missing_memory_and_reused_cache(base):
    ids=fixture()
    with adapted(base,AdapterConfig(layers=(14,),branch='memory')) as model,torch.no_grad():
        with pytest.raises(ValueError,match='missing support memory'):
            model(ids,phase='query',memory={})
        prefix=model(ids[:,:3],use_cache=True).cache
        copied=prefix.reorder(torch.arange(ids.shape[0],device=ids.device))
        first=model(ids[:,3:4],cache=prefix,attention_mask=torch.ones_like(ids[:,:4]),use_cache=True)
        same=model(ids[:,3:4],cache=copied,attention_mask=torch.ones_like(ids[:,:4]),use_cache=True)
        torch.testing.assert_close(first.logits,same.logits,atol=0,rtol=0)
        with pytest.raises(ValueError,match='stale or reused cache'):
            model(ids[:,3:4],cache=prefix,attention_mask=torch.ones_like(ids[:,:4]),use_cache=True)
