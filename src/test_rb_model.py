"""Real four-layer RB0 checks; GPU BF16 bounds remain E0001/E0004 tolerances."""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from bypass_core import process_prefix
from bypass_model import AdapterConfig,BranchModel,BranchCache
from calibrate import fixture,load_base
from rb_train import adapter_config,make_model,PARAMETERS
from test_model import base,adapted,measure,nonzero_exits,chunked,TOL
from test_rb_core import close


@contextmanager
def attached(base,config,model_class=BranchModel):
    torch.manual_seed(184)
    wrapped=model_class(base,config)
    try:
        yield wrapped
    finally:
        for i,layer in wrapped.adapters().items():
            for handle in layer.handles:
                handle.remove()
            base.model.layers[i]=layer.original
        base.zero_grad(set_to_none=True)


@pytest.mark.parametrize('branch,impl',[('memory','prefix'),('memory','reference'),('kernel_tokenwise','prefix')])
@pytest.mark.parametrize('fusion',['F_out','C_dual','C_up'])
def test_four_layer_zero_exit_and_one_original_call(base,branch,impl,fusion,record_property):
    ids=fixture()
    with torch.no_grad():
        reference=base(ids,use_cache=False).logits
    config=AdapterConfig(branch=branch,impl=impl,fusion=fusion)
    with attached(base,config) as model,torch.no_grad():
        calls={i:{'attention':0,'ffn':0} for i in range(28)}
        handles=[]
        def counter(i,name):
            def hook(*args):
                calls[i][name]+=1
            return hook
        for i,layer in enumerate(base.model.layers):
            layer=getattr(layer,'original',layer)
            handles.extend([layer.self_attn.register_forward_hook(counter(i,'attention')),
                            layer.mlp.register_forward_hook(counter(i,'ffn'))])
        try:
            actual=model(ids)
        finally:
            for h in handles:
                h.remove()
        measure(actual.logits,reference,record_property,'four_layer_zero',atol=2e-5)
        assert all(v=={'attention':1,'ffn':1} for v in calls.values())


def test_residual_topology_uses_unchanged_first_FFN_input(base,record_property):
    ids=fixture()
    reference_u=[]
    handle=base.model.layers[6].mlp.register_forward_pre_hook(lambda m,a:reference_u.append(a[0].clone()))
    with torch.no_grad():
        base(ids,use_cache=False)
    handle.remove()
    with attached(base,adapter_config('RB1-02')) as model,torch.no_grad():
        nonzero_exits(model)
        records={i:{} for i in model.config.layers}
        handles=[]
        for i,layer in model.adapters().items():
            r=records[i]
            def save_input(m,a,r=r): r['X']=a[0].clone()
            def save_attn(m,a,o,r=r): r['o']=o[0].clone()
            def save_mlp(m,a,o,r=r): r['U']=a[0].clone();r['F']=o.clone()
            def save_side(m,a,o,r=r): r['Z']=o[0].clone()
            def save_output(m,a,o,r=r): r['Y']=o.clone()
            handles.extend([layer.register_forward_pre_hook(save_input),
                layer.original.self_attn.register_forward_hook(save_attn),
                layer.original.mlp.register_forward_hook(save_mlp),
                layer.side.register_forward_hook(save_side),layer.register_forward_hook(save_output)])
        try:
            output=model(ids)
        finally:
            for h in handles: h.remove()
        assert torch.equal(records[6]['U'],reference_u[0])
        for i,r in records.items():
            h=r['X']+r['o']
            original=h+r['F']
            delta=(r['Z']@model.adapters()[i].side.fusion.P).to(h.dtype)
            assert torch.equal(r['Y'],original+delta)
            assert torch.equal(r['U'],model.adapters()[i].original.post_attention_layernorm(h))
            # D0026:subtracting the rounded BF16 sum cannot recover an addend
            # smaller than half an ulp. The gate above is exact,with zero tolerance.
            error=(r['Y'].float()-original.float()-delta.float()).abs()
            _,exponent=torch.frexp(r['Y'].float().abs())
            ulp=torch.ldexp(torch.ones_like(error),exponent-8).clamp_min(2.**-133)
            ulp=torch.where(r['Y']==0,torch.full_like(ulp,2.**-133),ulp)
            record_property(f'layer{i}.residual_subtraction.max_abs',error.max().item())
            record_property(f'layer{i}.residual_subtraction.max_ulp_bf16_Y',(error/ulp).max().item())
            assert output.accounting[i]['base']=={'attention':1,'ffn':1}


@pytest.mark.parametrize('arm',['RB1-02','RB1-04','RB1-08'])
def test_four_layer_two_step_gradient_and_frozen_Jacobian(base,arm,record_property):
    ids=fixture()
    with attached(base,adapter_config(arm)) as model:
        originals={n:p for n,p in base.named_parameters() if '.side.' not in n}
        before={n:p.detach().clone() for n,p in originals.items()}
        optimizer=torch.optim.SGD(model.adapter_parameters(),lr=.1)
        for step in (1,2):
            model.zero_grad(set_to_none=True)
            logits=model(ids).logits.float()
            F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1)).backward()
            for i,layer in model.adapters().items():
                for name,p in layer.side.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    norm=p.grad.norm().item()
                    record_property(f'grad.layer{i}.step{step}.{name}',norm)
                    assert norm>0 if step==2 or name.startswith('fusion.') else norm==0,(i,step,name,norm)
            assert all(p.grad is None for p in originals.values())
            optimizer.step()
        assert max((p-before[n]).abs().max().item() for n,p in originals.items())==0


def test_parameter_counts_all_arms_widths_and_initialization_pairing(base,record_property):
    initial=None
    for number in range(1,10):
        arm=f'RB1-{number:02}'
        if arm=='RB1-07':
            continue
        with patch('rb_train.load_base',return_value=base):
            model=make_model(arm)
        try:
            params=list(model.adapter_parameters())
            assert sum(p.numel() for p in params)==PARAMETERS[arm]
            assert all(p.requires_grad and p.dtype==torch.float32 for p in params)
            record_property(arm+'.parameters',sum(p.numel() for p in params))
            if number<=6:
                values={str(i)+'.'+n:p.detach().cpu().clone() for i,l in model.adapters().items() for n,p in l.side.named_parameters()}
                if initial is None: initial=values
                assert set(values)==set(initial)
                assert all(torch.equal(values[n],initial[n]) for n in initial)
            nonzero_exits(model)
            model(fixture()).logits.float().square().mean().backward()
            assert sum(p.numel() for p in params if p.grad is not None)==PARAMETERS[arm]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in params)
        finally:
            for i,layer in model.adapters().items():
                for h in layer.handles:h.remove()
                base.model.layers[i]=layer.original
    for rank,expected in ((44,916608),(64,1343488)):
        with attached(base,replace(adapter_config('RB1-02'),rank=rank)) as model:
            assert sum(p.numel() for p in model.adapter_parameters())==expected
            nonzero_exits(model)
            loss=model(fixture()).logits.float().square().mean()
            loss.backward()
            assert sum(p.numel() for p in model.adapter_parameters() if p.grad is not None)==expected
    from test_v1_lora import adapted as lora_attached
    from v1_lora import LoRAConfig
    for layers,rank,expected in (((6,13,20,27),7,630784),(tuple(range(28)),1,630784),(tuple(range(28)),2,1261568)):
        with lora_attached(base,LoRAConfig(layers=layers,rank=rank)) as model:
            params=list(model.adapter_parameters())
            assert sum(p.numel() for p in params)==expected
            with torch.no_grad():
                for target in model._targets.values():target.B.normal_(std=.01)
            model(fixture()).logits.float().square().mean().backward()
            assert sum(p.numel() for p in params if p.grad is not None)==expected


def test_fixed_query_wiring_has_linear_S4_increment(base):
    with attached(base,adapter_config('RB1-05')) as model,torch.no_grad():
        nonzero_exits(model)
        handles=[]
        def check(side,inputs,output):
            u,features,state,valid,phase,*_=inputs
            q,k,v=side.projection(features['q'],features['k'],features['v'])
            one=process_prefix(q,k,v,side.reader,valid=valid,steps=1,fixed_query=True)
            close(output[0],4*one.outputs)
        for layer in model.adapters().values():handles.append(layer.side.register_forward_hook(check))
        try:model(fixture())
        finally:
            for h in handles:h.remove()


@pytest.mark.parametrize('impl',['prefix','reference'])
def test_four_layer_cache_causality_padding_and_ownership(base,impl,record_property):
    ids=fixture();mask=torch.ones_like(ids)
    mask[0,:2]=0;mask[1,-2:]=0
    with attached(base,adapter_config('RB1-04',impl)) as model,torch.no_grad():
        nonzero_exits(model)
        full=model(ids,attention_mask=mask)
        for chunk in (3,1):
            values,cache=chunked(model,ids,mask,chunk)
            if chunk==3:
                measure(values[mask.bool()],full.logits[mask.bool()],record_property,f'{impl}.chunk{chunk}')
            else:
                # D0026:retain the locked BF16 comparison as a reported
                # diagnostic. FP32 continuation below is the decisive carry gate.
                delta=(values[mask.bool()].float()-full.logits[mask.bool()].float()).abs().max().item()
                record_property(f'{impl}.chunk1.max_abs_err',delta)
                record_property(f'{impl}.chunk1.within_locked_bf16',int(delta<=TOL['bf16_atol']))
            for state in cache.memory.values():assert torch.equal(state.counts,mask.sum(-1))
        changed=ids.clone();changed[:,5:]=330
        other=model(changed,attention_mask=mask)
        measure(other.logits[:,:5][mask[:,:5].bool()],full.logits[:,:5][mask[:,:5].bool()],record_property,'causal',atol=2e-5)
        order=torch.tensor([1,0,0],device=ids.device)
        moved=model(ids.index_select(0,order),attention_mask=mask.index_select(0,order))
        expected=full.logits.index_select(0,order)
        measure(moved.logits[mask.index_select(0,order).bool()],expected[mask.index_select(0,order).bool()],record_property,'row_reorder')
        prefix=model(ids[:,:3],attention_mask=mask[:,:3],use_cache=True)
        source={i:s.M.clone() for i,s in prefix.cache.memory.items()}
        owner=prefix.cache.reorder(order)
        model(ids[:,3:4].index_select(0,order),attention_mask=mask[:,:4].index_select(0,order),cache=owner,use_cache=True)
        assert all(torch.equal(s.M,source[i]) for i,s in prefix.cache.memory.items())
        for counts in full.accounting.values():
            assert counts['side']['writes']==mask.sum() and counts['side']['reads']==mask.sum()*4


def test_BF16_token_decode_prefix_reference_agreement(base,record_property):
    ids=fixture();mask=torch.ones_like(ids)
    mask[0,:2]=0;mask[1,-2:]=0
    outputs=[]
    for impl in ('reference','prefix'):
        with attached(base,adapter_config('RB1-04',impl)) as model,torch.no_grad():
            nonzero_exits(model)
            values,_=chunked(model,ids,mask,1)
            outputs.append(values)
    measure(outputs[1][mask.bool()],outputs[0][mask.bool()],record_property,'BF16.token.prefix_reference')


@pytest.mark.parametrize('impl',['prefix','reference'])
def test_FP32_four_layer_cache_and_state_clear_negative_control(impl,record_property):
    # D0026:fresh FP32 backbone;do not recast the BF16 fixture or rotary buffers.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    fresh=load_base().float()
    ids=fixture();mask=torch.ones_like(ids)
    mask[0,:2]=0;mask[1,-2:]=0
    with attached(fresh,adapter_config('RB1-04',impl)) as model,torch.no_grad():
        nonzero_exits(model)
        whole=model(ids,attention_mask=mask)
        outputs={8:whole.logits}
        for chunk in (3,1):
            values,cache=chunked(model,ids,mask,chunk)
            outputs[chunk]=values
            for i,state in cache.memory.items():
                close(state.M,whole.cache.memory[i].M,record_property,f'{impl}.FP32.chunk{chunk}.layer{i}.M')
                close(state.c,whole.cache.memory[i].c,record_property,f'{impl}.FP32.chunk{chunk}.layer{i}.c')
                assert torch.equal(state.counts,whole.cache.memory[i].counts)
        for a,b in ((3,8),(1,8),(1,3)):
            close(outputs[a][mask.bool()],outputs[b][mask.bool()],record_property,f'{impl}.FP32.logits.{a}_{b}')
        # Keep backbone KV,deliberately erase only side M,c,n at each boundary.
        pieces=[];cache=None
        for start in range(ids.shape[1]):
            if cache is not None:
                cache=BranchCache({i:s.reset() for i,s in cache.memory.items()},cache.backbone,cache.phase)
            result=model(ids[:,start:start+1],attention_mask=mask[:,:start+1],cache=cache,use_cache=True)
            cache=result.cache;pieces.append(result.logits)
        bad=torch.cat(pieces,1)
        with pytest.raises(AssertionError):
            close(bad[mask.bool()],whole.logits[mask.bool()])
        record_property(impl+'.FP32.state_clear_rejected',1)


@pytest.mark.parametrize('arm',['RB1-01','RB1-02','RB1-05','RB1-06','RB1-08','RB1-09','RB1-07'])
def test_every_checkpoint_class_fresh_base_reload(arm,tmp_path,record_property):
    model=make_model(arm)
    model.eval()
    with torch.no_grad():
        if isinstance(model,BranchModel):nonzero_exits(model)
        else:
            for target in model._targets.values():target.B.normal_(std=.01)
        expected=model(fixture(),use_cache=False).logits
    model.export_adapter(tmp_path/arm)
    del model
    torch.cuda.empty_cache()
    model=make_model(arm,checkpoint=tmp_path/arm).eval()
    with torch.no_grad():actual=model(fixture(),use_cache=False).logits
    measure(actual,expected,record_property,arm+'.reload',atol=0)
    assert torch.equal(actual.argmax(-1),expected.argmax(-1))


@pytest.mark.parametrize('arm',['RB1-01','RB1-04','RB1-08','RB1-09'])
def test_history_interventions_model_identities_and_causality(base,arm,record_property):
    ids=fixture();mask=torch.ones_like(ids)
    with torch.no_grad():reference=base(ids,use_cache=False).logits
    with attached(base,adapter_config(arm)) as model,torch.no_grad():
        model.eval();nonzero_exits(model)
        normal=model(ids,capture_history=True,trace_reads=True)
        donor={i:record['history'] for i,record in normal.diagnostics.items()} if arm!='RB1-01' else {}
        identities=[{'kind':'iv_window','window':64},{'kind':'iv_swap','donor':donor},
            {'kind':'iv_permute','identity_permutation':True,'phase':'R1','example_ids':['a','b']}]
        for options in identities:
            value=model(ids,intervention=options)
            measure(value.logits,normal.logits,record_property,arm+'.identity')
        off=model(ids,intervention={'kind':'iv_off'})
        measure(off.logits,reference,record_property,'iv_off',atol=2e-5)
        for kind in ('iv_window1','iv_window8','iv_strict','iv_swap','iv_permute'):
            options={'kind':kind,'donor':donor,'phase':'R1','example_ids':['a','b']}
            value=model(ids,intervention=options)
            changed=ids.clone();changed[:,5:]=330
            other=model(changed,intervention=options)
            measure(other.logits[:,:5],value.logits[:,:5],record_property,kind+'.causal',atol=2e-5)
            if arm=='RB1-01':assert torch.equal(value.logits,normal.logits)


def test_default_paths_bitwise_match_task_master(base):
    root=Path('/work/xg24i002/x10041/bypass-lora/snapshots/de9da72acc1c9f9e0903ff1e144a87cc4ba4c615/src')
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        value=importlib.util.module_from_spec(spec)
        sys.modules[name]=value
        spec.loader.exec_module(value)
        return value
    old_core=module('_rb_baseline_core',root/'bypass_core.py')
    with patch.dict(sys.modules,{'bypass_core':old_core}):
        old_model=module('_rb_baseline_model',root/'bypass_model.py')
    ids=fixture();mask=torch.ones_like(ids)
    for kwargs in ({'branch':'tokenwise_silu','fusion':'C_up'},
                   {'branch':'memory','fusion':'F_out','steps':4},
                   {'branch':'memory','fusion':'C_dual','layers':(14,),'steps':2}):
        outputs=[];parameters=[]
        for cls,cfg_cls in ((old_model.BranchModel,old_model.AdapterConfig),(BranchModel,AdapterConfig)):
            with attached(base,cfg_cls(**kwargs),cls) as model,torch.no_grad():
                parameters.append([p.detach().clone() for p in model.adapter_parameters()])
                for layer in model.adapters().values():
                    for p in layer.side.fusion.parameters():p.normal_(std=.01)
                outputs.append(model(ids,attention_mask=mask).logits)
        assert all(torch.equal(a,b) for a,b in zip(*parameters))
        assert torch.equal(*outputs)
