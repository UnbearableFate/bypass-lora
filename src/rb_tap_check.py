"""E-0028 initialization and normalized-input tap checks; run in PBS."""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path

import torch

from calibrate import REVISION
from v1_run import batch_tensors
from v5_train import write_json


def initial_state(model):
    state={name:value.detach().cpu().clone() for name,value in model.named_parameters()
           if value.requires_grad}
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode()+b'\0'+str(tuple(value.shape)).encode()+b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest(),state


@torch.no_grad()
def step0_diagnostics(model,example,tokenizer):
    """Describe vbar/Z on the first fixed R1 training candidate, before updates."""
    records={}
    handles=[]
    batch=batch_tensors([example],tokenizer.pad_token_id)
    valid=batch['attention_mask'].bool()
    def rms(value):
        selected=value[valid].double()
        return float(selected.square().mean().sqrt().item())
    for index,layer in model.adapters().items():
        records[index]={}
        def projection(module,args,result,index=index):
            records[index]['vbar_rms']=rms(result[2])
        def output(module,args,result,index=index):
            records[index]['Z_rms']=rms(result[0])
        handles.extend([layer.side.projection.register_forward_hook(projection),
                        layer.side.register_forward_hook(output)])
    try:
        model.eval()
        model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(records)!=4 or any(set(r)!= {'vbar_rms','Z_rms'} for r in records.values()):
        raise ValueError('step0 RMS hooks did not execute for every target layer')
    return {'scope':'first R1 training candidate; all valid input positions; before optimizer updates',
            'valid_tokens':int(valid.sum()),'eta':model.config.eta,'tap':model.config.tap,
            'layers':records}


@torch.no_grad()
def check_direct_hidden(model,ids):
    """Compare actual projection inputs against actual input_layernorm outputs."""
    if model.config.tap!='direct_hidden':
        raise ValueError('direct-hidden model required')
    expected,checks,handles={},{},[]
    for index,layer in model.adapters().items():
        def norm(module,args,result,index=index):
            expected[index]=result
        def projection(module,args,index=index):
            checks[index]=all(torch.equal(value,expected[index]) for value in args)
        handles.extend([layer.original.input_layernorm.register_forward_hook(norm),
                        layer.side.projection.register_forward_pre_hook(projection)])
    try:
        model.eval()(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return len(checks)==len(model.config.layers) and all(checks.values())


def run(args):
    from rb_train import E0028_ARMS, ETA_ARMS, PARAMETERS, adapter_config, make_model
    result={'contract':'E-0028@1','source_commit':os.environ['SR_CODE_COMMIT'],
            'model_revision':REVISION,'arms':{},'checks':{}}
    anchors={}
    ids=torch.tensor([[1,511,894,42,8]],device='cuda')
    for arm in ('RB1-09',*E0028_ARMS):
        model=make_model(arm)
        digest,state=initial_state(model)
        tap=model.config.tap
        if tap not in anchors:
            anchors[tap]=state
        same=state.keys()==anchors[tap].keys() and all(torch.equal(v,anchors[tap][k]) for k,v in state.items())
        init_ok=True
        for layer in model.adapters().values():
            side=layer.side
            init_ok &= bool(torch.count_nonzero(side.fusion.C_u)==0)
            init_ok &= torch.equal(side.reader.Rq,torch.eye(32,device=side.reader.Rq.device))
            init_ok &= torch.equal(side.reader.Ro,torch.eye(32,device=side.reader.Ro.device))
            init_ok &= all(bool(torch.count_nonzero(p)>0) for p in side.projection.parameters())
        config=asdict(model.config)
        stored=sum(p.numel() for p in model.adapter_parameters())
        tap_ok=check_direct_hidden(model,ids) if tap=='direct_hidden' else True
        anchor_config=asdict(adapter_config('X-01' if tap=='direct_hidden' else 'RB1-09'))
        # T0 differs only in branch; grid candidates differ only in eta.
        permitted='branch' if arm in ('X-02','X-03') else 'eta'
        config_pair={k:v for k,v in config.items() if k!=permitted}
        reference_pair={k:v for k,v in anchor_config.items() if k!=permitted}
        result['arms'][arm]={'config':config,'parameters':stored,'registered':PARAMETERS[arm],
            'init_sha256':digest,'elementwise_equal_to_tap_anchor':same,
            'zero_Cu_identity_R_nonzero_A':bool(init_ok),'input_layernorm_bitwise_equal':tap_ok,
            'only_permitted_config_difference':config_pair==reference_pair}
        del model,state
        torch.cuda.empty_cache()
    result['checks']={
        'e_paired_init':all(v['elementwise_equal_to_tap_anchor'] for v in result['arms'].values()),
        'f_X_is_input_layernorm_output':all(v['input_layernorm_bitwise_equal'] for v in result['arms'].values()),
        'h_eta_init_and_config':all(v['only_permitted_config_difference'] for v in result['arms'].values()),
        'initialization_rule':all(v['zero_Cu_identity_R_nonzero_A'] for v in result['arms'].values()),
        'd_parameter_registry':all(v['parameters']==v['registered'] for v in result['arms'].values())}
    result['passed']=all(result['checks'].values())
    write_json(args.output/'tap-registration.json',result)
    if not result['passed']:
        raise ValueError(f'E-0028 initialization checks failed: {result["checks"]}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
