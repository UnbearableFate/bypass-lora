from dataclasses import asdict, replace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from bypass_model import BranchModel
from rb_tap_check import check_direct_hidden, initial_state
from rb_train import E0028_ARMS, ETA_ARMS, PARAMETERS, adapter_config


def tiny_base():
    config=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=32,attention_dropout=0.)
    config._attn_implementation='eager'
    return Qwen3ForCausalLM(config).eval()


def wrapped(arm):
    base=tiny_base()
    torch.manual_seed(916)
    return BranchModel(base,replace(adapter_config(arm),layers=(0,1),rank=4))


@pytest.mark.parametrize('anchor,control',[('X-01','X-02'),('RB1-09','X-03')])
def test_paired_T0_and_memory_init_elementwise_identical(anchor,control):
    left=initial_state(wrapped(anchor));right=initial_state(wrapped(control))
    assert left[0]==right[0]
    assert left[1].keys()==right[1].keys()
    assert all(torch.equal(value,right[1][key]) for key,value in left[1].items())


@pytest.mark.parametrize('arm',ETA_ARMS)
def test_eta_only_changes_reader_scalar_and_not_parameters(arm):
    tap,eta=ETA_ARMS[arm]
    anchor='X-01' if tap=='X' else 'RB1-09'
    expected=asdict(adapter_config(anchor));expected['eta']=eta
    assert asdict(adapter_config(arm))==expected
    left=wrapped(anchor);right=wrapped(arm)
    assert initial_state(left)[0]==initial_state(right)[0]
    assert all(layer.side.reader.eta==eta for layer in right.adapters().values())


def test_direct_hidden_projection_receives_exact_normalized_layer_input():
    model=wrapped('X-01')
    # The checker exercises the real layer forward and all three projection inputs.
    assert check_direct_hidden(model,torch.tensor([[1,2,3,4]]))
    for layer in model.adapters().values():
        assert layer.side.projection.A_q.shape==layer.side.projection.A_k.shape==layer.side.projection.A_v.shape==(16,4)


def test_X_T0_and_eta_configs_keep_locked_structure_and_counts():
    for arm in E0028_ARMS:
        cfg=adapter_config(arm)
        assert cfg.fusion=='C_up' and cfg.rank==32 and cfg.steps==1
        assert cfg.impl=='prefix' and cfg.mode=='inclusive'
        assert cfg.branch==('kernel_tokenwise' if arm in ('X-02','X-03') else 'memory')
        width=1024 if cfg.tap=='direct_hidden' else 2048
        assert PARAMETERS[arm]==4*(32*(width+1024+1024)+2*32**2+32*3072)
