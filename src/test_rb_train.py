import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rb_train import GRIDS,EXTENSIONS,REPRESENTATIVES,extension_lr,lock_lrs,adapter_config


@pytest.mark.parametrize('family',list(GRIDS))
def test_family_extension_uses_raw_minimum_not_one_percent_tie(family):
    grid=GRIDS[family]
    assert extension_lr(family,dict(zip(grid,[1.009,1.005,1.])))==EXTENSIONS[family][1]
    assert extension_lr(family,dict(zip(grid,[1.,1.005,1.009])))==EXTENSIONS[family][0]
    assert extension_lr(family,dict(zip(grid,[1.1,1.,1.2]))) is None


def calibration_inputs(tmp_path):
    paths=[]
    for family,grid in GRIDS.items():
        for lr,nll in zip(grid,[1.1,1.,1.2]):
            value={'stage':'calibration','phase':'V0','seed':0,'arm':REPRESENTATIVES[family],
                'family':family,'status':'complete','validator':{'passed':True},
                'metrics':{'learning_rate':lr,'calibration.nll':nll,'split_manifest.sha256':'a'*64,
                    'plan.order_sha256':'b'*64,'plan.steps':1234,'train.steps':1234,'train.input_tokens':200100}}
            path=tmp_path/(family+'-'+str(lr)+'.json')
            path.write_text(json.dumps(value));paths.append(path)
    return paths


def test_lock_family_mapping_and_nonfailed_null_rejected(tmp_path):
    paths=calibration_inputs(tmp_path)
    result=lock_lrs(paths)
    assert result['locked_lrs']=={f:grid[1] for f,grid in GRIDS.items()}
    assert len(result['inputs'])==12
    value=json.loads(paths[0].read_text())
    value['metrics']['calibration.nll']=None
    paths[0].write_text(json.dumps(value))
    with pytest.raises(ValueError,match='finite positive NLL'):lock_lrs(paths)


def test_lock_refuses_changed_order_and_missing_extension(tmp_path):
    paths=calibration_inputs(tmp_path)
    value=json.loads(paths[0].read_text())
    value['metrics']['plan.order_sha256']='c'*64
    paths[0].write_text(json.dumps(value))
    with pytest.raises(ValueError,match='order'):lock_lrs(paths)
    value['metrics']['plan.order_sha256']='b'*64
    value['metrics']['calibration.nll']=.8
    paths[0].write_text(json.dumps(value))
    with pytest.raises(ValueError,match='endpoint'):lock_lrs(paths)


def test_registered_loop_controls_are_explicit_and_defaults_unchanged():
    assert adapter_config('RB1-02').steps==1
    assert adapter_config('RB1-03').steps==2
    assert adapter_config('RB1-04').steps==4
    assert adapter_config('RB1-05').fixed_query is True
    assert adapter_config('RB1-06').eta==.0625
    assert adapter_config('RB1-08').fusion=='C_dual'
    assert adapter_config('RB1-09').fusion=='C_up'


def test_T0_rejects_backend_that_would_create_an_unused_temperature():
    from bypass_model import AdapterConfig
    with pytest.raises(ValueError,match='parameter family'):
        AdapterConfig(branch='kernel_tokenwise',backend='centered_cos_slot')


@pytest.mark.parametrize('double_writes',[False,True])
def test_train_accounting_gate_rejects_doubled_writes(tmp_path,monkeypatch,double_writes):
    import torch
    import rb_train
    from bypass_model import BranchModel
    model=BranchModel.__new__(BranchModel)
    torch.nn.Module.__init__(model)
    model.config=adapter_config('RB1-02')
    def forward(self,*args):
        return SimpleNamespace(accounting={6:{'base':{'attention':1,'ffn':1},
            'side':{'valid_tokens':torch.tensor(1),'writes':torch.tensor(2 if double_writes else 1),
                    'reads':torch.tensor(1)}}},cache=SimpleNamespace(memory={6:None}))
    monkeypatch.setattr(BranchModel,'forward',forward)
    def fake_train(current,*args,**kwargs):
        current(None);current(None)
        return {'train.input_tokens':2,'train.examples':2}
    monkeypatch.setattr(rb_train,'v1_train',fake_train)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    if double_writes:
        with pytest.raises(AssertionError):
            rb_train.train(model,[],None,.001,tmp_path)
    else:
        result=rb_train.train(model,[],None,.001,tmp_path)
        assert result['runtime.accounting']['layers']['6']['writes']==2


def test_calibration_entry_evaluates_once_without_export_or_reload(tmp_path,monkeypatch):
    import torch
    import rb_train
    split=tmp_path/'split.json';split.write_text('{}')
    args=SimpleNamespace(output=tmp_path/'run',stage='calibration',phase='R1',seed=0,
        arm='RB1-02',lr=.001,lr_lock=None,split_manifest=split,data_dir='fixture',
        three_epochs=False,eval_limit=None,smoke_steps=None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.new=torch.nn.Parameter(torch.zeros(rb_train.PARAMETERS[args.arm]))
        def adapter_parameters(self):return self.parameters()
        def export_adapter(self,*args):pytest.fail('calibration must not export')
    loads=[];evaluations=[]
    def make(arm,phase,seed):
        loads.append((arm,phase,seed));return Model()
    def evaluate(model,splits,tok,limit,split_names):
        evaluations.append(split_names)
        return {'calibration.nll':1.2},[{'id':1,'prediction':0}]
    monkeypatch.setattr(rb_train.AutoTokenizer,'from_pretrained',lambda *a,**k:SimpleNamespace(eos_token_id=0))
    monkeypatch.setattr(rb_train,'load_data',lambda *a:({},[]))
    monkeypatch.setattr(rb_train,'plan_batches',lambda *a:[None])
    monkeypatch.setattr(rb_train,'plan_metadata',lambda *a:{'plan.steps':1,'plan.order_sha256':'b'*64})
    monkeypatch.setattr(rb_train,'make_model',make)
    monkeypatch.setattr(rb_train,'train',lambda *a:{'params.with_grad':rb_train.PARAMETERS[args.arm],
        'train.steps':1,'train.input_tokens':200001})
    monkeypatch.setattr(rb_train,'evaluate',evaluate)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'unit fixture')
    monkeypatch.setenv('SR_SPEC_DIGEST','unit-fixture')
    rb_train.run(args)
    value=json.loads((args.output/'manifest.json').read_text())
    assert value['status']=='complete' and value['validator']['passed'] is True
    assert value['phase']=='V0' and value['stage']=='calibration'
    assert loads==[('RB1-02','V0',0)] and evaluations==[('calibration',)]
    assert value['metrics']['calibration.export_reload_skipped'] is True
    assert value['metrics']['failed'] is False and value['metrics']['calibration.nll']==1.2
    assert 'reload.identical' not in value['metrics'] and 'save.seconds' not in value['metrics']
    assert not (args.output/'adapter').exists() and not (args.output/'checkpoint.json').exists()
