import json
from pathlib import Path

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
