"""Consequential ordering and same-job fallback checks for rule 1(g)."""
import json
from types import SimpleNamespace

import pytest
import torch

import rb_tap_eval as entry
from rb_tap_identity import LEGACY_COMMIT,PINNED_PROFILE


def fixture(tmp_path,monkeypatch,hardware=PINNED_PROFILE):
    origin=tmp_path/'origin';origin.mkdir()
    old=[{'task':'piqa','id':1,'gold':0,'prediction':0}]
    new=[dict(old[0],prediction=1)]
    (origin/'predictions.json').write_text(json.dumps(old))
    split=tmp_path/'split.json';split.write_text('{}')
    import rb_data
    monkeypatch.setattr(entry,'validate_formal_run',lambda *a:({'metrics':{
        'hardware':PINNED_PROFILE,'split_manifest.sha256':entry.reference(split)['sha256']}},{},origin))
    monkeypatch.setattr(entry.AutoTokenizer,'from_pretrained',lambda *a,**k:SimpleNamespace(eos_token_id=0))
    monkeypatch.setattr(entry,'load_data',lambda *a:({},[]))
    monkeypatch.setattr(entry,'make_model',lambda *a,**k:torch.nn.Identity())
    monkeypatch.setattr(entry,'initial_state',lambda *a:('b'*64,{}))
    monkeypatch.setattr(entry,'step0_diagnostics',lambda *a:{})
    monkeypatch.setattr(rb_data,'plan_batches',lambda *a:[[{}]])
    monkeypatch.setattr(entry.torch.cuda,'get_device_name',lambda:hardware)
    monkeypatch.setattr(entry,'validate_calibration',lambda *a:None)
    monkeypatch.setenv('SR_CODE_COMMIT','a'*40)
    monkeypatch.setenv('SR_SPEC_DIGEST','unit fixture')
    monkeypatch.setenv('PBS_JOBID','123.opbs')
    calls=[]
    def evaluate(*args,split_names):
        calls.append(split_names)
        return {},new
    monkeypatch.setattr(entry,'evaluate',evaluate)
    args=SimpleNamespace(training_run=origin,data_dir=tmp_path,split_manifest=split,
                         output=tmp_path/'out',calibration_only=False)
    return args,calls,new


def test_calibration_is_retained_before_same_profile_prediction_gate_fails(tmp_path,monkeypatch):
    args,calls,new=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='rule 1.g.'):
        entry.run(args)
    assert calls==[('dev',),('calibration',)]
    assert json.loads((args.output/'calibration-predictions.json').read_text())==new
    manifest=json.loads((args.output/'manifest.json').read_text())
    assert manifest['status']=='partial' and manifest['validator']['passed'] is False


def test_cross_profile_uses_accepted_source_in_same_allocation(tmp_path,monkeypatch):
    hardware='NVIDIA GH200 120GB MIG 1g.24gb'
    args,calls,new=fixture(tmp_path,monkeypatch,hardware)
    def legacy(adapter,data_dir,split,out):
        # The calibration file must exist before even the fallback comparison.
        assert (out/'calibration-predictions.json').exists()
        value={'predictions':new,'metrics':{},'hardware':hardware,'job_id':'123.opbs','source_commit':LEGACY_COMMIT}
        (out/'legacy-source-dev.json').write_text(json.dumps(value))
        return value,{}
    monkeypatch.setattr(entry,'evaluate_legacy_source',legacy)
    entry.run(args)
    manifest=json.loads((args.output/'manifest.json').read_text())
    check=manifest['metrics']['compatibility']
    assert manifest['status']=='complete' and check['passed'] is True
    assert check['predictions_equal_pinned'] is False and check['gate_reference']=='same_allocation'
