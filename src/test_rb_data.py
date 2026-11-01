import copy

import pytest

from rb_data import (confirmation_manifest,apply_confirmation,training_stream,plan_batches)


def rows():
    return {'splits':{task:{'train':[{'task':task,'id':i} for i in range(1009)],
                           'dev':[{'task':task,'id':'dev'}],
                           'calibration':[{'task':task,'id':'calibration'}]}
                     for task in ('piqa','boolq')},'metadata':{}}


def test_A2_confirmation_selection_and_train_exclusion():
    original=rows()
    manifest=confirmation_manifest(original)
    assert manifest==confirmation_manifest(original)
    assert manifest['counts']=={'piqa':1000,'boolq':1000}
    kept=apply_confirmation(original,manifest)
    for task in ('piqa','boolq'):
        assert len(kept['splits'][task]['train'])==9
        assert not set(manifest['ids'][task]).intersection(r['id'] for r in kept['splits'][task]['train'])
        assert kept['splits'][task]['dev']==original['splits'][task]['dev']
        assert kept['splits'][task]['calibration']==original['splits'][task]['calibration']
    altered=copy.deepcopy(manifest)
    altered['ids']['piqa'][0]='foreign'
    with pytest.raises(ValueError):
        apply_confirmation(original,altered)


def test_RB_order_seed_epoch_and_exclusions():
    original=rows()
    kept=apply_confirmation(original,confirmation_manifest(original))
    first=training_stream(kept,'R1',0,0)
    assert first==training_stream(kept,'R1',0,0)
    assert first!=training_stream(kept,'V0',0,0)
    assert first!=training_stream(kept,'R1',0,1)
    assert [r['task'] for r in first[:6]]==['piqa','boolq']*3
    excluded=[(first[0]['task'],first[0]['id'])]
    filtered=training_stream(kept,'R1',0,0,excluded)
    assert len(filtered)==len(first)-1 and first[0] not in filtered


def test_planning_full_steps_epoch_carry_and_no_confirmation_encoding(monkeypatch):
    original=rows()
    kept=apply_confirmation(original,confirmation_manifest(original))
    ids={task:{r['id'] for r in kept['splits'][task]['train']} for task in ('piqa','boolq')}
    def encode(tok,row,max_length):
        assert row['id'] in ids[row['task']]
        return {'task':row['task'],'id':row['id'],'input_ids':[1]*7,'labels':[-100]*6+[2]}
    monkeypatch.setattr('rb_data.encode_training_example',encode)
    batches=plan_batches(kept,None,'R1',0,token_budget=31)
    assert len(batches)==3 and sum(len(e['input_ids']) for b in batches for e in b)==42
    batches=plan_batches(kept,None,'R1',0,token_budget=None)
    assert sum(len(b) for b in batches)==54
    assert len(batches)==27 and all(len(b)==2 for b in batches)
    assert sorted(e['epoch'] for b in batches for e in b)==[0]*18+[1]*18+[2]*18
