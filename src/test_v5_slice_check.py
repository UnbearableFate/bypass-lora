import json
from types import SimpleNamespace

import pytest

from v5_train import CONTRACT,lock_lrs,write_json
from v5_slice_check import analyze,PAIRS


def fixtures(tmp_path,new_c=1.591):
    old_paths,new_paths = [],[]
    values = {'V5-C':[1.7,1.63,1.59,1.60],'V5-F':[1.65,1.60,1.56,1.61],
              'V5-L':[1.66,1.58,1.55,1.64]}
    for arm in values:
        for lr,nll in zip((1e-4,3e-4,1e-3,3e-3),values[arm]):
            old = tmp_path/'original'/f'{arm}-{lr}'/'manifest.json'
            old.parent.mkdir(parents=True)
            value = {'contract':CONTRACT,'phase':'calibration','seed':0,'arm':arm,
                     'spec_digest':f'{arm}-{lr}','status':'complete','validator':{'passed':True},
                     'metrics':{'learning_rate':lr,'calibration.nll':nll,
                     'hardware':'GPU MIG '+('1g.24gb' if (arm,lr) in PAIRS else '2g.24gb'),
                     'plan.order_sha256':'same','plan.steps':1101,'plan.input_tokens':200103},'artifacts':[]}
            write_json(old,value)
            old_paths.append(old)
            if (arm,lr) not in PAIRS:
                new_paths.append(old)
                continue
            new = tmp_path/'reruns'/f'{arm}-{lr}'/'manifest.json'
            new.parent.mkdir(parents=True)
            profile = new.parent/'mig-profile.json'
            write_json(profile,{'passed':True,'profile':'2g.24gb','uuid':'MIG-test',
                                'cuda_visible_devices':'MIG-test'})
            value['metrics']['hardware'] = 'GPU MIG 2g.24gb'
            value['metrics']['calibration.nll'] = new_c if arm=='V5-C' else 1.659
            value['artifacts'] = [{'path':str(profile),'role':'mig-profile'}]
            write_json(new,value)
            new_paths.append(new)
    original_lock = tmp_path/'lr-lock.json'
    lock_lrs(SimpleNamespace(manifests=old_paths,output=original_lock))
    return original_lock,new_paths


@pytest.mark.parametrize('new_c,same,chosen_c',[(1.591,True,1e-3),(1.625,False,3e-3)])
def test_recomputes_separate_lock_and_preserves_formal_input(tmp_path,new_c,same,chosen_c):
    original,paths = fixtures(tmp_path,new_c)
    before = original.read_bytes()
    report = analyze(paths,original,tmp_path/'report')
    assert original.read_bytes() == before
    assert report['choices_match'] is same
    assert report['check_locked_lrs']['V5-C'] == chosen_c
    delta = next(row['delta_2g_minus_1g'] for row in report['comparisons'] if row['arm']=='V5-C')
    assert delta == pytest.approx(new_c-1.59)
    assert report['check_lock'] != str(original)


def test_requires_actual_guard_evidence_for_replacement(tmp_path):
    original,paths = fixtures(tmp_path)
    path = next(path for path in paths if '/reruns/' in str(path))
    profile = path.parent/'mig-profile.json'
    value = json.loads(profile.read_text())
    value['profile'] = '1g.24gb'
    write_json(profile,value)
    with pytest.raises(ValueError,match='guard evidence'):
        analyze(paths,original,tmp_path/'report')


def test_rejects_changed_original_calibration_provenance(tmp_path):
    original,paths = fixtures(tmp_path)
    lock = json.loads(original.read_text())
    ref = next(ref for ref in lock['calibration_manifests'] if (ref['arm'],ref['lr']) in PAIRS)
    path = next(path for path in tmp_path.glob('original/*/manifest.json') if str(path)==ref['path'])
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='provenance mismatch'):
        analyze(paths,original,tmp_path/'report')
