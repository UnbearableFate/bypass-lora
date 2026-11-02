from fractions import Fraction as F
import math

import pytest

from rb_eta_select import GRID, best_eta, choose_eta, extension_eta


def grid(winner=.25):
    return {eta:F(4 if eta==winner else 3,5) for eta in GRID}


def test_ties_prefer_nearest_baseline_then_lower_eta():
    candidates={eta:F(3,4) for eta in GRID}
    assert choose_eta(candidates)['eta']==.25
    candidates[.25]=F(1,2)
    assert best_eta(candidates)==.125
    candidates[.125]=candidates[.5]=F(1,2)
    assert best_eta(candidates)==.0625


@pytest.mark.parametrize('endpoint,extension',[(.0625,.03125),(2.,4.)])
def test_endpoint_extends_exactly_once_and_new_boundary_is_selected(endpoint,extension):
    candidates=grid(endpoint)
    assert extension_eta(candidates)==extension
    with pytest.raises(ValueError,match='endpoint extension'):
        choose_eta(candidates)
    candidates[extension]=F(9,10)
    assert choose_eta(candidates)=={'eta':extension,'extension':extension,'grid_boundary':True}


def test_endpoint_tie_uses_same_tie_rule_before_extending():
    candidates=grid(.0625);candidates[.125]=candidates[.0625]
    assert extension_eta(candidates) is None
    assert choose_eta(candidates)['eta']==.125


def test_nonfinite_training_is_excluded_but_missing_point_is_not():
    candidates=grid();candidates[.25]=None
    assert choose_eta(candidates)['eta']==.125
    with pytest.raises(ValueError,match='six-point'):
        choose_eta({k:v for k,v in candidates.items() if v is not None})
    with pytest.raises(ValueError,match='no finite'):
        choose_eta({eta:None for eta in GRID})


@pytest.mark.parametrize('score',[float('nan'),float('inf'),.8,F(101,100)])
def test_nonfraction_or_invalid_accuracy_is_rejected(score):
    candidates=grid();candidates[.25]=score
    with pytest.raises(ValueError,match='exact finite'):
        choose_eta(candidates)


def test_untriggered_or_double_endpoint_extension_is_rejected():
    for candidates in (grid()|{4.:F(9,10)},grid(2.)|{4.:F(9,10),.03125:F(1,2)}):
        with pytest.raises(ValueError,match='endpoint extension'):
            choose_eta(candidates)


def candidate(tmp_path,failed=False,failure_kind=None,status='complete'):
    import json
    from rb_eta_select import SPLIT_SHA256,ORDER_SHA256
    path=tmp_path/'manifest.json'
    metrics={'split_manifest.sha256':SPLIT_SHA256,'plan.order_sha256':ORDER_SHA256,
             'learning_rate':.001,'failed':failed,'failure_kind':failure_kind,
             'train.steps':13115,'train.examples':26230,'reload.identical':True}
    rows=[]
    for task in ('piqa','boolq'):
        rows.extend({'task':task,'split':'calibration','id':i,'gold':0,'prediction':0,
            'scores':[-1.,-2.],'gold_nll_sum':1.,'gold_loss_tokens':1} for i in range(2048))
        metrics.update({f'calibration.{task}.{k}':v for k,v in
            {'accuracy':1.,'nll':1.,'examples':2048,'loss_tokens':2048}.items()})
    metrics['calibration.nll']=1.
    value={'status':status,'validator':{'passed':True},'arm':'ETA-X-1',
           'seed':0,'phase':'R1','contract':'E-0028@1','stage':'train','metrics':metrics}
    path.write_text(json.dumps(value))
    (tmp_path/'calibration-predictions.json').write_text(json.dumps(rows))
    return path


def test_candidate_validation_uses_calibration_counts_and_binds_input_hashes(tmp_path):
    import json
    from rb_eta_select import load_candidate
    path=candidate(tmp_path)
    value=json.loads(path.read_text());value['metrics']['dev.piqa.accuracy']=0.
    path.write_text(json.dumps(value))
    tap,eta,score,record=load_candidate(path)
    assert (tap,eta,score)==('X',1.,F(1))
    assert set(record['inputs'])=={'manifest','calibration_predictions'}
    rows=json.loads((tmp_path/'calibration-predictions.json').read_text())
    rows[1]['id']=rows[0]['id']
    (tmp_path/'calibration-predictions.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='duplicate'):
        load_candidate(path)


@pytest.mark.parametrize('status,failed,kind',[
    ('partial',True,'training_nonfinite'),('complete',True,'oom'),('complete',True,'evaluation_nonfinite')])
def test_engineering_failure_is_not_silently_eliminated(tmp_path,status,failed,kind):
    from rb_eta_select import load_candidate
    with pytest.raises(ValueError,match='rerun|only non-finite'):
        load_candidate(candidate(tmp_path,failed,kind,status))


def test_explicit_nonfinite_training_candidate_is_eliminated(tmp_path):
    from rb_eta_select import load_candidate
    tap,eta,score,record=load_candidate(candidate(tmp_path,True,'training_nonfinite'))
    assert score is None and record['excluded']=='training_nonfinite'
