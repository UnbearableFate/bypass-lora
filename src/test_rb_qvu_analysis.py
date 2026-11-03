"""Exact boundary and split-stratified pairing checks for E-0031 rules."""
from fractions import Fraction as F

import pytest

from rb_qvu_analysis import paired_intervals, rules


def scores(h=F(81,100), t=F(79,100), x=F(80,100), l=F(81,100)):
    merged = {'H-QVU':h, 'T0-QVU':t, 'X-01':x, 'RB1-07':l}
    return merged, dict(merged, Base=F(70,100))


def test_inclusive_exact_practical_and_history_gates():
    merged, dev = scores(t=F(80,100))
    value = rules(merged,dev,F(1,100))
    assert value['3_H0030']['outcome'] == 'supports'
    assert value['4_H0031']['outcome'] == 'supports'
    assert value['6_conditional_controls']['triggered']
    assert value['7_replication']['propose_two_new_seeds']


@pytest.mark.parametrize('h,t,swap,h30,h31', [
    (F(80,100),F(80,100),F(2,100),'challenges','challenges'),
    (F(805,1000),F(79,100),F(2,100),'inconclusive','supports'),
    (F(81,100),F(805,1000),F(2,100),'supports','inconclusive'),
    (F(81,100),F(79,100),F(999,100000),'supports','inconclusive')])
def test_unrounded_boundaries_and_swap_cannot_replace_history(h,t,swap,h30,h31):
    merged,dev = scores(h=h,t=t)
    value = rules(merged,dev,swap)
    assert value['3_H0030']['outcome'] == h30
    assert value['4_H0031']['outcome'] == h31


def test_lora_floor_and_tokenwise_practical_trigger_are_independent():
    merged,dev = scores(h=F(81,100),t=F(82,100),l=F(82,100))
    value = rules(merged,dev,F(2,100))
    assert value['3_H0030']['outcome'] == 'inconclusive'
    assert value['4_H0031']['outcome'] == 'challenges'
    assert value['5_T0_practical']['passes']
    assert value['6_conditional_controls']['triggered']
    assert not value['10_route']['coordinator_stop_review_required']


def test_no_practical_signal_triggers_only_coordinator_stop_review():
    merged,dev = scores(h=F(80,100),t=F(80,100))
    value = rules(merged,dev,F(0))
    assert not value['6_conditional_controls']['triggered']
    assert value['10_route']['coordinator_stop_review_required']


def rows(offset=0):
    return [dict(task=t,split=s,id=offset+100*i+j,gold=0,prediction=0,gold_loss_tokens=1)
        for i,s in enumerate(('dev','calibration')) for t in ('piqa','boolq') for j in range(3+i)]


def test_common_stratified_indices_preserve_exact_paired_differences():
    left = rows()
    right = [dict(r,prediction=1) for r in left]
    result = paired_intervals({'left':left,'right':right},repeats=217)
    assert result['left_minus_right']['ci95_pp'] == pytest.approx([100,100])
    assert result['left_minus_right']['estimate_pp'] == 100
    assert result == paired_intervals({'right':right,'left':left},repeats=217)


@pytest.mark.parametrize('mutation', ['gold','id','split','gold_loss_tokens','order'])
def test_pairing_rejects_consequential_membership_or_label_changes(mutation):
    left = rows(); right = [dict(r) for r in left]
    if mutation == 'order':
        right.reverse()
    else:
        right[0][mutation] = 'calibration' if mutation == 'split' else 2
    with pytest.raises(ValueError,match='paired rows'):
        paired_intervals({'left':left,'right':right},repeats=2)


@pytest.fixture
def analysis_fixture(tmp_path,monkeypatch):
    """Real paired prediction files; only unrelated checkpoint I/O is stubbed."""
    import json
    from pathlib import Path
    import rb_qvu_analysis as entry
    source = 'a'*40
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    def predictions(split,correct):
        size = 1000 if split == 'dev' else 2048
        return [dict(task=task,split=split,id=i+(1000 if split == 'calibration' else 0),
            gold=0,prediction=int(i >= int(size*correct)),scores=[-1.,-2. if i < int(size*correct) else 0.],
            gold_nll_sum=1.,gold_loss_tokens=1) for task in ('piqa','boolq') for i in range(size)]
    def metric(rows):
        value = {}
        for split in {r['split'] for r in rows}:
            value[f'{split}.nll'] = 1.
            for task in ('piqa','boolq'):
                subset = [r for r in rows if r['split'] == split and r['task'] == task]
                value.update({f'{split}.{task}.nll':1.,f'{split}.{task}.accuracy':sum(r['prediction']==0 for r in subset)/len(subset),
                    f'{split}.{task}.examples':len(subset),f'{split}.{task}.loss_tokens':len(subset)})
        return value
    roots,refs,manifests = {},{},{}
    def producer(arm,name,correct,new=False):
        root = tmp_path/name;roots[arm] = root
        dev = predictions('dev',correct);cal = predictions('calibration',correct)
        metrics = metric(dev+cal)
        metrics.update({'source.commit':source,'split_manifest.sha256':entry.SPLIT_SHA256,
            'plan.order_sha256':entry.ORDER_SHA256,'train.steps':13115,'train.examples':26230,
            'train.input_tokens':2000069,'train.loss_tokens':387653,'learning_rate':.001,
            'hardware':'NVIDIA GH200 120GB MIG 2g.24gb','exec.pbs_job_id':f'{name}.opbs',
            'init.parameters.sha256':'paired-init','reload.identical':True})
        manifest = dict(status='complete',validator={'passed':True},metrics=metrics)
        manifests[arm] = manifest
        write(root/'manifest.json',manifest)
        write(root/'predictions.json',dev);write(root/'calibration-predictions.json',cal)
        refs[arm] = {'manifest':entry.reference(root/'manifest.json')}
        if new:
            base_rows = predictions('dev',.7)
            write(root/'dev-step-0.json',{'predictions':base_rows})
            write(root/'dev-base-same-allocation.json',{'predictions':base_rows,'job_id':metrics['exec.pbs_job_id'],
                'hardware':metrics['hardware'],'source_commit':source})
            write(root/'iv-off.json',{'predictions':base_rows})
            for step in (3279,6558,9837):
                write(root/f'dev-step-{step}.json',{'step':step,'predictions':dev,'metrics':metric(dev)})
        return root
    producer('Base','Base',.7)
    for arm,score in [('X-01',.8),('X-02',.8)]:
        producer(arm,arm,score)
    # RB1-02 is deliberately not created: no attempted read should occur.
    for arm,attempt in entry.LEGACY.items():
        if arm != 'RB1-02':
            producer(arm,attempt,.82 if arm == 'RB1-07' else .78)
    producer('H-QVU','H-QVU',.84,new=True)
    producer('T0-QVU','T0-QVU',.80,new=True)
    def formal(root,arm,source_commit=None):
        assert arm != 'RB1-02', 'unavailable checkpoint was accessed'
        return manifests[arm],dict(refs[arm]),Path(root)/'adapter'
    monkeypatch.setattr(entry,'validate_formal_run',formal)
    monkeypatch.setattr(entry,'ACCEPTED_BASE_MANIFEST_SHA256',entry.reference(roots['Base']/'manifest.json')['sha256'])
    monkeypatch.setattr(entry,'X_FILES',{arm:{name:entry.reference(roots[arm]/name)['sha256'] for name in names}
        for arm,names in entry.X_FILES.items()})
    # Numerical bootstrap is tested separately; retain common-index resampling
    # here while bounding the end-to-end control-flow regression to13replicates.
    bootstrap = entry.paired_intervals
    monkeypatch.setattr(entry,'paired_intervals',lambda rows:bootstrap(rows,repeats=13))
    reevaluations = {'RB1-02':{'unavailable':'RuntimeError loading historical adapter under current code'}}
    for arm in ('RB1-01','RB1-07','RB1-09'):
        root = tmp_path/('re-'+arm)
        dev = json.loads((roots[arm]/'predictions.json').read_text())
        cal = json.loads((roots[arm]/'calibration-predictions.json').read_text())
        write(root/'manifest.json',{'status':'complete','validator':{'passed':True},'arm':arm,
            'stage':'checkpoint_evaluation','metrics':{**metric(dev+cal),'source.commit':source,
                'split_manifest.sha256':entry.SPLIT_SHA256,'compatibility':{'flips':0,'same_profile':True}}})
        write(root/'predictions.json',dev);write(root/'calibration-predictions.json',cal)
        write(root/'provenance.json',{'training_run':refs[arm]})
        reevaluations[arm] = str(root)
    smoke = tmp_path/'smoke'
    write(smoke/'manifest.json',{'status':'complete','validator':{'passed':True},
        'metrics':{'pytest_exit':0,'source_commit':source}})
    write(smoke/'registration.json',{arm:{'initial_sha256':'paired-init'} for arm in ('H-QVU','T0-QVU')})
    write(smoke/'cost-estimate.json',{'fixture':'cost already checked independently'})
    intervention = tmp_path/'intervention'
    write(intervention/'manifest.json',{'status':'complete','validator':{'passed':True},
        'arm':'H-QVU','stage':'intervention','metrics':{'source.commit':source}})
    write(intervention/'provenance.json',{'training_run':refs['H-QVU']})
    write(intervention/'conditions.json',{kind:{'predictions':p,'metrics':metric(p)} for kind,p in
        [('normal',predictions('dev',.84)),('iv_swap',predictions('dev',.81))]})
    return {'source_commit':source,'base_run':str(roots['Base']),
        'references':{a:str(p) for a,p in roots.items() if a not in ('Base','H-QVU','T0-QVU')},
        'reevaluations':reevaluations,'training_runs':{a:str(roots[a]) for a in ('H-QVU','T0-QVU')},
        'smoke_run':str(smoke),'intervention_run':str(intervention)}


def test_unavailable_rule8_checkpoint_does_not_block_primary_analysis(analysis_fixture):
    from rb_qvu_analysis import analyze
    result = analyze(analysis_fixture)
    assert result['rules']['3_H0030']['outcome'] == 'supports'
    assert result['rules']['4_H0031']['outcome'] == 'supports'
    assert 'RB1-02' not in result['merged_performance']
    resolution = result['rules']['8_resolution']
    assert resolution['unavailable']['RB1-02'].startswith('RuntimeError loading')
    assert not resolution['comparisons']['RB1-02_minus_RB1-01']['available']
    assert 'ci95_pp' in resolution['comparisons']['RB1-07_minus_X-01']


def test_primary_lora_checkpoint_cannot_use_diagnostic_unavailable_escape(analysis_fixture):
    from rb_qvu_analysis import analyze
    analysis_fixture['reevaluations']['RB1-07'] = {'unavailable':'checkpoint missing'}
    with pytest.raises(ValueError,match='mandatory for rule3'):
        analyze(analysis_fixture)


@pytest.mark.parametrize('entry',[{}, {'unavailable':''}, {'unavailable':'reason','other':1}])
def test_unavailable_diagnostic_requires_a_reason_and_no_extra_override_fields(entry):
    from rb_qvu_analysis import LEGACY, legacy_inputs
    spec = {a:'/unused/'+a for a in LEGACY}
    spec['RB1-02'] = entry
    with pytest.raises(ValueError,match='explicit reason'):
        legacy_inputs(spec)
