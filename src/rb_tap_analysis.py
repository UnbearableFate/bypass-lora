"""E-0028 locked rules and artifact-bound dev analysis. Execute only in PBS.

An eta-lock, rebuilt from calibration inputs, is mandatory before this module
opens any new dev results. It records screening outcomes, never route decisions
or hypothesis status changes.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path

from rb1_analysis import (ACCEPTED_BASE_MANIFEST_SHA256,check_metrics,measured_cost,
                          validate_read_summary,validate_rows)
from rb1_artifacts import finite_tree,read,reference,validate_formal_run
from rb1_intervene import MODES
from rb1_rules import TASKS,counts_from_rows,paired_bootstrap
from rb_eta_select import ORDER_SHA256,SPLIT_SHA256,selection
from rb_train import E0028_ARMS,ETA_ARMS,PARAMETERS,TAP_ARMS
from v1_analysis import _read_run
from v5_train import write_json

ONE_PP=Fraction(1,100)


def accuracy(rows):
    counts=counts_from_rows(rows)
    return sum((Fraction(*counts[t]) for t in TASKS),Fraction())/2


def pp(delta):
    return {'numerator':delta.numerator,'denominator':delta.denominator,'percentage_points':float(100*delta)}


def direction(delta):
    return 'supports' if delta>=ONE_PP else 'challenges' if delta<=0 else 'descriptive'


def rule_outcomes(scores,selected,swap_drop,passed=True):
    """Exact rational gate arithmetic; no CI/rounding enters any decision."""
    if not passed:
        raise ValueError('implementation checks failed; scientific outcomes withheld')
    tap_delta=scores['X-01']-scores['RB1-09']
    history={'X':scores['X-01']-scores['X-02'],'QK':scores['RB1-09']-scores['X-03']}
    eta_delta={tap:scores[arm]-scores['X-01' if tap=='X' else 'RB1-09'] for tap,arm in selected.items()}
    eta_direction={tap:direction(delta) for tap,delta in eta_delta.items()}
    h22=('supports' if 'supports' in eta_direction.values() else 'challenges'
         if set(eta_direction.values())=={'challenges'} else 'inconclusive')
    lora={arm:scores[arm]-scores['RB1-07'] for arm in {'X-01',*selected.values()}}
    best_tap=scores[selected['X']]-scores[selected['QK']]
    return {
        '2_learnability':{arm:{'at_least_Base_plus_1pp':score-scores['Base']>=ONE_PP,
                              'minus_Base':pp(score-scores['Base'])} for arm,score in scores.items()
                          if arm in E0028_ARMS},
        '3_H0021':{'d_tap':pp(tap_delta),'outcome':direction(tap_delta),'evidence_level':'screening'},
        '4_history':{tap:{'d_hist':pp(delta),'iv_swap_drop':pp(swap_drop[tap]),
                          'ask_coordinator_about_H0014':delta>=ONE_PP and swap_drop[tap]>=ONE_PP}
                     for tap,delta in history.items()},
        '6_H0022':{'per_tap':{tap:{'d_eta':pp(delta),'outcome':eta_direction[tap],
                                  'selected_arm':selected[tap]} for tap,delta in eta_delta.items()},
                    'outcome':h22,'evidence_level':'screening',
                    'new_two_seed_replication_required':h22=='supports'},
        '7_LoRA':{'comparisons':{arm:pp(delta) for arm,delta in lora.items()},
                  'any_at_least_1pp_above':any(delta>=ONE_PP for delta in lora.values()),
                  'new_two_seed_replication_required':any(delta>=ONE_PP for delta in lora.values()),
                  'limitation':'side branch additionally searches eta; single seed'},
        '8_descriptive':{'X02_minus_X03':pp(scores['X-02']-scores['X-03']),
                          'd_tap_star':pp(best_tap)},
        '10_no_more_ordinary_tap_variants':{
            'triggered':(direction(tap_delta)=='challenges' and best_tap<=0) or
                          scores[selected['X']]<scores['RB1-07']-ONE_PP,
            'coordinator_decision_required':True}}


def same_predictions(left,right):
    return [(r['task'],r['id'],r['gold'],r['prediction']) for r in left]==[(r['task'],r['id'],r['gold'],r['prediction']) for r in right]


def selected_arm(tap,eta):
    return ('X-01' if tap=='X' else 'RB1-09') if eta==.25 else f'ETA-{tap}-{eta:g}'


def analyze(spec):
    # Validate the already-written calibration lock before opening any dev file.
    lock_path=Path(spec['eta_lock']);lock_ref=reference(lock_path)
    if lock_ref['sha256']!=spec['eta_lock_sha256']:
        raise ValueError('eta-lock hash differs from registered analysis input')
    lock=read(lock_path)
    rebuilt=selection(spec['calibration_manifests'],final=True)
    if lock!=rebuilt:
        raise ValueError('eta-lock no longer reconstructs from its calibration manifests')
    selected={tap:selected_arm(tap,value['eta']) for tap,value in lock['choices'].items()}
    # The selected candidate manifests are identical to those used by training/
    # compatibility evaluation, preventing substitution after eta selection.
    manifest_paths={record['arm']:record['inputs']['manifest']['path']
                    for candidates in lock['candidates'].values() for record in candidates.values()}
    for arm,root in spec['training_runs'].items():
        if arm in ('X-02','X-03'):
            continue
        if str((Path(root)/'manifest.json').resolve())!=manifest_paths.get(arm):
            raise ValueError('analysis training run is not the locked calibration producer')
    if str((Path(spec['baseline_eval'])/'manifest.json').resolve())!=manifest_paths.get('RB1-09'):
        raise ValueError('RB1-09 compatibility evaluation differs from locked calibration producer')
    base_root=Path(spec['base_run'])
    if reference(base_root/'manifest.json')['sha256']!=ACCEPTED_BASE_MANIFEST_SHA256:
        raise ValueError('Base manifest differs from accepted reference')
    base_rows,base_manifest=_read_run(base_root,'Base','base')
    rows={'Base':validate_rows(base_rows,'Base')}
    performance={'Base':check_metrics(rows['Base'],base_manifest['metrics'],'Base')}
    sources={'eta_lock':lock_ref};training={};initial={};curves={};rms={};runtime={};formal_sources={}
    for arm in ('RB1-07','RB1-09'):
        root=Path(spec['reference_runs'][arm])
        manifest,refs,_=validate_formal_run(root,arm)
        rows[arm]=validate_rows(read(root/'predictions.json'),arm,rows['Base'])
        performance[arm]=check_metrics(rows[arm],manifest['metrics'],arm)
        sources[arm]=refs
    expected=spec['training_source_commit']
    for arm,root_name in spec['training_runs'].items():
        if arm not in E0028_ARMS:
            raise ValueError('unexpected E-0028 training arm')
        root=Path(root_name);manifest=read(root/'manifest.json');m=manifest['metrics']
        init=read(root/'dev-step-0.json')
        initial[arm]=validate_rows(init['predictions'],arm+'/init',rows['Base'])
        if m.get('failed'):
            if arm not in ETA_ARMS or m.get('failure_kind')!='training_nonfinite':
                raise ValueError('engineering failure must retry before analysis')
            continue
        manifest,refs,_=validate_formal_run(root,arm,expected)
        formal_sources[arm]=dict(refs)
        if (m['plan.order_sha256']!=ORDER_SHA256 or m['split_manifest.sha256']!=SPLIT_SHA256 or
            m['train.steps']!=13115 or m['train.examples']!=26230 or m['train.input_tokens']!=2000069 or
            m['train.loss_tokens']!=387653 or m['learning_rate']!=.001):
            raise ValueError('training budget/order/data/LR differs from E-0028')
        rows[arm]=validate_rows(read(root/'predictions.json'),arm,rows['Base'])
        performance[arm]=check_metrics(rows[arm],m,arm)
        curves[arm]={}
        for step in (3279,6558,9837):
            value=read(root/f'dev-step-{step}.json')
            curve=validate_rows(value['predictions'],f'{arm}/{step}',rows['Base'])
            if value.get('step')!=step:
                raise ValueError('intermediate step changed')
            curves[arm][str(step)]=check_metrics(curve,value['metrics'],f'{arm}/{step}')
            refs[f'curve_{step}']=reference(root/f'dev-step-{step}.json')
        curves[arm]['13115']=performance[arm]
        rms[arm]=read(root/'step0-rms.json');refs['step0_rms']=reference(root/'step0-rms.json')
        refs['step0_dev']=reference(root/'dev-step-0.json')
        sources[arm]=refs;training[arm]=manifest
        runtime[arm]={k:v for k,v in m.items() if k.startswith(('train.','plan.','runtime.','params.','init.')) or
                     k in ('learning_rate','hardware','source.commit','environment.lock_sha256')}
    if not set(TAP_ARMS).issubset(training):
        raise ValueError('Part A requires all three completed arms')
    candidates=set(manifest_paths)-{'RB1-09'}
    if set(spec['training_runs'])!=candidates|{'X-02','X-03'}:
        raise ValueError('analysis must include every locked eta candidate and all Part A arms')
    baseline_root=Path(spec['baseline_eval']);baseline=read(baseline_root/'manifest.json')
    if (baseline.get('status')!='complete' or baseline.get('stage')!='checkpoint_evaluation' or
            baseline.get('validator',{}).get('passed') is not True or baseline['metrics'].get('source.commit')!=expected):
        raise ValueError('missing reviewed RB1-09 compatibility evaluation')
    repeated=validate_rows(read(baseline_root/'predictions.json'),'RB1-09 re-eval',rows['Base'])
    registration=read(spec['registration'])
    if registration.get('contract')!='E-0028@1' or registration.get('source_commit')!=expected:
        raise ValueError('initialization registration is from a different source')
    checks={
        'a_initial_predictions_Base':all(same_predictions(value,rows['Base']) for value in initial.values()),
        'c_reload':all(value['metrics']['reload.identical'] for value in training.values()),
        'd_parameters':all(all(value['metrics'][key]==PARAMETERS[arm] for key in
                          ('params.stored','params.trainable','params.with_grad')) for arm,value in training.items()),
        'e_f_h_init_and_tap':registration.get('passed') is True,
        'g_RB109_dev_unchanged':same_predictions(repeated,rows['RB1-09']),
        'i_selection_tests':read(spec['smoke_manifest']).get('metrics',{}).get('pytest_exit')==0,
        'init_digests_match_registration':all(m['metrics'].get('init.parameters.sha256')==registration['arms'][arm]['init_sha256'] for arm,m in training.items())}
    ivroot=Path(spec['intervention_run']);ivmanifest=read(ivroot/'manifest.json')
    provenance=read(ivroot/'provenance.json');conditions=read(ivroot/'conditions.json')
    if (ivmanifest.get('status')!='complete' or ivmanifest.get('validator',{}).get('passed') is not True or
        ivmanifest.get('contract')!='E-0028@1' or ivmanifest.get('arm')!='X-01' or
        provenance.get('source_commit')!=expected or set(conditions)!= {'normal','Base_same_allocation',*MODES} or
        provenance.get('formal_run')!=formal_sources['X-01']):
        raise ValueError('X-01 intervention provenance/grid differs from training')
    for kind,value in conditions.items():
        validated=validate_rows(value['predictions'],'X-01/'+kind,rows['Base'])
        check_metrics(validated,value['metrics'],'X-01/'+kind)
        if kind not in ('normal','Base_same_allocation'):
            rows['X-01/'+kind]=validated
        if kind!='Base_same_allocation':
            validate_read_summary(value.get('read_diagnostics',[]),'X-01',kind,share_metrics=True)
    checks['b_off_equals_Base']=(conditions['iv_off']['predictions']==conditions['Base_same_allocation']['predictions']
                                 and same_predictions(conditions['iv_off']['predictions'],rows['Base']))
    checks['normal_same_as_final']=same_predictions(conditions['normal']['predictions'],rows['X-01'])
    if not all(checks.values()):
        raise ValueError(f'implementation checks failed: {checks}')
    score={arm:accuracy(value) for arm,value in rows.items()}
    swap_drop={'X':accuracy(conditions['normal']['predictions'])-accuracy(conditions['iv_swap']['predictions']),
               'QK':Fraction(195,10000)}
    rules=rule_outcomes(score,selected,swap_drop)
    rules['1_implementation']={'passed':True,'checks':checks}
    rules['5_eta_selection']=lock
    arrays={arm:{task:[int(r['prediction']==r['gold']) for r in value if r['task']==task] for task in TASKS}
            for arm,value in rows.items()}
    # Keep each intervention contrast paired to its own normal path.
    arrays['X-01/normal']={task:[int(r['prediction']==r['gold']) for r in conditions['normal']['predictions'] if r['task']==task] for task in TASKS}
    comparisons=paired_bootstrap(arrays,repeats=10000,seed=20261101)
    sources.update(registration=reference(spec['registration']),baseline_evaluation=reference(baseline_root/'manifest.json'),
                   intervention={name:reference(ivroot/name) for name in ('manifest.json','provenance.json','conditions.json')})
    result={'contract':'E-0028@1','rules':rules,'performance':performance,'eta_lock_sha256':lock_ref['sha256'],
        'selected_arms':selected,'curves':curves,'step0_rms':rms,'runtime':runtime,'sources':sources,
        'bootstrap':{'repeats':10000,'seed':20261101,'comparisons':comparisons,
                     'scope':'descriptive paired within-task resampling; excludes seed and eta selection uncertainty'},
        'measured_cost':measured_cost(spec.get('observations',[])),
        'limitations':['seed0 screening','dev also used for decisions; 1000 examples per task',
            'LR1e-3 inherited, not recalibrated for tap/T0/eta interactions','T0 eta fixed at0.25',
            'two-tap at-least-one support rule entails multiple comparisons','no confirmation or official validation read']}
    finite_tree(result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    value=analyze(read(args.input));write_json(args.output/'analysis.json',value)
    write_json(args.output/'manifest.json',{'status':'complete','contract':'E-0028@1',
        'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':{'implementation_passed':True,'rules':value['rules']},
        'artifacts':[{'path':str(args.output/'analysis.json'),'role':'analysis'}],
        'validator':{'name':'e0028_locked_analysis','passed':True}})
