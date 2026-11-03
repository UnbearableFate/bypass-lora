"""E-0031 locked screening rules and paired split/task-stratified intervals (PBS)."""
import argparse
from fractions import Fraction
from itertools import combinations
import os
from pathlib import Path

import numpy as np

from rb1_analysis import ACCEPTED_BASE_MANIFEST_SHA256, check_metrics, summarize_rows, validate_rows
from rb1_artifacts import finite_tree, read, reference, validate_formal_run
from rb_eta_select import ORDER_SHA256, SPLIT_SHA256, validate_calibration
from rb_qvu_eval import LEGACY
from rb_tap_analysis import accuracy, pp
from rb_tap_identity import prediction_keys
from rb_train import E0031_ARMS, PARAMETERS
from v5_train import write_json

ONE_PP = Fraction(1, 100)
TASKS = ('piqa', 'boolq')
X_FILES = {
    'X-01': {'manifest.json':'11348b031d25ba6a2210573f75bef1e84fb01c198a8530cb470699725d5e0915',
        'predictions.json':'452a04443f210834bdbf53c62eb56e435e3897bf656de036e977614462e5fa78',
        'calibration-predictions.json':'51243a0c0191f0deb6630c1354fd4c7c4bcefcbba407c96347608390eaa6182e'},
    'X-02': {'manifest.json':'05fcfd25cf4c3076ac48c93ef6a14cbbf9d8a9f89c11217b223f5b207833f696',
        'predictions.json':'e5d46ccc6bfe5023d139e583dd3ebf92702fe1aade0b2276cd8b77ce65a12bbb'}}


def practical(score, scores):
    return score-scores['X-01'] >= ONE_PP and score-scores['RB1-07'] >= 0


def rules(scores, dev_scores, swap_drop):
    dx = scores['H-QVU']-scores['X-01']
    dl = scores['H-QVU']-scores['RB1-07']
    dh = scores['H-QVU']-scores['T0-QVU']
    h30 = 'supports' if dx >= ONE_PP and dl >= 0 else 'challenges' if dx <= 0 else 'inconclusive'
    h31 = 'supports' if dh >= ONE_PP and swap_drop >= ONE_PP else 'challenges' if dh <= 0 else 'inconclusive'
    conditional = any(practical(scores[a], scores) for a in ('H-QVU','T0-QVU'))
    result = {
        '2_learnability':{a:{'minus_Base':pp(s-dev_scores['Base']),
            'at_least_Base_plus_1pp':s-dev_scores['Base'] >= ONE_PP}
            for a,s in dev_scores.items() if a in E0031_ARMS},
        '3_H0030':{'d_X':pp(dx), 'd_L':pp(dl), 'outcome':h30, 'evidence_level':'screening',
            'dev_d_X':pp(dev_scores['H-QVU']-dev_scores['X-01']),
            'dev_d_L':pp(dev_scores['H-QVU']-dev_scores['RB1-07'])},
        '4_H0031':{'d_h':pp(dh), 'dev_swap_drop':pp(swap_drop), 'outcome':h31, 'evidence_level':'screening'},
        '5_T0_practical':{'d_X':pp(scores['T0-QVU']-scores['X-01']),
            'd_L':pp(scores['T0-QVU']-scores['RB1-07']), 'passes':practical(scores['T0-QVU'], scores)},
        '6_conditional_controls':{'triggered':conditional, 'complete':not conditional or all(a in scores for a in ('U-25','T0-U-25'))},
        '7_replication':{'propose_two_new_seeds':h30 == h31 == 'supports',
            'coordinator_new_decision_required':h30 == 'supports' or practical(scores['T0-QVU'],scores)},
        '9_scope':{'bootstrap_repeats':10000,'bootstrap_seed':20261101,
            'strata':'dev and calibration separately, within each task; common indices for all arms',
            'level':'screening; seed0;LR1e-3;row sampling only;calibration used historically for LR/eta selection;A2 and official validation sealed'},
        '10_route':{'coordinator_stop_review_required':h30 != 'supports' and h31 != 'supports' and not practical(scores['T0-QVU'],scores)}}
    if 'U-25' in scores and 'T0-U-25' in scores:
        result['6_conditional_controls'].update({'H_minus_U25':pp(scores['H-QVU']-scores['U-25']),
            'T0_minus_T0U25':pp(scores['T0-QVU']-scores['T0-U-25'])})
    return result


def paired_intervals(rows, repeats=10000, seed=20261101):
    """Common resampling indices across all arms; separate split/task strata."""
    labels = sorted(rows)
    identities = lambda rs:[(r['split'],r['task'],r['id'],r['gold'],r['gold_loss_tokens']) for r in rs]
    expected = identities(rows[labels[0]])
    if any(identities(rows[a]) != expected for a in labels):
        raise ValueError('bootstrap requires identical ordered paired rows')
    strata = {}
    totals = {task:0 for task in TASKS}
    for split in ('dev','calibration'):
        for task in TASKS:
            indices = [i for i,r in enumerate(rows[labels[0]]) if r['split'] == split and r['task'] == task]
            if not indices:
                continue
            strata[(split,task)] = np.asarray([[int(rows[a][i]['prediction'] == rows[a][i]['gold']) for i in indices]
                                              for a in labels], dtype=np.float64)
            totals[task] += len(indices)
    if any(n == 0 for n in totals.values()):
        raise ValueError('both bootstrap tasks must be nonempty')
    rng = np.random.default_rng(seed)
    boot = np.zeros((len(labels),repeats), dtype=np.float64)
    for start in range(0,repeats,100):
        count = min(100,repeats-start)
        for (_,task),values in strata.items():
            idx = rng.integers(0,values.shape[1],size=(count,values.shape[1]))
            for a in range(len(labels)):
                boot[a,start:start+count] += values[a][idx].sum(axis=1)/totals[task]/2
    return {f'{left}_minus_{right}':{'estimate_pp':float(100*(accuracy(rows[left])-accuracy(rows[right]))),
        'ci95_pp':np.quantile(100*(boot[i]-boot[j]), [.025,.975]).tolist()}
        for (i,left),(j,right) in combinations(enumerate(labels),2)}


def complete(path):
    value = read(path)
    if value.get('status') != 'complete' or value.get('validator',{}).get('passed') is not True:
        raise ValueError('requires a complete passing artifact: '+str(path))
    return value


def interval(comparisons, left, right):
    if left < right:
        return comparisons[f'{left}_minus_{right}']
    value = comparisons[f'{right}_minus_{left}']
    return {'estimate_pp':-value['estimate_pp'], 'ci95_pp':[-value['ci95_pp'][1],-value['ci95_pp'][0]]}


def analyze(spec):
    sources, dev, cal, runtime, curves = {}, {}, {}, {}, {}
    base = Path(spec['base_run'])
    if reference(base/'manifest.json')['sha256'] != ACCEPTED_BASE_MANIFEST_SHA256:
        raise ValueError('Base reference changed')
    base_manifest = complete(base/'manifest.json')
    dev['Base'] = validate_rows(read(base/'predictions.json'),'Base')
    check_metrics(dev['Base'],base_manifest['metrics'],'Base')
    sources['Base'] = {name:reference(base/name) for name in ('manifest.json','predictions.json')}
    for arm in ('X-01','X-02'):
        root = Path(spec['references'][arm])
        sources[arm] = {name:reference(root/name) for name in X_FILES[arm]}
        if any(sources[arm][name]['sha256'] != digest for name,digest in X_FILES[arm].items()):
            raise ValueError('accepted E-0028 reference bytes changed')
        manifest = complete(root/'manifest.json')
        dev[arm] = validate_rows(read(root/'predictions.json'),arm,dev['Base'])
        check_metrics(dev[arm],manifest['metrics'],arm)
        if arm == 'X-01':
            cal[arm] = read(root/'calibration-predictions.json')
            validate_calibration(cal[arm],manifest['metrics'])
    for arm,attempt in LEGACY.items():
        root = Path(spec['references'][arm])
        if root.name != attempt:
            raise ValueError('legacy training reference differs from contract')
        manifest, refs, _ = validate_formal_run(root,arm)
        dev[arm] = validate_rows(read(root/'predictions.json'),arm,dev['Base'])
        check_metrics(dev[arm],manifest['metrics'],arm)
        rerun = Path(spec['reevaluations'][arm])
        m = complete(rerun/'manifest.json')
        if (m.get('arm') != arm or m.get('stage') != 'checkpoint_evaluation' or
                m['metrics'].get('source.commit') != spec['source_commit'] or
                m['metrics'].get('split_manifest.sha256') != SPLIT_SHA256):
            raise ValueError('legacy reevaluation differs from reviewed source/split/arm')
        provenance = read(rerun/'provenance.json')
        if provenance['training_run'] != refs:
            raise ValueError('legacy reevaluation checkpoint provenance differs')
        reevaluated = validate_rows(read(rerun/'predictions.json'),arm+'/reevaluation',dev[arm])
        flips = sum(a['prediction'] != b['prediction'] for a,b in zip(reevaluated,dev[arm]))
        if flips != m['metrics']['compatibility']['flips'] or (m['metrics']['compatibility']['same_profile'] and flips):
            raise ValueError('legacy compatibility check failed')
        cal[arm] = read(rerun/'calibration-predictions.json')
        validate_calibration(cal[arm],m['metrics'])
        sources[arm] = {'training':refs, 'reevaluation':{name:reference(rerun/name) for name in
            ('manifest.json','predictions.json','calibration-predictions.json','provenance.json')}}
    smoke_root = Path(spec['smoke_run'])
    smoke = complete(smoke_root/'manifest.json')
    if smoke['metrics'].get('pytest_exit') != 0 or smoke['metrics'].get('source_commit') != spec['source_commit']:
        raise ValueError('smoke did not test this source snapshot')
    registration = read(smoke_root/'registration.json')
    sources['smoke'] = {name:reference(smoke_root/name) for name in ('manifest.json','registration.json','cost-estimate.json')}
    initial_checks = {}
    for arm,run in spec['training_runs'].items():
        if arm not in E0031_ARMS:
            raise ValueError('unexpected E-0031 arm')
        root = Path(run)
        manifest, refs, _ = validate_formal_run(root,arm,spec['source_commit'])
        m = manifest['metrics']
        if (m['plan.order_sha256'] != ORDER_SHA256 or m['train.steps'] != 13115 or
                m['train.examples'] != 26230 or m['train.input_tokens'] != 2000069 or
                m['train.loss_tokens'] != 387653 or m['learning_rate'] != .001 or
                m['split_manifest.sha256'] != SPLIT_SHA256):
            raise ValueError('E-0031 training budget/order/split/LR differs')
        if m['init.parameters.sha256'] != registration[arm]['initial_sha256']:
            raise ValueError('formal initialization differs from checked initialization')
        initial = validate_rows(read(root/'dev-step-0.json')['predictions'],arm+'/init',dev['Base'])
        local = read(root/'dev-base-same-allocation.json')
        local_rows = validate_rows(local['predictions'],arm+'/localBase',dev['Base'])
        if (local['job_id'] != m['exec.pbs_job_id'] or local['hardware'] != m['hardware'] or
                local['source_commit'] != spec['source_commit'] or prediction_keys(initial) != prediction_keys(local_rows)):
            raise ValueError('initial predictions differ from same-allocation Base')
        off = read(root/'iv-off.json')
        if prediction_keys(validate_rows(off['predictions'],arm+'/off',dev['Base'])) != prediction_keys(local_rows):
            raise ValueError('iv_off predictions differ from same-allocation Base')
        dev[arm] = validate_rows(read(root/'predictions.json'),arm,dev['Base'])
        check_metrics(dev[arm],m,arm)
        cal[arm] = read(root/'calibration-predictions.json')
        validate_calibration(cal[arm],m)
        initial_checks[arm] = {'same_allocation_Base':True, 'iv_off_Base':True,
            'reload_identical':m['reload.identical'], 'registration':registration[arm]}
        curves[arm] = {}
        for step in (3279,6558,9837):
            value = read(root/f'dev-step-{step}.json')
            if value['step'] != step:
                raise ValueError('dev curve step differs')
            curve = validate_rows(value['predictions'],f'{arm}/{step}',dev['Base'])
            curves[arm][str(step)] = check_metrics(curve,value['metrics'],arm)
            refs[f'curve_{step}'] = reference(root/f'dev-step-{step}.json')
        curves[arm]['13115'] = summarize_rows(dev[arm])
        refs.update({name:reference(root/name) for name in ('dev-step-0.json', 'dev-base-same-allocation.json',
            'iv-off.json', 'predictions.json', 'calibration-predictions.json')})
        sources[arm] = refs
        runtime[arm] = {k:v for k,v in m.items() if k.startswith(('params.','train.','runtime.','plan.'))}
    if not all(a in dev for a in ('H-QVU','T0-QVU')):
        raise ValueError('both new QVU arms are required')
    if runtime['H-QVU'] and complete(Path(spec['training_runs']['H-QVU'])/'manifest.json')['metrics']['hardware'] != complete(
            Path(spec['training_runs']['T0-QVU'])/'manifest.json')['metrics']['hardware']:
        raise ValueError('H/T0 MIG specifications differ')
    identity = lambda rows:[(r['task'],r['id'],r['gold'],r['gold_loss_tokens']) for r in rows]
    for arm,rows in cal.items():
        if identity(rows) != identity(cal['X-01']):
            raise ValueError('calibration membership/order/gold/token counts differ')
        if {(r['task'],r['id']) for r in rows} & {(r['task'],r['id']) for r in dev[arm]}:
            raise ValueError('dev and calibration overlap')
    merged = {arm:dev[arm]+rows for arm,rows in cal.items()}
    intervention = Path(spec['intervention_run'])
    iv = complete(intervention/'manifest.json')
    if iv.get('arm') != 'H-QVU' or iv.get('stage') != 'intervention' or iv['metrics'].get('source.commit') != spec['source_commit']:
        raise ValueError('intervention producer differs')
    if read(intervention/'provenance.json')['training_run'] != {k:v for k,v in sources['H-QVU'].items() if k in
            ('manifest','run_config','checkpoint_json','steps','lr_lock','split_manifest','adapter_config','adapter_weights')}:
        raise ValueError('intervention checkpoint differs from H-QVU')
    conditions = read(intervention/'conditions.json')
    normal = validate_rows(conditions['normal']['predictions'],'normal',dev['Base'])
    swapped = validate_rows(conditions['iv_swap']['predictions'],'swap',dev['Base'])
    if prediction_keys(normal) != prediction_keys(dev['H-QVU']):
        raise ValueError('normal intervention dev differs from formal training')
    for name,rows in (('normal',normal),('iv_swap',swapped)):
        check_metrics(rows,conditions[name]['metrics'],name)
    sources['intervention'] = {name:reference(intervention/name) for name in ('manifest.json','conditions.json','provenance.json')}
    scores, dev_scores = {a:accuracy(r) for a,r in merged.items()}, {a:accuracy(r) for a,r in dev.items()}
    result = rules(scores,dev_scores,accuracy(normal)-accuracy(swapped))
    if not result['6_conditional_controls']['triggered'] and any(a in merged for a in ('U-25','T0-U-25')):
        raise ValueError('untriggered conditional controls were trained')
    result['1_implementation'] = {'passed':True,'arms':initial_checks,
        'checkpoint_compatibility':{a:complete(Path(spec['reevaluations'][a])/'manifest.json')['metrics']['compatibility'] for a in LEGACY}}
    intervals = paired_intervals(merged)
    result['3_H0030']['interval_d_X'] = interval(intervals,'H-QVU','X-01')
    result['3_H0030']['interval_d_L'] = interval(intervals,'H-QVU','RB1-07')
    result['4_H0031']['interval_d_h'] = interval(intervals,'H-QVU','T0-QVU')
    result['8_resolution'] = {f'{a}_minus_{b}':interval(intervals,a,b) for a,b in
        (('RB1-02','RB1-01'),('RB1-07','RB1-09'),('RB1-07','X-01'))}
    return {'contract':'E-0031@1', 'source_commit':spec['source_commit'], 'rules':result,
        'merged_performance':{a:summarize_rows(r) for a,r in merged.items()},
        'dev_performance':{a:summarize_rows(r) for a,r in dev.items()}, 'curves':curves, 'runtime':runtime,
        'merged_intervals':intervals, 'dev_intervals':paired_intervals({**dev,'H-QVU/iv_swap':swapped}),
        'sources':sources, 'complete':result['6_conditional_controls']['complete']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    value = analyze(read(args.input))
    finite_tree(value)
    write_json(args.output/'analysis.json',value)
    write_json(args.output/'manifest.json',{'status':'complete','contract':'E-0031@1',
        'spec_digest':os.environ['SR_SPEC_DIGEST'],'stage':'analysis',
        'metrics':{'rules':value['rules'],'conditional_controls_complete':value['complete']},
        'artifacts':[{'path':str((args.output/'analysis.json').resolve()),'role':'locked_analysis'},
                     {'path':str(args.input.resolve()),'role':'analysis_input'}],
        'validator':{'name':'e0031_locked_analysis','passed':True}})
