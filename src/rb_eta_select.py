"""E-0028 rule 5: calibration-only eta selection and immutable input inventory.

This module never opens a dev prediction file or examines a dev metric. Run in
PBS. `plan` identifies the one permitted endpoint extension; `lock` requires it
and writes the final lock exclusively (an existing lock cannot be overwritten).
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

GRID=(.0625,.125,.25,.5,1.,2.)
TASKS=('piqa','boolq')
SPLIT_SHA256='34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc'
ORDER_SHA256='33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf'


def ref(path):
    path=Path(path).resolve()
    data=path.read_bytes()
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def best_eta(candidates):
    finite={eta:score for eta,score in candidates.items() if score is not None}
    if not finite:
        raise ValueError('no finite calibration candidate')
    if any(not isinstance(score,Fraction) or not 0<=score<=1 for score in finite.values()):
        raise ValueError('scores must be exact finite accuracy fractions')
    return min(finite,key=lambda eta:(-finite[eta],abs(math.log2(eta/.25)),eta))


def extension_eta(candidates):
    if set(candidates)!=set(GRID):
        raise ValueError('extension requires exactly the initial six-point grid')
    best=best_eta(candidates)
    return .03125 if best==GRID[0] else 4. if best==GRID[-1] else None


def choose_eta(candidates):
    initial={eta:score for eta,score in candidates.items() if eta in GRID}
    extension=extension_eta(initial)
    expected=set(GRID)|({extension} if extension is not None else set())
    if set(candidates)!=expected:
        raise ValueError('missing or unexpected endpoint extension')
    selected=best_eta(candidates)
    return {'eta':selected,'extension':extension,'grid_boundary':selected==extension if extension else False}


def validate_calibration(rows,metrics):
    if not isinstance(rows,list) or len(rows)!=4096:
        raise ValueError('calibration requires 2048 examples per task')
    seen=set();counts={};nll={}
    for task in TASKS:
        subset=[r for r in rows if r.get('task')==task]
        if len(subset)!=2048:
            raise ValueError('calibration requires 2048 examples per task')
        correct,total,tokens=0,0.,0
        for row in subset:
            key=(task,row.get('id'))
            if type(row.get('id')) is not int or key in seen or row.get('split')!='calibration':
                raise ValueError('invalid/duplicate calibration identity')
            seen.add(key)
            if any(type(row.get(k)) is not int or row[k] not in (0,1) for k in ('gold','prediction')):
                raise ValueError('invalid calibration gold/prediction')
            scores=row.get('scores')
            loss=row.get('gold_nll_sum');nt=row.get('gold_loss_tokens')
            if (not isinstance(scores,list) or len(scores)!=2 or
                    any(type(x) not in (int,float) or not math.isfinite(x) for x in scores) or
                    type(loss) not in (int,float) or not math.isfinite(loss) or loss<0 or type(nt) is not int or nt<=0):
                raise ValueError('non-finite or invalid calibration likelihood')
            if row['prediction']!=(0 if scores[0]>=scores[1] else 1):
                raise ValueError('calibration prediction violates V1 tie rule')
            if not math.isclose(scores[row['gold']],-loss/(nt if task=='piqa' else 1),rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('calibration score differs from answer NLL')
            correct+=row['prediction']==row['gold'];total+=loss;tokens+=nt
        counts[task]=[correct,2048];nll[task]=total/tokens
        expected={'accuracy':correct/2048,'nll':nll[task],'examples':2048,'loss_tokens':tokens}
        for key,value in expected.items():
            actual=metrics.get(f'calibration.{task}.{key}')
            if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,value,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('calibration manifest differs from predictions')
    if not math.isclose(metrics.get('calibration.nll',math.nan),sum(nll.values())/2,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('calibration macro NLL mismatch')
    return sum((Fraction(*counts[t]) for t in TASKS),Fraction())/2,counts,nll


def load_candidate(path):
    path=Path(path).resolve();value=json.loads(path.read_text());m=value.get('metrics',{})
    arm=value.get('arm')
    if arm=='RB1-09':tap,eta='QK',.25
    elif arm=='X-01':tap,eta='X',.25
    elif isinstance(arm,str) and arm.startswith(('ETA-QK-','ETA-X-')):
        _,tap,raw=arm.split('-');eta=float(raw)
    else:
        raise ValueError('unexpected eta-selection arm')
    if eta not in (*GRID,.03125,4.):
        raise ValueError('unregistered eta')
    if value.get('status')!='complete' or value.get('validator',{}).get('passed') is not True or value.get('seed')!=0 or value.get('phase')!='R1':
        raise ValueError('engineering/incomplete evaluation must rerun, not be excluded')
    if value.get('contract')!='E-0028@1' or value.get('stage') not in ('train','checkpoint_evaluation'):
        raise ValueError('eta selection requires E-0028 final-checkpoint evaluation')
    if m.get('split_manifest.sha256')!=SPLIT_SHA256 or m.get('plan.order_sha256')!=ORDER_SHA256 or m.get('learning_rate')!=.001:
        raise ValueError('candidate split/order/LR differs from E-0028')
    refs={'manifest':ref(path)}
    if m.get('failed') is True:
        if m.get('failure_kind')!='training_nonfinite' or not arm.startswith('ETA-'):
            raise ValueError('only non-finite TRAINING grid candidates may be excluded')
        return tap,eta,None,{'arm':arm,'excluded':'training_nonfinite','inputs':refs}
    if m.get('failed') is not False or m.get('train.steps')!=13115 or m.get('train.examples')!=26230 or m.get('reload.identical') is not True:
        raise ValueError('candidate did not complete the locked training/reload budget')
    predictions=path.parent/'calibration-predictions.json'
    rows=json.loads(predictions.read_text())
    score,counts,nll=validate_calibration(rows,m)
    refs['calibration_predictions']=ref(predictions)
    return tap,eta,score,{'arm':arm,'score':{'numerator':score.numerator,'denominator':score.denominator},
        'counts':counts,'nll':nll,'inputs':refs,
        'identities':[(r['task'],r['id'],r['gold'],r['gold_loss_tokens']) for r in rows]}


def selection(manifests,final=True):
    groups={tap:{} for tap in ('QK','X')};records={tap:{} for tap in groups}
    paired=None
    for path in manifests:
        tap,eta,score,record=load_candidate(path)
        if eta in groups[tap]:
            raise ValueError('duplicate tap/eta candidate')
        identities=record.pop('identities',None)
        if identities is not None:
            if paired is not None and identities!=paired:
                raise ValueError('calibration membership/order/gold differs between candidates')
            paired=identities
        groups[tap][eta]=score;records[tap][f'{eta:g}']=record
    choices={tap:choose_eta(scores) if final else {'extension':extension_eta(scores)} for tap,scores in groups.items()}
    return {'contract':'E-0028@1','selection_split':'calibration','selection_metric':'macro_accuracy',
            'tie_rule':'max accuracy, min abs(log2(eta/0.25)), min eta',
            'split_manifest_sha256':SPLIT_SHA256,'choices':choices,'candidates':records}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','lock'))
    parser.add_argument('--manifests',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=selection(args.manifests,final=args.command=='lock')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
