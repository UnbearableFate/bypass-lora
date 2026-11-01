"""Validate calibration artifacts and apply E-0009's fixed NLL selection rule."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from v2_schedule import select_learning_rate


def report(paths, final=False):
    rows = []
    for path in paths:
        path = Path(path)
        manifest = json.loads((path/'manifest.json').read_text())
        if manifest['status'] != 'complete' or not manifest['validator']['passed']:
            raise ValueError(f'incomplete calibration: {path}')
        payload = (path/'metrics.json').read_bytes()
        arms = json.loads(payload)
        if len(arms) != 1:
            raise ValueError('calibration requires one representative arm per attempt')
        arm, r = next(iter(arms.items()))
        cfg = r['config']
        if cfg['tap'] != 'qknorm_pre_rope' or cfg['fusion'] != 'F_out':
            raise ValueError('wrong calibration representative')
        if (r['schedule']['steps'], r['schedule']['tokens'], r['schedule']['warmup_steps']) != (300, 200170, 15):
            raise ValueError('wrong calibration schedule')
        if r['schedule']['order_sha256'] != '0d4b55ed8655d91cf0d0532bdac1820a518b5c5730ef70e48f264da6b4bc9e79':
            raise ValueError('wrong sample order')
        history = [json.loads(s) for s in (path/arm/'train.jsonl').read_text().splitlines()]
        finite_loss = len(history) == 300 and all(math.isfinite(s['loss']) for s in history)
        if not r['checkpoint_reload']['predictions_identical']:
            raise ValueError('checkpoint reload failed')
        cal = r['evaluation']['calibration']
        if any(cal[m]['count'] != 2048 for m in ('normal', 'zero', 'shuffled')):
            raise ValueError('calibration split incomplete')
        rows.append(dict(attempt=path.name, backend=cfg['backend'], lr=r['lr'],
            nll=cal['normal']['nll'], finite_loss=finite_loss,
            controls={m: {k: cal[m][k] for k in ('nll','em','count')} for m in ('normal','zero','shuffled')},
            gradient_norm_mean=r['train']['gradient_norm_mean'],gradient_norm_max=r['train']['gradient_norm_max'],
            clip_fraction=r['train']['clip_fraction'],training_seconds=r['train']['training_seconds'],
            tokens_per_second=r['train']['tokens_per_second'],peak_allocated_bytes=r['train']['peak_allocated_bytes'],
            wall_seconds=r['wall_seconds'],metrics_sha256=hashlib.sha256(payload).hexdigest()))
    selections = {}
    for backend in ('kernel_matrix','soft_slot'):
        candidates = [r for r in rows if r['backend'] == backend]
        if len({r['lr'] for r in candidates}) != len(candidates):
            raise ValueError('duplicate learning-rate candidate')
        if not {.0003,.001,.003} <= {r['lr'] for r in candidates}:
            raise ValueError('initial candidate grid incomplete')
        chosen = select_learning_rate(candidates)
        extension = None
        if not final:
            if chosen['lr'] == .0003:
                extension = .0001
            elif chosen['lr'] == .003:
                extension = .01
        selections[backend] = dict(selected_lr=chosen['lr'],normal_nll=chosen['nll'],attempt=chosen['attempt'],extension_lr=extension)
    return dict(candidates=rows,selections=selections,final=final,
                selection_metric='calibration normal full-vocabulary answer NLL only; strict <0.5% relative-to-min tie prefers lower LR',
                historical_diagnostic_limit='f7c4d2b reconstruction_max_abs is weighted mean of batch maxima; unused here')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--final',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    result=report(a.runs,a.final)
    target=a.output/'calibration-report.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output/'manifest.json').write_text(json.dumps(dict(status='complete',spec_digest=os.environ['SR_SPEC_DIGEST'],
        metrics={b+'.'+k:v for b,s in result['selections'].items() for k,v in s.items() if isinstance(v,(int,float))},
        artifacts=[dict(path=str(target.resolve()),role='calibration_selection')],
        validator=dict(name='sealed_schedule_and_calibration_selection',passed=True)),indent=2)+'\n')
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
