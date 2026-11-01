"""Deterministic initial-grid report using the reviewed V5 selection functions.

Run in PBS after the whole initial grid ends. This report schedules the required
endpoint runs; only v5_train lock, after those runs, produces the final LR lock.
"""
import argparse
import json
import math
import os
from pathlib import Path

from v5_train import ARMS, CONTRACT, GRID, extension_lr, select_lr, write_json


def summarize(manifests):
    rows, grouped, plans = [], {arm:{} for arm in ARMS}, set()
    for path in manifests:
        m = json.loads(Path(path).read_text())
        if (m['status'] != 'complete' or not m['validator']['passed']
                or m['phase'] != 'calibration' or m['contract'] != CONTRACT or m['seed'] != 0):
            raise ValueError(f'invalid calibration manifest: {path}')
        arm, metrics = m['arm'], m['metrics']
        lr = metrics['learning_rate']
        if arm not in grouped or lr not in GRID or lr in grouped[arm]:
            raise ValueError('report requires each initial-grid candidate exactly once')
        failed = metrics.get('run.failed_lr',False)
        nll = None if failed else metrics['calibration.nll']
        if not failed and (nll is None or not math.isfinite(nll) or nll <= 0):
            raise ValueError('finite positive calibration NLL required')
        grouped[arm][lr] = nll
        plans.add((metrics['plan.order_sha256'], metrics['plan.steps'],metrics['plan.input_tokens']))
        if metrics['plan.input_tokens'] < 200000:
            raise ValueError('calibration plan did not reach200k tokens')
        if not failed and (metrics['train.steps'] != metrics['plan.steps']
                           or metrics['train.input_tokens'] != metrics['plan.input_tokens']):
            raise ValueError('finite candidate did not complete its training plan')
        rows.append({'arm':arm,'lr':lr,'calibration_nll':nll,'failed_lr':failed,
                     'manifest':str(Path(path).resolve()),
                     'train_steps':metrics['train.steps'],'input_tokens':metrics['train.input_tokens'],
                     'steady_step_seconds':metrics.get('train.steady_step_seconds'),
                     'calibration_eval_seconds':metrics.get('eval.seconds'),
                     'elapsed_s':metrics['elapsed_s']})
    if len(plans) != 1:
        raise ValueError('calibration candidates used different sample orders/budgets')
    extensions = {}
    initial_choices = {}
    for arm,candidates in grouped.items():
        extensions[arm] = extension_lr(candidates)
        initial_choices[arm] = select_lr(candidates)
    order,steps,tokens = next(iter(plans))
    return {'contract':CONTRACT,'required_extensions':extensions,
            'provisional_initial_choices':initial_choices,'rows':rows,
            'paired_plan':{'order_sha256':order,'steps':steps,'input_tokens':tokens},
            'note':'Initial grid only. Run required extensions, then v5_train lock; no final LR is locked here.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    passed = False
    try:
        report = summarize(args.manifests)
        write_json(out/'initial-grid.json',report)
        passed = True
    finally:
        write_json(out/'manifest.json',{'status':'complete' if passed else 'partial',
                   'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':{'grid.complete':passed},
                   'artifacts':[{'path':str(out/'initial-grid.json'),'role':'initial_lr_grid'}],
                   'validator':{'name':'v5_initial_grid_report','passed':passed}})


if __name__ == '__main__':
    main()
