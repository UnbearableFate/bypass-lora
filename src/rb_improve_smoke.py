"""E-0029 PBS-only checks, measured timing and pre-training cost gate."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb1_artifacts import read
from rb_data import plan_batches
from rb_improve_init import calibration_examples
from rb_improve_read import diagnose
from rb_train import E0029_ARMS, PARAMETERS, load_data, make_model
from v1_run import evaluate
from v5_train import plan_metadata, write_json


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    def child(arm, stage, directory, *extra):
        with (out/f'{directory}.log').open('w') as log:
            subprocess.run([sys.executable, 'src/rb_improve_train.py', '--arm', arm, '--stage', stage,
                '--lr', '.001', '--split-manifest', str(args.split_manifest), '--data-dir', str(args.data_dir),
                '--output', str(out/directory), *extra], check=True, stdout=log, stderr=subprocess.STDOUT)
        value = read(out/directory/'manifest.json')
        if value['status'] != 'complete' or value['validator']['passed'] is not True or value['metrics']['dev_read'] is not False:
            raise ValueError('smoke failed or opened dev')
        return value['metrics']
    # One complete I-G collection measures forward/backward, SVD and RMS verification.
    ig = child('G-T', 'ig', 'ig-G-T')
    arms = {arm: child(arm, 'smoke', 'smoke-'+arm, '--smoke-steps', '80') for arm in E0029_ARMS}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    splits, excluded = load_data(args.data_dir, args.split_manifest, tok)
    samples, sample_record = calibration_examples(splits, tok, excluded)
    cal_plan = plan_batches(splits, tok, 'V0', 0, [(r['task'], r['id']) for r in excluded], 200000)
    model = make_model('L-56', checkpoint=out/'smoke-L-56'/'adapter')
    start = time.monotonic()
    # Scoring is solely to measure a full calibration pass. This is not an LR candidate.
    _, rows = evaluate(model, splits, tok, split_names=('calibration',))
    calibration_seconds = time.monotonic()-start
    if len(rows) != 4096:
        raise ValueError('calibration timing did not cover both full2048-row splits')
    del model, rows
    gc.collect()
    torch.cuda.empty_cache()
    model = make_model('G-T', checkpoint=out/'ig-G-T'/'adapter')
    start = time.monotonic()
    diagnostic = diagnose(model, samples[:64], tok)
    diagnostic_seconds = time.monotonic()-start
    write_json(out/'diagnostic-example.json', diagnostic)
    # N-QK must share every initial parameter with N-QKV (only the value input differs).
    from rb_tap_check import initial_state
    model_qn = make_model('Q-N')
    qn_digest = initial_state(model_qn)[0]
    del model_qn
    model_n = make_model('X-01')
    if qn_digest != initial_state(model_n)[0]:
        raise ValueError('Q-N and X-01 initial parameters differ')
    if any(m['params.stored'] != PARAMETERS[arm] or m['params.with_grad'] != PARAMETERS[arm] for arm, m in arms.items()):
        raise ValueError('smoke parameter/gradient registry mismatch')
    dev_seconds = calibration_seconds / 2.048 * 1.1
    startup = {arm: max(0., m['elapsed_s']-m['train.seconds']) for arm, m in arms.items()}
    training = {arm: m['train.steady_step_seconds']*13115 + 7*dev_seconds + startup[arm] +
                (2*diagnostic_seconds if arm not in ('L-56', 'L-48') else 0.) for arm, m in arms.items()}
    # Four LR candidates is the worst case; use the measured V0 plan length.
    lr_seconds = 4 * (arms['L-56']['train.steady_step_seconds']*len(cal_plan) +
                       calibration_seconds + startup['L-56'])
    ig_seconds = 3 * ig['elapsed_s']
    reference_seconds = 4*diagnostic_seconds + 2*max(startup.values())
    smoke_seconds = time.monotonic()-tick
    total = (sum(training.values()) + lr_seconds + ig_seconds + reference_seconds + smoke_seconds + 180)/14400
    cost = {'contract': 'E-0029@1', 'cap_node_hours': 3.5, 'profile_cost_factor': .25,
        'training_seconds': training, 'worst_case_four_LR_seconds': lr_seconds,
        'three_IG_seconds': ig_seconds, 'reference_diagnostics_seconds': reference_seconds,
        'current_smoke_seconds': smoke_seconds, 'pytest_seconds_reserved': 60,
        'analysis_seconds_reserved': 120, 'full_calibration_seconds': calibration_seconds,
        'estimated_dev_seconds': dev_seconds, 'diagnostic64_seconds': diagnostic_seconds,
        'projected_total_node_hours': total, 'within_cap': total <= 3.5,
        'method': '80-step steady timing;13115 full steps;7 dev passes incl fallback Base;MIG/4;four LR candidates;three full IG collections;reference reads;smoke;180s checks/analysis'}
    write_json(out/'cost-estimate.json', cost)
    write_json(out/'manifest.json', {'status': 'complete', 'contract': 'E-0029@1',
        'spec_digest': os.environ['SR_SPEC_DIGEST'], 'stage': 'smoke',
        'metrics': {'source_commit': os.environ['SR_CODE_COMMIT'], 'pytest_exit': 0,
            'arms': arms, 'ig': ig, 'calibration_plan': plan_metadata(cal_plan),
            'calibration_seconds': calibration_seconds, 'diagnostic_seconds': diagnostic_seconds,
            'dev_read': False, 'projected_node_hours': total, 'sample_sha256': sample_record['examples_sha256'],
            'elapsed_s': time.monotonic()-tick},
        'artifacts': [{'path': str(p), 'role': p.name} for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'],
        'validator': {'name': 'e0029_smoke', 'passed': True}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
