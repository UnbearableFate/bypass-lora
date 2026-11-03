"""E-0031 PBS-only real counts, paired initialization, timing and budget forecast."""
import argparse
import gc
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from rb1_artifacts import read
from rb_qvu_eval import LEGACY
from rb_tap_check import initial_state
from rb_train import E0031_ARMS, PARAMETERS, make_model
from v5_train import write_json


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    passed, metrics = False, {}
    def child(script, name, *extra):
        target = out/name
        with (out/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, 'src/'+script, '--output', str(target),
                '--split-manifest', str(args.split_manifest), '--data-dir', str(args.data_dir), *extra],
                check=True, stdout=log, stderr=subprocess.STDOUT)
        manifest = read(target/'manifest.json')
        if manifest['status'] != 'complete' or manifest['validator']['passed'] is not True:
            raise ValueError('smoke child incomplete')
        return manifest['metrics']
    try:
        if not math.isfinite(args.prior_node_hours) or args.prior_node_hours < 0:
            raise ValueError('prior measured node-hours must be finite and nonnegative')
        with (out/'pytest.txt').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                'src/test_rb_qvu.py', 'src/test_rb_qvu_analysis.py', 'src/test_mig_profile_guard.py', 'src/test_rb_tap.py',
                'src/test_rb_train.py', 'src/test_rb1_artifacts.py', 'src/test_rb1_intervene.py'],
                check=True, stdout=log, stderr=subprocess.STDOUT)
        metrics['pytest_exit'] = 0
        anchors, registration = {}, {}
        ids = torch.tensor([[1, 5, 9, 13, 18]], device='cuda')
        for arm in E0031_ARMS:
            model = make_model(arm)
            digest, state = initial_state(model)
            pair = 'qvu' if arm.endswith('QVU') else 'up'
            anchor = anchors.setdefault(pair, state)
            same = state.keys() == anchor.keys() and all(torch.equal(v,anchor[k]) for k,v in state.items())
            if not same:
                raise ValueError('real initialized tensors are not paired')
            versions = {n:p._version for n,p in model.named_parameters() if not p.requires_grad}
            model.train()
            logits = model(ids).logits.float()
            torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).backward()
            grad_count = sum(p.numel() for p in model.adapter_parameters() if p.grad is not None)
            exits = {f'{i}.{n}': float(p.grad.norm()) for i,l in model.adapters().items()
                     for n,p in l.side.fusion.named_parameters()}
            if (grad_count != PARAMETERS[arm] or not all(v > 0 for v in exits.values()) or
                    any(p.grad is not None or p._version != versions[n] for n,p in model.named_parameters() if not p.requires_grad)):
                raise ValueError('real gradient/frozen-parameter check failed')
            registration[arm] = {'initial_sha256':digest, 'paired_elementwise':same,
                'stored':sum(p.numel() for p in model.adapter_parameters()),
                'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'with_grad':grad_count, 'exit_gradient_norms':exits, 'backbone_frozen':True}
            del model, logits, state
            gc.collect(); torch.cuda.empty_cache()
        write_json(out/'registration.json', registration)
        arms = {arm: child('rb_qvu_train.py', 'smoke-'+arm, '--arm', arm, '--stage', 'smoke',
                    '--lr-lock', str(args.lr_lock), '--smoke-steps', '80', '--eval-limit', '16')
                for arm in E0031_ARMS}
        # Full dev/calibration timing and compatibility for one accepted checkpoint;
        # the other three get bounded load/scoring checks here and full reevaluation later.
        refs = {arm: child('rb_qvu_eval.py', 'reference-'+arm, '--arm', arm,
                    '--training-run', str(args.project/'runs'/attempt),
                    *(() if arm == 'RB1-07' else ('--eval-limit', '16')))
                for arm,attempt in LEGACY.items()}
        calibration_s = refs['RB1-07']['calibration.seconds']
        # LoRA dev timing is historically about5% below side-branch timing;
        # reserve10% explicitly rather than treating it as a matched measurement.
        measured_dev_s = refs['RB1-07']['dev_eval.seconds']
        dev_s = measured_dev_s * 1.1
        training = {arm: value['train.steady_step_seconds']*13115 + 8*dev_s + 2*calibration_s +
            value['load.seconds'] + 30 for arm,value in arms.items()}
        # All four complete legacy reevaluations are reserved even though one was
        # measured above, so that the forecast is conservative and transparent.
        reevaluation_s = 4*(dev_s + calibration_s + 30)
        intervention_s = 3*dev_s + 30  # normal + swap (recipient and donor)
        smoke_s = time.monotonic()-tick
        initial = (sum(training[a] for a in ('H-QVU','T0-QVU')) + reevaluation_s +
            intervention_s + smoke_s + 120)/14400 + args.prior_node_hours
        conditional = initial + sum(training[a] for a in ('U-25','T0-U-25'))/14400
        cost = {'contract':'E-0031@1', 'source_commit':os.environ['SR_CODE_COMMIT'],
            'prior_measured_node_hours':args.prior_node_hours,
            'measured_LoRA_dev_seconds':measured_dev_s, 'dev_timing_reserve_factor':1.1,
            'training_seconds':training, 'calibration_seconds':calibration_s,
            'dev_seconds':dev_s, 'reevaluation_seconds':reevaluation_s,
            'intervention_seconds':intervention_s, 'smoke_seconds':smoke_s,
            'analysis_and_retry_reserve_seconds':120, 'initial_node_hours':initial,
            'with_conditionals_node_hours':conditional, 'initial_cap':1.3, 'total_cap':2.0,
            'within_caps':initial <= 1.3 and conditional <= 2.0,
            'method':'80 steady steps;13115 total;8 dev and2 calibration passes per arm;LoRA dev timing x1.1 reserve;MIG-hours/4;full4 legacy reevaluations;3 intervention dev passes;this smoke plus120seconds;all prior measured node-hours added once'}
        write_json(out/'cost-estimate.json', cost)
        metrics.update({'arms':arms, 'references':refs, 'source_commit':os.environ['SR_CODE_COMMIT'],
            'hardware':torch.cuda.get_device_name(), 'initial_projected_node_hours':initial,
            'conditional_projected_node_hours':conditional, 'cost_within_caps':cost['within_caps']})
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-tick
        write_json(out/'manifest.json', {'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'], 'contract':'E-0031@1', 'stage':'smoke',
            'metrics':metrics, 'artifacts':[{'path':str(p),'role':p.name} for p in sorted(out.iterdir())
                if p.is_file() and p.name != 'manifest.json'],
            'validator':{'name':'e0031_smoke','passed':passed}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--lr-lock', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior-node-hours', type=float, required=True)
    run(p.parse_args())
