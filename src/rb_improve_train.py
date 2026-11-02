"""E-0029 runner. Calibration and smoke never open new dev predictions."""
import argparse
from contextlib import contextmanager
import gc
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from bypass_model import BranchModel
from calibrate import MODEL, REVISION
from rb1_artifacts import read, reference, verify_lr_lock
from rb_data import plan_batches
from rb_improve_init import InitializationCheckFailed, calibration_examples, initialize_gradient
from rb_improve_lr import verify_lock
from rb_improve_read import diagnose
from rb_tap_check import initial_state
from rb_tap_identity import PINNED_PROFILE, reference_check
from rb_train import E0029_ARMS, FAMILIES, IG_ARMS, PARAMETERS, UP_LORA_ARMS, load_data, make_model, train
from v1_run import evaluate, preserved_evaluation_state
from v5_train import plan_metadata, write_json


class ForwardAccounting:
    def __init__(self):
        self.tokens, self.calls, self.handles = 0, 0, []

    def attach(self, model):
        def observe(module, args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            mask = kwargs.get('attention_mask')
            self.tokens += ids.numel() if mask is None else int(mask.sum())
            self.calls += 1
        self.handles.append(model.model.register_forward_pre_hook(observe, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def backbone_only(model):
    if isinstance(model, BranchModel):
        from rb1_intervene import original_backbone
        with original_backbone(model) as backbone:
            yield backbone
    else:
        targets = []
        for (index, path), adapter in model._targets.items():
            parent = model.model.model.layers[index]
            parts = path.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            targets.append((parent, parts[-1], adapter))
        for parent, name, adapter in targets:
            setattr(parent, name, adapter.base)
        try:
            yield model.model
        finally:
            for parent, name, adapter in targets:
                setattr(parent, name, adapter)


def init_check(model, arm, random_state, require_change=True):
    digest, actual = initial_state(model)
    allowed = '.projection.A_v' if arm in IG_ARMS else '.projection.A_q' if arm in ('P-T', 'P-N') else None
    if actual.keys() != random_state.keys():
        raise ValueError('initial parameter names differ from random reference')
    changed = [name for name, value in actual.items() if not torch.equal(value, random_state[name])]
    if any(allowed is None or not name.endswith(allowed) for name in changed):
        raise ValueError('initialization changed an unregistered parameter')
    if allowed and require_change and len(changed) != len(model.config.layers):
        raise ValueError('initialization did not change exactly the intended per-layer matrices')
    if arm in UP_LORA_ARMS and any(path != 'mlp.up_proj' for _, path in model._targets):
        raise ValueError('up-only LoRA changed another module')
    return {'passed': True, 'sha256': digest, 'changed_from_random_reference': changed,
            'parameters': sum(p.numel() for p in model.adapter_parameters()),
            'allowed_parameter_suffix': allowed}


def random_reference(arm, phase):
    # P arms are already paired in make_model; their random references must be unpaired.
    reference_arm = {'G-T': 'RB1-09', 'P-T': 'RB1-09', 'G-QN': 'Q-N',
                     'G-N': 'X-01', 'P-N': 'X-01'}.get(arm, arm)
    model = make_model(reference_arm, phase=phase)
    _, state = initial_state(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return state


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics, passed = {}, False
    forward_accounting = ForwardAccounting()
    tick = time.monotonic()
    phase = 'V0' if args.stage == 'calibration' else 'R1'
    try:
        if args.arm not in E0029_ARMS or not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError('invalid E-0029 arm/LR')
        if args.stage == 'calibration' and args.arm != 'L-56':
            raise ValueError('up-only LR representative is L-56')
        if args.stage != 'smoke' and (args.smoke_steps is not None or args.eval_limit is not None):
            raise ValueError('partial budgets are smoke-only')
        if args.arm not in UP_LORA_ARMS and args.lr != .001:
            raise ValueError('side branch LR is fixed at1e-3')
        if args.stage == 'train':
            up_lock = verify_lock(args.up_lr_lock)
            if up_lock['identity']['source.commit'] != os.environ['SR_CODE_COMMIT']:
                raise ValueError('LoRA calibration source differs from training release')
            lr_lock = up_lock if args.arm in UP_LORA_ARMS else verify_lr_lock(args.lr_lock)
            if lr_lock['locked_lrs'][FAMILIES[args.arm]] != args.lr:
                raise ValueError('training LR differs from its lock')
            split_hash = reference(args.split_manifest)['sha256']
            if lr_lock['split_manifest_sha256'] != split_hash or up_lock['split_manifest_sha256'] != split_hash:
                raise ValueError('split differs from LR calibration')
            metrics.update({'lr_lock.sha256': reference(args.lr_lock)['sha256'],
                'up_lr_lock.sha256': reference(args.up_lr_lock)['sha256']})
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        tok.pad_token_id = tok.eos_token_id
        splits, excluded = load_data(args.data_dir, args.split_manifest, tok)
        examples, sample = calibration_examples(splits, tok, excluded)
        write_json(out/'ig-samples.json', sample)
        write_json(out/'excluded-train.json', excluded)
        config = dict(vars(args), phase=phase, seed=0, three_epochs=False)
        write_json(out/'config.json', config)
        metrics.update({'learning_rate': args.lr, 'failed': False, 'dev_read': False,
            'source.commit': os.environ['SR_CODE_COMMIT'], 'model_revision': REVISION,
            'tokenizer_revision': REVISION, 'environment.lock_sha256': reference(Path(__file__).resolve().parents[1]/'uv.lock')['sha256'],
            'split_manifest.sha256': reference(args.split_manifest)['sha256'],
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'hardware': torch.cuda.get_device_name(),
            'exec.pbs_job_id': os.environ['PBS_JOBID'], 'ig.sample_sha256': sample['examples_sha256']})
        baseline_state = random_reference(args.arm, phase)
        start = time.monotonic()
        checkpoint = None
        if args.stage == 'train' and args.arm in IG_ARMS:
            init_root = Path(args.ig_init)
            init_manifest = read(init_root/'manifest.json')
            if (init_manifest.get('status') != 'complete' or init_manifest.get('arm') != args.arm or
                    init_manifest.get('stage') != 'ig' or init_manifest.get('validator', {}).get('passed') is not True or
                    init_manifest['metrics'].get('source.commit') != os.environ['SR_CODE_COMMIT'] or
                    init_manifest['metrics'].get('ig.sample_sha256') != sample['examples_sha256']):
                raise ValueError('I-G artifact differs from training release/sample')
            checkpoint = init_root/'adapter'
            inventory = read(init_root/'checkpoint.json')
            if any(reference(checkpoint/name) != record for name, record in inventory.items()):
                raise ValueError('I-G checkpoint bytes changed')
            metrics['ig.initialization'] = reference(init_root/'manifest.json')
        model = make_model(args.arm, phase=phase, checkpoint=checkpoint)
        forward_accounting.attach(model)
        metrics['load.seconds'] = time.monotonic()-start
        if args.stage == 'ig':
            if args.arm not in IG_ARMS:
                raise ValueError('I-G collection requires a G arm')
            try:
                values = initialize_gradient(model, examples, tok)
            except InitializationCheckFailed as exc:
                write_json(out/'ig-initialization.json', exc.diagnostics)
                raise
            write_json(out/'ig-initialization.json', values)
            metrics.update({'ig.collection_seconds': values['collection_seconds'],
                'ig.total_seconds': values['total_seconds'], 'ig.input_tokens': sample['input_tokens']})
        registration = init_check(model, args.arm, baseline_state, require_change=args.stage != 'smoke')
        registration['timing_only_random_IG'] = args.stage == 'smoke' and args.arm in IG_ARMS
        write_json(out/'initialization.json', registration)
        metrics['init.parameters.sha256'] = registration['sha256']
        del baseline_state
        metrics['params.stored'] = sum(p.numel() for p in model.adapter_parameters())
        metrics['params.trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if metrics['params.stored'] != PARAMETERS[args.arm] or metrics['params.trainable'] != PARAMETERS[args.arm]:
            raise ValueError('registered parameter count differs')
        if args.stage != 'ig':
            budget = 200000 if args.stage == 'calibration' else 2000000
            batches = plan_batches(splits, tok, phase, 0, [(r['task'], r['id']) for r in excluded], budget)
            if args.stage == 'smoke':
                batches = batches[:args.smoke_steps or 80]
            metrics.update(plan_metadata(batches))
            if args.stage == 'train' and (metrics['plan.steps'] != 13115 or metrics['plan.examples'] != 26230):
                raise ValueError('E-0029 requires the locked RB1 full training budget')
            def monitor(step, current):
                values, rows = evaluate(current, splits, tok, split_names=('dev',))
                write_json(out/f'dev-step-{step}.json', {'step': step, 'metrics': values, 'predictions': rows})
                return rows
            if args.stage == 'train':
                metrics['dev_read'] = True
                start = time.monotonic()
                with preserved_evaluation_state(model):
                    initial = monitor(0, model)
                    local_rows = None
                    if metrics['hardware'] != PINNED_PROFILE:
                        with backbone_only(model) as backbone:
                            local_metrics, local_rows = evaluate(backbone, splits, tok, split_names=('dev',))
                        write_json(out/'dev-base-same-allocation.json', {'hardware': metrics['hardware'],
                            'job_id': os.environ['PBS_JOBID'], 'source_commit': os.environ['SR_CODE_COMMIT'],
                            'metrics': local_metrics, 'predictions': local_rows})
                    from rb1_analysis import ACCEPTED_BASE_MANIFEST_SHA256
                    if reference(Path(args.base_run)/'manifest.json')['sha256'] != ACCEPTED_BASE_MANIFEST_SHA256:
                        raise ValueError('Base manifest differs from E-0021 locked reference')
                    pinned = [row for row in read(Path(args.base_run)/'predictions.json') if row['split'] == 'dev']
                    initial_gate = reference_check(initial, pinned,
                        metrics['hardware'], same_allocation=local_rows)
                    write_json(out/'initial-identity.json', initial_gate)
                    if not initial_gate['passed']:
                        raise ValueError('step0 predictions differ from required Base reference')
                metrics['init_eval.seconds'] = time.monotonic()-start
                if isinstance(model, BranchModel):
                    with preserved_evaluation_state(model):
                        write_json(out/'step0-read.json', diagnose(model, examples[:64], tok))
            metrics.update(train(model, batches, tok, args.lr, out, monitor if args.stage == 'train' else None))
            if metrics['params.with_grad'] != PARAMETERS[args.arm]:
                raise ValueError('not all registered parameters received a gradient')
            if args.stage != 'smoke':
                start = time.monotonic()
                split = 'calibration' if args.stage == 'calibration' else 'dev'
                values, rows = evaluate(model, splits, tok, split_names=(split,))
                metrics.update(values)
                metrics['eval.seconds'] = time.monotonic()-start
                write_json(out/'predictions.json', rows)
            if args.stage == 'train' and isinstance(model, BranchModel):
                write_json(out/'final-read.json', diagnose(model, examples[:64], tok))
        if args.stage != 'calibration':
            model.export_adapter(out/'adapter')
            write_json(out/'checkpoint.json', {name: reference(out/'adapter'/name) for name in ('config.json', 'adapter.pt')})
            if args.stage == 'train':
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = make_model(args.arm, phase=phase, checkpoint=out/'adapter')
                forward_accounting.attach(model)
                values, reloaded = evaluate(model, splits, tok, split_names=('dev',))
                metrics['reload.identical'] = rows == reloaded and all(metrics[key] == value for key, value in values.items())
                if not metrics['reload.identical']:
                    raise ValueError('checkpoint reload differs')
        passed = True
    finally:
        forward_accounting.close()
        metrics['failed'] = not passed
        metrics['runtime.total_forward_input_tokens'] = forward_accounting.tokens
        metrics['runtime.total_forward_calls'] = forward_accounting.calls
        metrics['elapsed_s'] = time.monotonic()-tick
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'contract': 'E-0029@1',
            'arm': args.arm, 'family': FAMILIES[args.arm], 'seed': 0, 'phase': phase, 'stage': args.stage,
            'metrics': metrics, 'artifacts': [{'path': str(p), 'role': p.name}
                for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'],
            'validator': {'name': 'e0029_runtime', 'passed': passed}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=E0029_ARMS, required=True)
    p.add_argument('--stage', choices=('smoke', 'ig', 'calibration', 'train'), required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--lr-lock', type=Path)
    p.add_argument('--up-lr-lock', type=Path)
    p.add_argument('--ig-init', type=Path)
    p.add_argument('--base-run', type=Path)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke-steps', type=int)
    p.add_argument('--eval-limit', type=int)
    run(p.parse_args())
