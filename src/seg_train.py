"""E-0025/E-0027/E-0030 training at the RB1 budget; no confirmation reader."""
import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb_data import plan_batches
from rb_train import load_data
from rb1_artifacts import verify_lr_lock
from seg_data import segment_plan, ORDER_SHA256_BY_SEED, STEPS, EXAMPLES, SEEDS
from seg_eval import evaluate, token_losses, arm_write
from seg_model import (make_model, KEEP, FAMILIES, PARAMETERS, state_description, parameter_digest,
                       SEED_ARMS, contract, E0032_ARMS, WRITE_RULE, SEG05_INITIAL_SHA256)
from bypass_model import BranchModel
from v1_run import lr_factor, preserved_evaluation_state, example_mean_objective, TrainingDiverged
from v5_train import file_sha256, write_json, plan_metadata
from rb_data import BASE_SEED, STUDY
from v1_data import derive_seed


def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    return tok


def seed_plan(splits, tok, seed, excluded):
    """First STEPS batches of the R1 stream for this seed (fixed sample/step budget).

    For seed 0 this is exactly E-0025's 2M-token-budget plan, which stops at
    STEPS; the registered order hash checks that equivalence.
    """
    rb = plan_batches(splits, tok, 'R1', seed, excluded, token_budget=None)
    if len(rb) < STEPS or any(len(b) != 2 for b in rb[:STEPS]):
        raise ValueError('R1 stream shorter than the fixed segmented step budget')
    return rb[:STEPS]


def prepare_plan(args, tok, require_registered=True):
    seed = getattr(args, 'seed', 0)
    if seed not in ORDER_SHA256_BY_SEED:
        raise ValueError('unknown segmented seed')
    lock = verify_lr_lock(args.lr_lock)
    if file_sha256(args.split_manifest) != lock['split_manifest_sha256']:
        raise ValueError('segmented split differs from the accepted RBc lock')
    if seed in (3, 4, 5) or getattr(args, 'preopening', False):
        from seg_a2_predata import load_data as load_preopening
        splits, excluded = load_preopening(args.data_dir, args.split_manifest, tok)
    else:
        splits, excluded = load_data(args.data_dir, args.split_manifest, tok)
    rb = seed_plan(splits, tok, seed, [(e['task'], e['id']) for e in excluded])
    metadata = plan_metadata(rb)
    registered = ORDER_SHA256_BY_SEED[seed]
    if registered is None and require_registered:
        raise ValueError('segmented seed order hash is not registered')
    if ((registered is not None and metadata['plan.order_sha256'] != registered) or
            metadata['plan.steps'] != STEPS or metadata['plan.examples'] != EXAMPLES):
        raise ValueError('segmented plan is not the registered RB1 sample budget/order')
    return splits, segment_plan(rb, splits, tok), metadata, excluded, lock


class Accounting:
    def __init__(self):
        self.parts = {}

    def add(self, parts):
        for phase, layers in parts.items():
            for i, counts in layers.items():
                target = self.parts.setdefault(phase, {}).setdefault(str(i),
                    dict(attention=0, ffn=0, writes=0, reads=0, valid_tokens=0))
                for k, v in counts['base'].items():
                    target[k] += v
                for k in ('writes', 'reads', 'valid_tokens'):
                    value = counts['side'][k]
                    target[k] += value.detach() if torch.is_tensor(value) else value

    def finish(self):
        return {phase: {i: {k: int(v) for k, v in counts.items()} for i, counts in layers.items()}
                for phase, layers in self.parts.items()}


def train(model, arm, batches, tok, lr, output, monitor=None):
    model.train()
    params = list(model.adapter_parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    totals = {'train.steps': 0, 'train.examples': 0, 'train.input_tokens': 0,
              'train.segment1_tokens': 0, 'train.segment2_tokens': 0, 'train.loss_tokens': 0}
    accounting, grad_seen, durations = Accounting(), set(), []
    segment1_writes = []
    state_shapes, evaluation_seconds = {}, 0.
    forward_seconds = {'segment1': 0., 'segment2': 0.}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (output/'steps.jsonl').open('w') as log:
        for step, examples in enumerate(batches):
            torch.cuda.synchronize(); tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            ce, events = 0., []
            for e in examples:
                first_write = arm_write(tok, arm, e['segment1_ids']) if arm in KEEP else None
                if arm in KEEP:
                    segment1_writes.append(len(e['segment1_ids']) if first_write is None else sum(first_write))
                losses, parts, result = token_losses(model, arm, e, tok.pad_token_id, timing_events=events,
                                                     first_write=first_write)
                if isinstance(model, BranchModel) and arm in KEEP:
                    state_shapes = state_description(result.cache.memory)
                accounting.add(parts)
                n1 = len(e['segment1_ids']) if arm in KEEP else 0
                n2 = len(e['input_ids'])
                totals['train.segment1_tokens'] += n1
                totals['train.segment2_tokens'] += n2
                totals['train.input_tokens'] += n1+n2
                totals['train.loss_tokens'] += losses.numel()
                totals['train.examples'] += 1
                example_mean_objective(losses, len(examples)).backward()
                ce += losses.detach().mean().item()
                del losses, parts, result
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=False).item()
            if not math.isfinite(norm):
                raise TrainingDiverged('nonfinite segmented gradient', totals)
            grad_seen.update(id(p) for p in params if p.grad is not None)
            rate = lr*lr_factor(step, len(batches))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.step()
            totals['train.steps'] += 1
            torch.cuda.synchronize(); duration = time.monotonic()-tick
            for phase, begin, end in events:
                forward_seconds[phase] += begin.elapsed_time(end)/1000
            durations.append(duration)
            log.write(json.dumps({'step': step+1, 'example_mean_nll': ce/len(examples),
                'preclip_norm': norm, 'lr': rate, 'seconds': duration,
                'input_tokens': sum(len(e['input_ids'])+(len(e['segment1_ids']) if arm in KEEP else 0)
                                    for e in examples)})+'\n')
            log.flush()
            write_json(output/'usage.json', totals)
            if monitor and step+1 in {math.ceil(len(batches)*f) for f in (.25, .5, .75)}:
                tick = time.monotonic()
                with preserved_evaluation_state(model):
                    monitor(step+1, model)
                evaluation_seconds += time.monotonic()-tick
    counts = accounting.finish()
    if isinstance(model, BranchModel):
        for phase, layers in counts.items():
            valid = totals['train.'+phase+'_tokens']
            for c in layers.values():
                assert c['valid_tokens'] == valid
                assert c['attention'] == c['ffn'] == totals['train.examples']
                expected = 0 if model.config.branch == 'kernel_tokenwise' else valid
                # E-0032: only segment-1 writes follow the registered mask; reads and
                # segment-2 writes stay on every valid token.
                writes = sum(segment1_writes) if phase == 'segment1' and expected else expected
                assert c['reads'] == expected and c['writes'] == writes
    seconds = time.monotonic()-started-evaluation_seconds
    totals.update({'train.seconds': seconds, 'intermediate.seconds': evaluation_seconds,
        'train.steady_step_seconds': sum(durations[10:])/len(durations[10:]) if len(durations)>10 else None,
        'train.peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'params.with_grad': sum(p.numel() for p in params if id(p) in grad_seen),
        'runtime.accounting': counts, 'state.shapes': state_shapes,
        'state.M_c_bytes': sum(x['M_c_bytes'] for x in state_shapes.values()),
        'state.counts_bytes': sum(x['counts_bytes'] for x in state_shapes.values()),
        'train.forward_seconds': forward_seconds})
    if segment1_writes:
        totals['train.segment1_writes'] = write_summary(segment1_writes)
    return totals


def write_summary(counts):
    return {'examples': len(counts), 'total': sum(counts), 'mean': sum(counts)/len(counts),
            'min': min(counts), 'max': max(counts)}


@contextmanager
def original_backbone(model):
    layers = model.adapters()
    for i, layer in layers.items():
        model.model.model.layers[i] = layer.original
    try:
        yield model.model
    finally:
        for i, layer in layers.items():
            model.model.model.layers[i] = layer


def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    values, passed, start = {}, False, time.monotonic()
    try:
        e0032 = args.arm in E0032_ARMS
        if args.arm not in SEED_ARMS.get(args.seed, ()) or args.stage not in ('train', 'base'):
            raise ValueError('unknown segmented seed/arm/stage')
        if (args.stage == 'base') != (args.arm == 'SEG-00'):
            raise ValueError('Base_seg is evaluation only')
        tok = tokenizer()
        splits, batches, original_plan, excluded, lock = prepare_plan(args, tok)
        lr = None if args.arm == 'SEG-00' else lock['locked_lrs'][FAMILIES[args.arm]]
        values.update(original_plan)
        values.update({'learning_rate': lr, 'source.commit': os.environ['SR_CODE_COMMIT'],
            'model_revision': REVISION, 'tokenizer_revision': REVISION,
            'environment.lock_sha256': file_sha256(Path(__file__).resolve().parents[1]/'uv.lock'),
            'lr_lock.sha256': file_sha256(args.lr_lock), 'split_manifest.sha256': file_sha256(args.split_manifest),
            'hardware': torch.cuda.get_device_name(), 'exec.pbs_job_id': os.environ.get('PBS_JOBID'),
            'hardware.cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'failed': False, 'seed': args.seed,
            'init_seed': derive_seed(BASE_SEED, STUDY, 'R1', 'init', args.seed),
            'train_seed': derive_seed(BASE_SEED, STUDY, 'R1', 'train', args.seed)})
        write_json(out/'config.json', vars(args)); write_json(out/'excluded-train.json', excluded)
        write_json(out/'seams-train.json', [e['seam'] for b in batches for e in b])
        tick = time.monotonic(); model = make_model(args.arm, seed=args.seed)
        values['load.seconds'] = time.monotonic()-tick
        if args.arm != 'SEG-00':
            values['initial.adapter_sha256'] = parameter_digest(model)
        if args.arm in WRITE_RULE:
            values['initial.equals_SEG-05'] = values['initial.adapter_sha256'] == SEG05_INITIAL_SHA256[args.seed]
            if not values['initial.equals_SEG-05']:
                raise ValueError('write-mask arm does not start from the registered SEG-05 parameters')
        values['params.stored'] = PARAMETERS[args.arm]
        values['params.trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.arm in KEEP or e0032:
            # Same allocation and shape, before training. No cross-job bitwise gate.
            # E-0032 rule 1: every new arm (Keep, Reset, T0) starts at Base_seg.
            with original_backbone(model) as base:
                _, base_preds, _ = evaluate(base, 'SEG-00', splits, tok)
            _, initial, initial_audit = evaluate(model, args.arm, splits, tok)
            values['initial.identical_to_Base_seg'] = initial == base_preds
            if not values['initial.identical_to_Base_seg']:
                raise ValueError('zero initialized R_keep does not match Base_seg')
            write_json(out/'initial-base-predictions.json', base_preds)
            write_json(out/'initial-predictions.json', initial)
            if args.arm in WRITE_RULE:
                values['dev.segment1_writes'] = write_summary(initial_audit['segment1_writes'])
                write_json(out/'segment1-writes-dev.json', initial_audit['segment1_writes'])
        elif args.seed in (3, 4, 5):
            im, initial, _ = evaluate(model, args.arm, splits, tok)
            write_json(out/'initial-predictions.json', initial)
            values['initial.dev_metrics'] = im
        if args.stage == 'train':
            def monitor(step, current):
                m, p, _ = evaluate(current, args.arm, splits, tok)
                write_json(out/f'dev-step-{step}.json', {'step': step, 'metrics': m, 'predictions': p})
            # E-0032 records only the final checkpoint's dev predictions (no curve evals).
            values.update(train(model, args.arm, batches, tok, lr, out, None if e0032 else monitor))
            if values['params.with_grad'] != PARAMETERS[args.arm]:
                raise ValueError('some segmented adapter parameters never received gradients')
            tick = time.monotonic(); model.export_adapter(out/'adapter')
            values['save.seconds'] = time.monotonic()-tick
        tick = time.monotonic(); m, p, audit = evaluate(model, args.arm, splits, tok)
        values.update(m); values['eval.seconds'] = time.monotonic()-tick
        write_json(out/'predictions.json', p); write_json(out/'evaluation-audit.json', audit)
        if args.stage == 'train':
            del model; gc.collect(); torch.cuda.empty_cache()
            tick = time.monotonic(); model = make_model(args.arm, out/'adapter', seed=args.seed)
            m2, p2, _ = evaluate(model, args.arm, splits, tok)
            values['reload.seconds'] = time.monotonic()-tick
            values['reload.identical'] = p == p2 and all(values[k] == v for k, v in m2.items())
            if not values['reload.identical']:
                raise ValueError('segmented same-job reload differs')
            if (args.seed in (3, 4, 5) and args.arm == 'SEG-05') or e0032:
                om, off, oa = evaluate(model, args.arm, splits, tok, kind='iv_off')
                values['off.identical_to_Base_seg'] = off == base_preds
                if not values['off.identical_to_Base_seg']:
                    raise ValueError('dev iv_off differs from same-allocation Base_seg')
                write_json(out/'iv-off-dev.json', {'metrics': om, 'predictions': off, 'audit': oa})
            if e0032 and args.arm in KEEP:
                # E-0025/E-0027 iv_swap on the reloaded final checkpoint: segment-1-only
                # within-task donor state (donor written under the same mask rule).
                tick = time.monotonic()
                sm, swap, sa = evaluate(model, args.arm, splits, tok, kind='iv_swap')
                values['swap.seconds'] = time.monotonic()-tick
                values.update({'swap.'+k: v for k, v in sm.items()})
                write_json(out/'iv-swap-dev.json', {'metrics': sm, 'predictions': swap, 'audit': sa})
            write_json(out/'checkpoint.json', {name: {'path': str(out/'adapter'/name),
                'sha256': file_sha256(out/'adapter'/name), 'bytes': (out/'adapter'/name).stat().st_size}
                for name in ('config.json', 'adapter.pt')})
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        artifacts = [{'path': str(p), 'role': p.name, 'sha256': file_sha256(p), 'bytes': p.stat().st_size}
                     for p in out.iterdir() if p.is_file() and p.name != 'manifest.json']
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial',
            'contract': contract(args.seed, args.arm),
            'arm': args.arm, 'seed': args.seed, 'phase': 'SEG', 'stage': args.stage,
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': values, 'artifacts': artifacts,
            'validator': {'name': 'segmented_training', 'passed': passed}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('SEG-00', *FAMILIES), required=True)
    p.add_argument('--stage', choices=('train', 'base'), required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, default=0)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--lr-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
