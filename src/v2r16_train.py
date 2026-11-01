"""E-0016 D1/D2a/D2b/D3 runner. Execute only in an allocated GPU job.

Schedule helpers deliberately have no torch dependency. Formal runs use the
sealed 6000-step recipe; smoke output is explicitly excluded from evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from v1_data import derive_seed
from v2_schedule import DATA_HASHES, ORDER_HASH_ENCODING, load_data, make_step_schedule

ARMS = ('D1', 'D2a', 'D2b', 'D3')
SCHEDULE_ARMS = (*ARMS, 'D4', 'D5')
RESUME_SEED_RULE = ('base = derive_seed(20261029, "concat_loop_v2", "V2R16", arm, 0); '
                    'epoch indices shuffled by random.Random(base + epoch), epochs 0..2')
STANDARD = ('normal', 'zero', 'shuffled', 'permuted', 'oracle_key', 'oracle_value')


def make_arm_schedule(rows, arm, steps=6000):
    if arm not in SCHEDULE_ARMS:
        raise ValueError('unsupported E-0016 arm')
    if arm in ('D1', 'D3', 'D4'):
        return make_step_schedule(rows, steps)
    if type(steps) is not int or steps <= 0 or not rows or len(rows) % 8:
        raise ValueError('positive integer steps and complete batch-8 rows required')
    order = []
    seed = derive_seed(20261029, 'concat_loop_v2', 'V2R16', arm, 0)
    for epoch in range(3):
        indices = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(indices)
        order.extend(indices)
    if steps * 8 > len(order):
        raise ValueError('step budget exceeds three epochs')
    batches = [order[i:i+8] for i in range(0, steps*8, 8)]
    return {'order': order, 'batches': batches, 'steps': steps,
            'tokens': sum(len(rows[i]['support_input_ids']) + len(rows[i]['question_input_ids'])
                          for batch in batches for i in batch),
            'warmup_steps': math.ceil(.05*steps), 'step_budget': steps,
            'order_sha256': hashlib.sha256(''.join(rows[i]['episode_id']+'\n' for i in order)
                                           .encode('utf-8')).hexdigest(),
            'order_hash_encoding': ORDER_HASH_ENCODING, 'order_seed_rule': RESUME_SEED_RULE,
            'order_seed': seed, 'order_arm': arm}


def trim_smoke_schedule(schedule, rows, steps):
    if type(steps) is not int or steps < 1:
        raise ValueError('positive integer smoke steps required')
    plan = dict(schedule)
    plan['batches'] = schedule['batches'][:steps]
    plan['steps'] = len(plan['batches'])
    plan['tokens'] = sum(len(rows[i]['support_input_ids']) + len(rows[i]['question_input_ids'])
                         for batch in plan['batches'] for i in batch)
    plan['warmup_steps'] = math.ceil(.05*plan['steps'])
    plan['smoke_steps'] = steps
    return plan


def file_record(path):
    payload = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def checkpoint_records(directory):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): file_record(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    if not files:
        raise ValueError('empty adapter checkpoint')
    return files


def resume_provenance(checkpoint, smoke):
    checkpoint = Path(checkpoint).resolve()
    manifest_path, metrics_path = checkpoint.parent/'manifest.json', checkpoint.parent/'metrics.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'complete' or manifest.get('arm') != 'D1' or
            manifest.get('contract') != 'E-0016@1' or manifest.get('seed') != 0 or
            manifest.get('validator', {}).get('passed') is not True):
        raise ValueError('D2 requires a completed D1 adapter')
    if not smoke and manifest.get('smoke_steps'):
        raise ValueError('formal D2 cannot resume a smoke checkpoint')
    files = checkpoint_records(checkpoint)
    if files != manifest.get('checkpoint_files'):
        raise ValueError('D1 checkpoint bytes differ from recorded provenance')
    if set(files) != {'adapter.pt', 'config.json'}:
        raise ValueError('only adapter parameters/config may be resumed')
    metrics = json.loads(metrics_path.read_text())
    reload = metrics.get('checkpoint_reload', {})
    if reload.get('full_records_identical') is not True or reload.get('summaries_identical') is not True:
        raise ValueError('D1 reload check did not pass')
    if not smoke and metrics.get('train', {}).get('steps') != 6000:
        raise ValueError('D1 actual training did not complete 6000 steps')
    return {'checkpoint': str(checkpoint), 'checkpoint_files': files,
            'source_manifest': file_record(manifest_path), 'source_metrics': file_record(metrics_path),
            'adapter_only': True, 'optimizer_state_loaded': False,
            'optimizer': 'fresh AdamW, weight_decay=0', 'warmup': 'fresh 300 steps for formal runs'}


def evaluation_protocols(arm):
    if arm == 'D3':
        return {'hard': ('hard', STANDARD), 'soft': ('soft', STANDARD),
                'marginal': ('marginal', ('normal', 'zero', 'shuffled', 'permuted'))}
    protocols = {'standard': ('standard', STANDARD)}
    if arm == 'D1':
        protocols['teacher'] = ('standard', ('tf_normal', 'tf_shuffled', 'tf_wrong', 'zero'))
    return protocols


def run(args):
    import torch
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION, load_base
    from v2_memory import config
    from v2r_data import METADATA_SHA256, load_variant
    from v2r16_memory import evaluate, freeze_for_address, train_forward
    from v2_train import adapted, train, write_json

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    started = time.monotonic()
    result, passed, failure, provenance = {}, False, None, None
    try:
        train_rows = load_data(args.data_dir/'train.jsonl', 'train')
        # Verify all released E=16 bytes, even though curves and final scoring use dev.
        load_data(args.data_dir/'calibration.jsonl', 'calibration')
        dev = load_data(args.data_dir/'dev.jsonl', 'dev')
        e1_metadata, e1 = load_variant(args.e1_data_dir, 1, 'dev') if args.arm != 'D3' else (None, None)
        schedule = make_arm_schedule(train_rows, args.arm, args.step_budget)
        if args.smoke_steps:
            schedule = trim_smoke_schedule(schedule, train_rows, args.smoke_steps)
        if args.smoke_eval_limit:
            dev = dev[:args.smoke_eval_limit]
            if e1 is not None:
                e1 = e1[:args.smoke_eval_limit]
        public_schedule = {k: v for k, v in schedule.items() if k not in ('order', 'batches')}
        write_json(output/'schedule.json', public_schedule)
        if args.checkpoint:
            provenance = resume_provenance(args.checkpoint, bool(args.smoke_steps))
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for rows in (dev, e1):
            for row in rows or []:
                ids = tokenizer.encode(' '+row['query']['head'], add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError('query subject is not a single token')
                row['_subject_token_id'] = ids[0]
        prior = Counter(r['query']['answer_token_id'] for r in train_rows).most_common(1)[0][0]
        cfg = config('direct_hidden', 'soft_slot', 'F_out')
        base = load_base()
        result = {'arm': args.arm, 'config': asdict(cfg), 'lr': .003, 'seed': 0,
                  'batch_size': 8, 'microbatch_size': 8, 'schedule': public_schedule,
                  'smoke_only': bool(args.smoke_steps), 'eligible_as_evidence': not bool(args.smoke_steps),
                  'resume_provenance': provenance}
        protocols = evaluation_protocols(args.arm)

        def evaluate_all(model, destination):
            summaries, records = {}, {}
            for name, (strategy, controls) in protocols.items():
                summaries[name], records[name] = evaluate(model, dev, pad_id, destination/name, 'dev',
                                                          controls, prior, strategy=strategy)
            if e1 is not None:
                summaries['e1'], records['e1'] = evaluate(
                    model, e1, pad_id, destination/'e1', 'dev', ('normal', 'zero', 'shuffled'),
                    e1_metadata['prior']['answer_token_id'])
            return summaries, records

        def curve_evaluator(model, rows, pad, destination, split, controls, prior_token):
            strategy = 'hard' if args.arm == 'D3' else 'standard'
            controls = ('tf_normal', 'zero', 'tf_shuffled') if args.arm == 'D1' else controls
            return evaluate(model, rows, pad, destination, split, controls, prior_token, strategy=strategy)

        with adapted(base, cfg=cfg, checkpoint=args.checkpoint) as model:
            if asdict(model.config) != asdict(cfg):
                raise ValueError('resume checkpoint does not match fixed E-0016 configuration')
            if args.arm == 'D2a':
                freeze_for_address(model)
            else:
                for parameter in model.adapter_parameters():
                    parameter.requires_grad_(True)
            side = model.adapters()[14].side
            selected = sorted(n for n, p in side.named_parameters() if p.requires_grad)
            result['requires_grad_parameter_names'] = selected
            frozen_before = {n: p.detach().clone() for n, p in side.named_parameters() if not p.requires_grad}
            forward = partial(train_forward, protocol='teacher' if args.arm == 'D1' else 'marginal')
            options = {'forward_fn': forward} if args.arm in ('D1', 'D3') else {}
            result['train'] = train(model, train_rows, schedule, pad_id, .003, 8, output, dev, prior,
                                    curve_evaluator=curve_evaluator, **options)
            if any(not torch.equal(p, frozen_before[n]) for n, p in side.named_parameters() if n in frozen_before):
                raise AssertionError('frozen adapter parameters changed')
            if args.arm == 'D2a':
                seen = result['train']['parameter_names_with_nonzero_gradient']
                if not seen or not set(seen) <= set(selected):
                    raise AssertionError('D2a gradients outside addressing parameters or no nonzero gradient')
            torch.cuda.synchronize()
            tick = time.monotonic()
            result['evaluation'], original_records = evaluate_all(model, output/'evaluation')
            torch.cuda.synchronize()
            result['evaluation_seconds'] = time.monotonic()-tick
            tick = time.monotonic()
            model.export_adapter(output/'adapter')
            torch.cuda.synchronize()
            result['checkpoint_save_seconds'] = time.monotonic()-tick
        tick = time.monotonic()
        with adapted(base, checkpoint=output/'adapter') as reloaded:
            torch.cuda.synchronize()
            load_seconds = time.monotonic()-tick
            if asdict(reloaded.config) != asdict(cfg):
                raise AssertionError('checkpoint reload configuration changed')
            reload_metrics, reload_records = evaluate_all(reloaded, output/'reload')
            if original_records != reload_records or result['evaluation'] != reload_metrics:
                raise AssertionError('checkpoint reload changed full per-example records or summaries')
            torch.cuda.synchronize()
        result['checkpoint_reload'] = {'full_records_identical': True, 'summaries_identical': True,
                                       'protocols': list(reload_records), 'load_seconds': load_seconds,
                                       'load_and_evaluation_seconds': time.monotonic()-tick}
        result['checkpoint_files'] = checkpoint_records(output/'adapter')
        passed = True
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        wall = time.monotonic()-started
        result.update(wall_seconds=wall, mig_instance_hours=wall/3600, node_hours=wall/14400)
        write_json(output/'metrics.json', result)
        artifacts = [{'path': str(p.resolve()), 'role': p.name, **file_record(p)}
                     for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']
        write_json(output/'manifest.json', {
            'status': 'complete' if passed else 'failed', 'experiment': 'E-0016', 'contract': 'E-0016@1', 'arm': args.arm,
            'spec_digest': os.environ.get('SR_SPEC_DIGEST'), 'model': MODEL, 'revision': REVISION,
            'smoke_steps': args.smoke_steps, 'smoke_eval_limit': args.smoke_eval_limit,
            'eligible_as_evidence': passed and not bool(args.smoke_steps),
            'seed': 0, 'lr': .003, 'batch_size': 8, 'microbatch_size': 8,
            'data_dir': str(args.data_dir.resolve()), 'data_sha256': DATA_HASHES,
            'e1_data_dir': str(args.e1_data_dir.resolve()) if args.arm != 'D3' else None,
            'e1_metadata_sha256': METADATA_SHA256[1] if args.arm != 'D3' else None,
            'resume_provenance': provenance, 'schedule': result.get('schedule'),
            'requires_grad_parameter_names': result.get('requires_grad_parameter_names'),
            'checkpoint_files': result.get('checkpoint_files'), 'torch': torch.__version__,
            'hardware': {'device': torch.cuda.get_device_name(0),
                         'multiprocessors': torch.cuda.get_device_properties(0).multi_processor_count},
            'cuda': torch.version.cuda, 'metrics': {'wall_seconds': wall, 'mig_instance_hours': wall/3600,
                                                   'node_hours': wall/14400},
            'artifacts': artifacts, 'failure': failure,
            'validator': {'name': 'v2r16_train', 'passed': passed}})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--e1-data-dir', type=Path,
                        default=Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0112-1/e1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step-budget', type=int, default=6000)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--smoke-eval-limit', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if bool(args.checkpoint) != (args.arm in ('D2a', 'D2b')):
        parser.error('only D2a/D2b require --checkpoint pointing to a completed D1 adapter')
    if args.step_budget != 6000:
        parser.error('E-0016 uses --step-budget 6000; shorten only with --smoke-steps')
    if not 0 <= args.smoke_steps <= args.step_budget:
        parser.error('smoke steps must be 0..6000')
    if args.smoke_eval_limit and (not args.smoke_steps or args.smoke_eval_limit < 2):
        parser.error('smoke eval limit >=2 requires --smoke-steps')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
