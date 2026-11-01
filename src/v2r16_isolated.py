"""E-0016 D4/D5 runner: isolated-encoding write (H-0012). Execute only in an allocated GPU job.

Every support fact is encoded alone in the E=1 format and written at its delimiter into
slot j = its support index (v2_memory.isolated_support, the E-0015 isolated encoding);
the question side and the read protocol are unchanged, and training uses the answer CE
only (no target index). D4 trains from scratch; D5 continues R1's final adapter
(ATT-0122-1): adapter parameters only, fresh AdamW, fresh 300-step warmup and cosine,
data order derive_seed(20261029, "concat_loop_v2", "V2R16", "D5", 0).

E-0017 (--experiment E-0017) replicates D5 with continuation order seed s (--seed) from a
start run of the R1 recipe (--r1-run, recorded seed --r1-seed): either the stage-A R1
retrained at seed s or ATT-0122-1. It also reports the isolated normal/zero/shuffled on
E=16 calibration, descriptively.

Schedules, the E-0016 evaluator and the training loop are shared with T-0017
(v2r16_train, v2r16_memory, v2_train.train). Every control of the gate is evaluated
under the isolated write (new protocol, reported apart from E-0010); the contextual
normal and oracle_value are reported for comparison with E-0015 only.
Smoke output is excluded from evidence. Confirmation is never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import json
import os
from pathlib import Path
import time

from v2_schedule import DATA_HASHES, load_data
from v2r16_analysis import h3_gate
from v2r16_train import STANDARD, checkpoint_records, file_record, make_arm_schedule, trim_smoke_schedule

ARMS = ('D4', 'D5')
R1_RUN = Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0122-1')
# ATT-0122-1's adapter.pt as loaded by E-0016 D5 (ATT-0159-1, RES-0153) while the replay of
# R1's recorded dev predictions was still exact (0/1000 mismatches): the trusted R1 digest.
R1_ADAPTER_SHA256 = '9ea30adb36d914fb09e689fddbb0f27d91e32323231ac359245fcbd3024eb616'
ARM_DIR = 'direct_hidden.soft_slot.F_out'
CONTEXT_CONTROLS = ('normal', 'oracle_value')  # descriptive comparison with E-0015 only
CALIBRATION_CONTROLS = ('normal', 'zero', 'shuffled')  # E-0017 descriptive only
EXPERIMENTS = {'E-0016': 'E-0016@1', 'E-0017': 'E-0017@1'}


def r1_provenance(run, cfg, seed=0, *, adapter_sha256):
    """A run of the R1 recipe (an E-0014/E-0017 v2_train run, e.g. ATT-0122-1 at seed 0):
    completed, formal, E=1 training of the fixed configuration at the expected seed,
    6000 steps, reload-identical. Its adapter.pt must have the expected sha256, which comes
    from outside the source directory (R1_ADAPTER_SHA256, or the digest a stage-A producer
    job published when it ended, pinned in the continuation job)."""
    if not (isinstance(adapter_sha256, str) and len(adapter_sha256) == 64
            and all(c in '0123456789abcdef' for c in adapter_sha256)):
        raise ValueError('an expected lowercase hex sha256 of the start adapter.pt is required')
    manifest_path, metrics_path = run/'manifest.json', run/'metrics.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True
            or manifest.get('seed') != seed or manifest.get('smoke_steps') or manifest.get('smoke_eval_limit')
            or manifest.get('options', {}).get('train_facts') != 1
            or manifest.get('options', {}).get('step_budget') != 6000):
        raise ValueError(f'R1 must be the completed formal E=1 6000-step run at seed {seed}')
    metrics = json.loads(metrics_path.read_text())[ARM_DIR]
    if metrics.get('train', {}).get('steps') != 6000:
        raise ValueError('R1 did not complete 6000 steps')
    if metrics.get('checkpoint_reload', {}).get('predictions_identical') is not True:
        raise ValueError('R1 reload check did not pass')
    if metrics.get('config') != json.loads(json.dumps(asdict(cfg))):
        raise ValueError('R1 configuration differs from the fixed E-0016 configuration')
    files = checkpoint_records(run/ARM_DIR/'adapter')
    if set(files) != {'adapter.pt', 'config.json'}:
        raise ValueError('only adapter parameters/config may be resumed')
    if files['adapter.pt']['sha256'] != adapter_sha256:
        raise ValueError('start adapter.pt differs from the expected producer digest')
    return {'checkpoint': str((run/ARM_DIR/'adapter').resolve()), 'checkpoint_files': files, 'start_seed': seed,
            'expected_adapter_sha256': adapter_sha256,
            'source_manifest': file_record(manifest_path), 'source_metrics': file_record(metrics_path),
            'adapter_only': True, 'optimizer_state_loaded': False,
            'optimizer': 'fresh AdamW, weight_decay=0', 'warmup': 'fresh 300 steps for formal runs'}


def replay_record(comparison, episodes):
    """The start adapter's replay of its source run's recorded contextual dev predictions.

    The episode identities and order must match (an error otherwise). Numerical prediction
    and NLL differences are recorded, not failures: a cross-job difference that the MIG slice
    explains is noted and the run continues (D-0018, D-0019). The start weights are bound by
    r1_provenance to an expected adapter.pt digest from outside the source directory, checked
    before loading and again after it; the same-job save/reload check stays exact."""
    normal = comparison['normal']
    if normal['same_episode_order'] is not True or normal['episodes'] != episodes:
        raise AssertionError('start-run replay does not cover the same dev episodes in the same order')
    return dict(normal, numerical_differences='recorded only (D-0018, D-0019)')


def gate(summary):
    """E-0010@2 H3 under the isolated write (new protocol, reported apart): the reviewed
    exact-fraction v2r16_analysis.h3_gate on correct counts, all-question-position target
    weight; the last-position weight is reported alongside."""
    result = h3_gate(summary, summary['normal']['diagnostics'].get('address.target_weight'))
    result['target_weight_last_position'] = summary['normal']['last_position']['target_weight_mean']
    return result


def run(args):
    import torch
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION, load_base
    from v2_isolated import annotate_isolated
    from v2_memory import collate, config, isolated_support
    from v2_train import adapted, evaluate as legacy_evaluate, train, write_json
    from v2r16_memory import evaluate, train_forward
    from v2r_diagnostics import compare_recorded

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    started = time.monotonic()
    result, passed, failure, provenance = {}, False, None, None
    try:
        train_rows = load_data(args.data_dir/'train.jsonl', 'train')
        calibration = load_data(args.data_dir/'calibration.jsonl', 'calibration')  # E-0017 descriptive only
        if args.experiment != 'E-0017':
            calibration = None
        dev = load_data(args.data_dir/'dev.jsonl', 'dev')
        schedule = make_arm_schedule(train_rows, args.arm, args.step_budget, args.seed)
        if args.smoke_steps:
            schedule = trim_smoke_schedule(schedule, train_rows, args.smoke_steps)
        if args.smoke_eval_limit:
            dev = dev[:args.smoke_eval_limit]
            calibration = calibration and calibration[:args.smoke_eval_limit]
        public_schedule = {k: v for k, v in schedule.items() if k not in ('order', 'batches')}
        write_json(output/'schedule.json', public_schedule)
        cfg = config('direct_hidden', 'soft_slot', 'F_out')
        checkpoint = None
        if args.arm == 'D5':
            provenance = r1_provenance(args.r1_run, cfg, args.r1_seed, adapter_sha256=args.r1_adapter_sha256)
            checkpoint = args.r1_run/ARM_DIR/'adapter'
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        tick = time.monotonic()
        differing = {'train': annotate_isolated(tokenizer, train_rows), 'dev': annotate_isolated(tokenizer, dev)}
        if calibration:
            differing['calibration'] = annotate_isolated(tokenizer, calibration)
        for row in dev + (calibration or []):
            ids = tokenizer.encode(' '+row['query']['head'], add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError('query subject is not a single token')
            row['_subject_token_id'] = ids[0]
        prior = Counter(r['query']['answer_token_id'] for r in train_rows).most_common(1)[0][0]
        base = load_base()
        result = {'experiment': EXPERIMENTS[args.experiment], 'arm': args.arm, 'config': asdict(cfg), 'lr': .003,
                  'seed': args.seed, 'batch_size': 8,
                  'microbatch_size': 8, 'schedule': public_schedule, 'write_protocol': 'isolated',
                  'smoke_only': bool(args.smoke_steps), 'eligible_as_evidence': not bool(args.smoke_steps),
                  'resume_provenance': provenance,
                  'isolated_annotation': {'context_fact_tokens_differ_from_isolated': differing,
                                          'seconds': time.monotonic()-tick}}
        isolated = partial(evaluate, support_fn=isolated_support)

        def evaluate_all(model, destination):
            summaries, records = {}, {}
            summaries['isolated'], records['isolated'] = isolated(model, dev, pad_id, destination/'isolated', 'dev',
                                                                  STANDARD, prior)
            summaries['contextual'], records['contextual'] = evaluate(model, dev, pad_id, destination/'contextual',
                                                                      'dev', CONTEXT_CONTROLS, prior)
            if calibration:
                summaries['calibration_isolated'], records['calibration_isolated'] = isolated(
                    model, calibration, pad_id, destination/'calibration-isolated', 'calibration',
                    CALIBRATION_CONTROLS, prior)
            return summaries, records

        def curve_evaluator(model, rows, pad, destination, split, controls, prior_token):
            return isolated(model, rows, pad, destination, split, controls, prior_token)

        first = collate([train_rows[i] for i in schedule['batches'][0]], pad_id, next(base.parameters()).device)
        with torch.no_grad():
            mask = first.question_mask
            reference = base(first.question_ids, attention_mask=mask,
                             position_ids=(mask.long().cumsum(-1)-1).clamp_min(0), use_cache=False).logits
            last = mask.long().sum(-1)-1
            reference = reference[torch.arange(len(last), device=last.device), last].float()
        forward = partial(train_forward, support_fn=isolated_support)
        with adapted(base, cfg=cfg, checkpoint=checkpoint) as model:
            if asdict(model.config) != asdict(cfg):
                raise ValueError('checkpoint does not match the fixed E-0016 configuration')
            if checkpoint is not None and checkpoint_records(checkpoint) != provenance['checkpoint_files']:
                raise ValueError('start adapter files changed between the digest check and loading')
            for parameter in model.adapter_parameters():
                parameter.requires_grad_(True)
            model.eval()
            if args.arm == 'D4':
                with torch.no_grad():
                    _, logits, _, _ = forward(model, first)
                result['initialization_max_abs'] = float((logits-reference).abs().max())
                if result['initialization_max_abs'] != 0.:
                    raise AssertionError('zero exit differs from Base under the isolated write')
            else:
                # Replay the start run's recorded contextual dev predictions (differences recorded).
                tick = time.monotonic()
                replay = output/'r1-replay'
                full_dev = load_data(args.data_dir/'dev.jsonl', 'dev')
                legacy_evaluate(model, full_dev, pad_id, replay, 'dev', ('normal',), prior)
                comparison = compare_recorded(args.r1_run/ARM_DIR, replay, 'dev', ('normal',))
                provenance['recorded_dev_replay'] = dict(replay_record(comparison, len(full_dev)),
                                                         seconds=time.monotonic()-tick)
            del first, reference
            side = model.adapters()[14].side
            result['requires_grad_parameter_names'] = sorted(n for n, p in side.named_parameters() if p.requires_grad)
            result['train'] = train(model, train_rows, schedule, pad_id, .003, 8, output, dev, prior,
                                    forward_fn=forward, curve_evaluator=curve_evaluator)
            torch.cuda.synchronize()
            tick = time.monotonic()
            result['evaluation'], original_records = evaluate_all(model, output/'evaluation')
            torch.cuda.synchronize()
            result['evaluation_seconds'] = time.monotonic()-tick
            result['gate'] = gate(result['evaluation']['isolated'])
            result['gate_by_target_position'] = {
                name: gate({m: result['evaluation']['isolated'][m]['position_strata'][name]
                            for m in ('normal', 'zero', 'shuffled')})
                for name in ('position0', 'position1plus')
                if all(name in result['evaluation']['isolated'][m]['position_strata'] for m in ('normal', 'zero', 'shuffled'))
                and result['evaluation']['isolated']['normal']['position_strata'][name]['diagnostics']
                .get('address.target_weight') is not None}
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
        artifacts = [{'path': str(p.resolve()), 'role': str(p.relative_to(output)), **file_record(p)}
                     for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']
        gate_result = result.get('gate')
        write_json(output/'manifest.json', {
            'status': 'complete' if passed else 'failed', 'experiment': args.experiment,
            'contract': EXPERIMENTS[args.experiment],
            'arm': args.arm, 'write_protocol': 'isolated', 'spec_digest': os.environ.get('SR_SPEC_DIGEST'),
            'model': MODEL, 'revision': REVISION, 'smoke_steps': args.smoke_steps,
            'smoke_eval_limit': args.smoke_eval_limit,
            'eligible_as_evidence': passed and not bool(args.smoke_steps),
            'seed': args.seed, 'start_seed': args.r1_seed if args.arm == 'D5' else None,
            'lr': .003, 'batch_size': 8, 'microbatch_size': 8,
            'data_dir': str(args.data_dir.resolve()), 'data_sha256': DATA_HASHES,
            'resume_provenance': provenance, 'schedule': result.get('schedule'),
            'requires_grad_parameter_names': result.get('requires_grad_parameter_names'),
            'checkpoint_files': result.get('checkpoint_files'), 'torch': torch.__version__,
            'hardware': {'device': torch.cuda.get_device_name(0),
                         'multiprocessors': torch.cuda.get_device_properties(0).multi_processor_count},
            'cuda': torch.version.cuda,
            'metrics': {'wall_seconds': wall, 'mig_instance_hours': wall/3600, 'node_hours': wall/14400,
                        **({'h3_isolated': gate_result} if gate_result else {})},
            'artifacts': artifacts, 'failure': failure,
            'validator': {'name': 'v2r16_isolated', 'passed': passed}})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS), default='E-0016')
    parser.add_argument('--r1-run', type=Path, default=R1_RUN, help='D5 start: a run of the R1 recipe')
    parser.add_argument('--r1-seed', type=int, default=0, help='the seed the start run must have recorded')
    parser.add_argument('--r1-adapter-sha256', default=None,
                        help='expected sha256 of the start adapter.pt (required unless --r1-run is ATT-0122-1)')
    parser.add_argument('--seed', type=int, default=0, help='D5 continuation order seed')
    parser.add_argument('--step-budget', type=int, default=6000)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--smoke-eval-limit', type=int, default=0)
    args = parser.parse_args()
    if args.step_budget != 6000:
        parser.error('E-0016 uses --step-budget 6000; shorten only with --smoke-steps')
    if not 0 <= args.smoke_steps <= args.step_budget:
        parser.error('smoke steps must be 0..6000')
    if args.smoke_eval_limit and (not args.smoke_steps or args.smoke_eval_limit < 2):
        parser.error('smoke eval limit >=2 requires --smoke-steps')
    if args.r1_adapter_sha256 is None:
        if args.arm == 'D5' and args.r1_run != R1_RUN:
            parser.error('--r1-adapter-sha256 is required for a start run other than ATT-0122-1')
        args.r1_adapter_sha256 = R1_ADAPTER_SHA256
    if args.experiment == 'E-0016' and (args.seed or args.r1_seed or args.r1_run != R1_RUN):
        parser.error('E-0016 is D4/D5 at seed 0 from ATT-0122-1')
    if args.experiment == 'E-0017' and (args.arm != 'D5' or args.seed not in (1, 2) or args.r1_seed not in (0, args.seed)):
        parser.error('E-0017 runs D5 with continuation seed 1 or 2 from R1 at seed 0 or the same seed')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
