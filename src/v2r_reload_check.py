"""E-0015 reload diagnostic (MSG-0402): redo the E=16 dev reload check of V2-03 and R1
in one process, with the recorded evaluator settings, and log device and batching.

Order: V2-03, R1, V2-03 again, then the Base model alone (the zero control does not
depend on the adapter). Run in an allocated GPU job; never on a login node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION, load_base
from v2_memory import collate
from v2_schedule import load_data
from v2_train import EVAL_BATCH_SIZE, adapted, evaluate, write_json
from v2r_diagnostics import ARM, CHECKPOINTS, compare_recorded

CONTROLS = ('normal', 'zero')


def device_info():
    info = {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'device_name': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0),
            'multi_processor_count': torch.cuda.get_device_properties(0).multi_processor_count,
            'total_memory': torch.cuda.get_device_properties(0).total_memory,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'deterministic': torch.are_deterministic_algorithms_enabled()}
    try:
        info['nvidia_smi_L'] = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # diagnostic only
        info['nvidia_smi_L'] = repr(exc)
    return info


@torch.no_grad()
def base_zero(base, rows, pad_id, device):
    """Base-only answer logits per episode, batched exactly as evaluate() batches."""
    predictions, nll = [], []
    for start in range(0, len(rows), EVAL_BATCH_SIZE):
        batch = collate(rows[start:start + EVAL_BATCH_SIZE], pad_id, device)
        mask = batch.question_mask
        logits = base(batch.question_ids, attention_mask=mask,
                      position_ids=(mask.long().cumsum(-1) - 1).clamp_min(0), use_cache=False).logits
        last = mask.long().sum(-1) - 1
        logits = logits[torch.arange(len(last), device=device), last].float()
        predictions += logits.argmax(-1).tolist()
        nll += F.cross_entropy(logits, batch.labels, reduction='none').tolist()
    return predictions, nll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('/work/xg24i002/x10041/bypass-lora/runs'))
    parser.add_argument('--e16-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    passed, failure, report = False, None, {}
    try:
        rows = load_data(args.e16_data / 'dev.jsonl', 'dev')
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for row in rows:
            row['_subject_token_id'] = tok.encode(' ' + row['query']['head'], add_special_tokens=False)[0]
        order = ''.join(r['episode_id'] + '\n' for r in rows).encode()
        lengths = [max(len(r['question_input_ids']) for r in rows[i:i + EVAL_BATCH_SIZE])
                   for i in range(0, len(rows), EVAL_BATCH_SIZE)]
        report['settings'] = {'eval_batch_size': EVAL_BATCH_SIZE, 'padding': 'right (v2_memory.pad), pad_id=%d' % pad_id,
                              'episode_order_sha256': hashlib.sha256(order).hexdigest(), 'episodes': len(rows),
                              'question_batch_max_lengths': lengths,
                              'support_batch_max_lengths': [max(len(r['support_input_ids']) for r in rows[i:i + EVAL_BATCH_SIZE])
                                                            for i in range(0, len(rows), EVAL_BATCH_SIZE)]}
        base = load_base()
        report['device'] = device_info()
        device = next(base.parameters()).device
        report['checks'] = []
        for step, label in enumerate(('V2-03', 'R1', 'V2-03')):
            arm_dir = args.runs / CHECKPOINTS[label] / ARM
            out = output / f'{step}-{label}'
            with adapted(base, checkpoint=arm_dir / 'adapter') as model:
                model.eval()
                evaluate(model, rows, pad_id, out, 'dev', CONTROLS, None)
            report['checks'].append({'step': step, 'checkpoint': label,
                                     'comparison': compare_recorded(arm_dir, out, 'dev', CONTROLS)})
            write_json(output / 'report.json', report)
        predictions, nll = base_zero(base, rows, pad_id, device)
        for label in ('V2-03', 'R1'):
            recorded = [json.loads(x) for x in (args.runs / CHECKPOINTS[label] / ARM / 'dev.zero.predictions.jsonl')
                        .read_text().splitlines() if x.strip()]
            report.setdefault('base_only_vs_recorded_zero', {})[label] = {
                'prediction_mismatches': sum(p != r['prediction'] for p, r in zip(predictions, recorded)),
                'nll_max_abs_difference': max(abs(a - r['nll']) for a, r in zip(nll, recorded))}
        passed = True
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        write_json(output / 'report.json', report)
        artifacts = [{'path': str(p), 'role': str(p.relative_to(output))} for p in output.rglob('*')
                     if p.is_file() and p.name != 'manifest.json']
        write_json(output / 'manifest.json', {
            'status': 'complete' if passed else 'failed', 'spec_digest': os.environ.get('SR_SPEC_DIGEST'),
            'experiment': 'E-0015@4', 'purpose': 'V2-03 reload diagnostic (MSG-0402)',
            'metrics': {'checks': [{'step': c['step'], 'checkpoint': c['checkpoint'],
                                    **{m: r['prediction_mismatches'] for m, r in c['comparison'].items()}}
                                   for c in report.get('checks', [])],
                        'base_only_vs_recorded_zero': report.get('base_only_vs_recorded_zero')},
            'failure': failure, 'artifacts': artifacts,
            'validator': {'name': 'v2r_reload_check', 'passed': passed}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
