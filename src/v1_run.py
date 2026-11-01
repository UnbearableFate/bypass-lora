"""Shared V1 train/evaluate entry point. Execute only in a PBS GPU allocation."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from calibrate import MODEL, REVISION, load_base
from bypass_model import AdapterConfig, BranchModel
from v1_lora import LoRAConfig, LoRAModel
from v1_data import (read_piqa_train, read_boolq_train, make_v1_splits,
                     build_training_stream, encode_candidates, encode_training_example,
                     collate_encoded, derive_seed)

SOURCES = {
    'piqa': {'repo': 'ybisk/ybisk.github.io', 'revision': '21edab439af693b961be2f069e8690a88d3b4e37'},
    'boolq': {'repo': 'google/boolq', 'revision': '35b264d03638db9f4ce671b711558bf7ff0f80d5'},
}
EXPECTED = {'lora': 901120, 'lora28': 1261568, 'F_out': 262144, 'F_pre': 262144,
            'C_gate': 524288, 'C_up': 524288, 'C_dual': 917504, 'linear': 917504}


class TrainingDiverged(Exception):
    """A scientific failed-LR outcome, distinct from an engineering exception."""
    def __init__(self, reason, metrics):
        super().__init__(reason)
        self.metrics = metrics


def example_mean_objective(losses, step_examples):
    return losses.mean() / step_examples


def prepare(data_dir):
    p = Path(data_dir)
    files = {'piqa': p/'piqa-train.jsonl', 'piqa_labels': p/'piqa-train-labels.lst',
             'boolq': p/'boolq-train.parquet'}
    expected_hashes = {'piqa':'f805f9c96518d7b59f2f870d90f712457cda949979469d9fa3fd706ce22e0a1a',
                      'piqa_labels':'df8ae080c41f02f88e81f6932d425f4b7ee72f433ad552e6a30d45e358e8f8bb',
                      'boolq':'4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623'}
    for key, path in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[key]:
            raise ValueError(f'source checksum mismatch: {key}')
    result = make_v1_splits(read_piqa_train(files['piqa'], files['piqa_labels']),
                          read_boolq_train(files['boolq']), source_files=files, source_metadata=SOURCES)
    if result['metadata']['split_mapping_sha256'] != '47cf5c777d07863b8def6903487697eafc970b0cd5b5d80babb7f20ded98571b':
        raise ValueError('split mapping differs from locked smoke mapping')
    return result


def make_model(arm, checkpoint=None):
    base = load_base()
    # load_base uses its own fixed diagnostic seed; reset before initializing new parameters.
    torch.manual_seed(derive_seed(20261029, 'concat_loop_v2', 'V1', 'init', 0))
    if arm == 'base':
        return base
    if arm in ('lora', 'lora28'):
        model = LoRAModel.load_adapter(base, checkpoint) if checkpoint else LoRAModel(
            base, LoRAConfig(layers=tuple(range(28)) if arm == 'lora28' else (6,13,20,27),
                             rank=2 if arm == 'lora28' else 10))
    else:
        model = BranchModel.load_adapter(base, checkpoint) if checkpoint else BranchModel(
            base, AdapterConfig(branch='tokenwise_linear' if arm == 'linear' else 'tokenwise_silu',
                                fusion='C_dual' if arm == 'linear' else arm))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == EXPECTED[arm]
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    return model


def batch_tensors(encoded, pad_id):
    return {k: torch.tensor(v, device='cuda', dtype=torch.long)
            for k,v in collate_encoded(encoded, pad_id).items()}


def token_losses(model, batch):
    labels = batch['labels'][:, 1:]
    valid = labels != -100
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
    # FP32 likelihoods only for supervised positions, avoiding full-vocabulary FP32 prompt copies.
    losses = F.cross_entropy(logits[:, :-1][valid].float(), labels[valid], reduction='none')
    return losses, valid


def lr_factor(step, total):
    warm = max(1, math.ceil(total * .05))
    if step < warm:
        return (step + 1) / warm
    return .5 * (1 + math.cos(math.pi * (step-warm+1) / max(1,total-warm)))


def plan_batches(splits, tok, budget, batch_size, max_epochs=3):
    batches, pending, tokens = [], [], 0
    for epoch in range(max_epochs):
        for row in build_training_stream(splits, epoch):
            encoded = encode_training_example(tok, row)
            encoded['epoch'] = epoch
            pending.append(encoded)
            if len(pending) == batch_size:
                batches.append(pending)
                tokens += sum(len(e['input_ids']) for e in pending)
                pending = []
                if tokens >= budget:
                    return batches
        # Preserve full batches over epoch boundaries, then flush only at the epoch cap.
    if pending:
        batches.append(pending)
    return batches


@torch.no_grad()
def evaluate(model, splits, tok, limit=None):
    model.eval()
    metrics, predictions = {}, []
    for split in ('calibration', 'dev'):
        task_nll = []
        for task in ('piqa', 'boolq'):
            total, count, correct = 0., 0, 0
            rows = splits['splits'][task][split]
            if limit:
                rows = rows[:limit]
            for row in rows:
                candidates = encode_candidates(tok, row)
                scores, sums, counts = [], [], []
                for candidate in candidates:
                    loss, _ = token_losses(model, batch_tensors([candidate], tok.pad_token_id))
                    sums.append(loss.sum().item())
                    counts.append(loss.numel())
                    # PIQA length normalization; BoolQ joint label probability (sum).
                    scores.append(-sums[-1] / (counts[-1] if task == 'piqa' else 1))
                gold = int(row['label']) if task == 'piqa' else (0 if row['label'] else 1)
                pred = 0 if scores[0] >= scores[1] else 1
                correct += pred == gold
                total += sums[gold]
                count += counts[gold]
                predictions.append({'task':task, 'split':split, 'id':row['id'], 'gold':gold,
                                    'prediction':pred, 'scores':scores, 'gold_nll_sum':sums[gold],
                                    'gold_loss_tokens':counts[gold]})
            metrics[f'{split}.{task}.nll'] = total/count
            metrics[f'{split}.{task}.accuracy'] = correct/len(rows)
            metrics[f'{split}.{task}.examples'] = len(rows)
            metrics[f'{split}.{task}.loss_tokens'] = count
            task_nll.append(total/count)
        metrics[f'{split}.nll'] = sum(task_nll)/2
    return metrics, predictions


def train(model, batches, tok, lr, output):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(.9,.999), eps=1e-8, weight_decay=.01)
    totals = {'train.input_tokens':0, 'train.loss_tokens':0, 'train.padded_tokens':0, 'train.examples':0}
    step_times = []
    start = time.monotonic()
    with (output/'steps.jsonl').open('w') as log:
        for step, examples in enumerate(batches):
            torch.cuda.synchronize()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            # Each example contributes equally, regardless of its completion length.
            summed = 0.
            for e in examples:
                loss, _ = token_losses(model, batch_tensors([e], tok.pad_token_id))
                totals['train.input_tokens'] += len(e['input_ids'])
                totals['train.loss_tokens'] += loss.numel()
                totals['train.examples'] += 1
                (output/'usage.json').write_text(json.dumps(totals)+'\n')
                if not torch.isfinite(loss).all():
                    raise TrainingDiverged('nonfinite loss', dict(totals))
                example_mean_objective(loss, len(examples)).backward()
                summed += loss.detach().mean().item()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=False).item()
            if not math.isfinite(norm):
                raise TrainingDiverged('nonfinite gradient norm', dict(totals))
            rate = lr * lr_factor(step, len(batches))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.monotonic()-tick
            step_times.append(duration)
            tokens = sum(len(e['input_ids']) for e in examples)
            log.write(json.dumps({'step':step+1,'example_mean_nll':summed/len(examples),'preclip_norm':norm,
                                  'lr':rate,'seconds':duration,'input_tokens':tokens})+'\n')
            log.flush()
    totals.update({'train.steps':len(batches),'train.seconds':time.monotonic()-start,
                   'train.steady_step_seconds':sum(step_times[10:])/len(step_times[10:]) if len(step_times)>10 else None,
                   'train.last_epoch':batches[-1][-1]['epoch'],
                   'train.peak_cuda_bytes':torch.cuda.max_memory_allocated()})
    totals['train.tokens_per_second'] = totals['train.input_tokens']/totals['train.seconds']
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['base', *EXPECTED], required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tokens', type=int, default=200000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--eval-limit', type=int)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    passed = False
    start = time.monotonic()
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        tok.pad_token_id = tok.eos_token_id
        splits = prepare(args.data_dir)
        mapping = json.dumps(splits['metadata']['split_ids'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
        (out/'split-map.json').write_text(mapping)
        (out/'data-metadata.json').write_text(json.dumps(splits['metadata'],indent=2)+'\n')
        (out/'config.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
        batches = plan_batches(splits,tok,args.tokens,args.batch)
        metrics['plan.steps'] = len(batches)
        metrics['plan.input_tokens'] = sum(len(e['input_ids']) for batch in batches for e in batch)
        metrics['plan.token_threshold'] = args.tokens
        metrics['plan.token_overshoot'] = max(0, metrics['plan.input_tokens']-args.tokens)
        metrics['plan.order_sha256'] = hashlib.sha256(json.dumps([(e['task'],e['id'],e['epoch']) for batch in batches for e in batch]).encode()).hexdigest()
        if args.smoke_steps:
            batches = batches[:args.smoke_steps]
        elif args.arm != 'base':
            assert len(batches) >= 100
        tick = time.monotonic()
        model = make_model(args.arm)
        torch.cuda.synchronize()
        metrics['load.seconds'] = time.monotonic()-tick
        if args.arm != 'base':
            metrics.update(train(model,batches,tok,args.lr,out))
            metrics['train.token_overshoot'] = max(0, metrics['train.input_tokens']-args.tokens)
            tick = time.monotonic()
            model.export_adapter(out/'adapter')
            metrics['save.seconds'] = time.monotonic()-tick
        tick = time.monotonic()
        eval_metrics, predictions = evaluate(model,splits,tok,args.eval_limit)
        metrics.update(eval_metrics)
        metrics['eval.seconds'] = time.monotonic()-tick
        (out/'predictions.json').write_text(json.dumps(predictions)+'\n')
        # Checkpoint roundtrip re-evaluates identical rows and scores on a fresh Base.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        tick = time.monotonic()
        model = make_model(args.arm, None if args.arm=='base' else out/'adapter')
        repeat_metrics, repeat = evaluate(model,splits,tok,args.eval_limit)
        metrics['reload.seconds'] = time.monotonic()-tick
        metrics['reload.identical'] = repeat == predictions and repeat_metrics == eval_metrics
        assert metrics['reload.identical'], 'checkpoint/repeated Base evaluation is not identical'
        passed = True
    except TrainingDiverged as exc:
        metrics.update(exc.metrics)
        metrics['train.diverged'] = True
        metrics['train.failure_reason'] = str(exc)
        metrics['train.token_overshoot'] = max(0, metrics['train.input_tokens']-args.tokens)
        # Record a complete failed-LR result and let a caller continue other arms.
        passed = True
    except Exception as exc:
        metrics['error'] = repr(exc)
        passed = False
        raise
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':metrics,
            'artifacts':[{'path':str(p),'role':p.stem} for p in out.glob('*.json') if p.name!='manifest.json'],
            'validator':{'name':'v1_runtime','passed':passed}},indent=2)+'\n')

if __name__ == '__main__':
    main()
