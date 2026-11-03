"""Segmented candidate scoring with the unchanged V1 loss and tie conventions."""
import math
import time

import torch
import torch.nn.functional as F

from rb_history import donor_indices
from seg_data import encode_segments, segment1_write
from seg_model import forward, first_forward, WRITE_RULE
from v1_run import batch_tensors, EvaluationNonfinite


def tensors(encoded, pad_id):
    second = batch_tensors([encoded], pad_id)
    first = torch.tensor([encoded['segment1_ids']], device=second['input_ids'].device)
    return first, torch.ones_like(first), second


def arm_write(tok, arm, ids):
    """The arm's registered segment-1 write mask (E-0032), or None (all tokens write)."""
    return segment1_write(tok, ids, WRITE_RULE.get(arm, 'all'))


def write_tensor(first_write, first):
    if first_write is None:
        return None
    mask = torch.tensor([first_write], dtype=torch.bool, device=first.device)
    if mask.shape != first.shape:
        raise ValueError('segment-1 write mask length differs from segment 1')
    return mask


def token_losses(model, arm, encoded, pad_id, kind='normal', donor_memory=None, force_first=False,
                 timing_events=None, first_write=None):
    first, mask, second = tensors(encoded, pad_id)
    output, parts = forward(model, arm, first, mask, second['input_ids'], second['attention_mask'],
                            kind=kind, donor_memory=donor_memory, force_first=force_first,
                            timing_events=timing_events, first_write=write_tensor(first_write, first))
    labels = second['labels'][:, 1:]
    valid = labels != -100
    losses = F.cross_entropy(output.logits[:, :-1][valid].float(), labels[valid], reduction='none')
    if not losses.numel() or not torch.isfinite(losses).all():
        raise EvaluationNonfinite('segmented candidate has nonfinite/empty answer loss')
    return losses, parts, output


@torch.no_grad()
def evaluate(model, arm, splits, tok, limit=None, kind='normal', split_name='dev', donor_phase='R1'):
    model.eval()
    if split_name not in ('dev', 'a2'):
        raise ValueError('unknown segmented evaluation split')
    rows = [r for task in ('piqa', 'boolq') for r in splits['splits'][task][split_name][:limit]]
    donors = donor_indices(rows, donor_phase) if kind == 'iv_swap' else None
    metrics, predictions, audit, donor_map = {}, [], [], []
    start = time.monotonic()
    segment_seconds = {'segment1': 0., 'segment2': 0.}
    donor_seconds = 0.
    writes = [] if arm in WRITE_RULE else None
    for ri, row in enumerate(rows):
        encoded = encode_segments(tok, row)
        audit.append(encoded[0]['seam'])
        first_write = arm_write(tok, arm, encoded[0]['segment1_ids'])
        if writes is not None:
            writes.append(sum(first_write))
        donor_memory = None
        if donors is not None:
            donor = rows[donors[ri]]
            donor_encoded = encode_segments(tok, donor)[0]
            ids, mask, _ = tensors(donor_encoded, tok.pad_token_id)
            donor_write = write_tensor(arm_write(tok, arm, donor_encoded['segment1_ids']), ids)
            torch.cuda.synchronize(); donor_tick = time.monotonic()
            donor_memory = first_forward(model, ids, mask, donor_write).cache.memory
            torch.cuda.synchronize(); donor_seconds += time.monotonic()-donor_tick
            donor_map.append({'task': row['task'], 'recipient': row['id'], 'donor': donor['id']})
        scores, sums, counts = [], [], []
        for candidate in encoded:
            events = []
            losses, _, _ = token_losses(model, arm, candidate, tok.pad_token_id, kind, donor_memory,
                                        timing_events=events, first_write=first_write)
            total = losses.sum().item()
            for phase, begin, end in events:
                segment_seconds[phase] += begin.elapsed_time(end)/1000
            sums.append(total); counts.append(losses.numel())
            scores.append(-total/(counts[-1] if row['task'] == 'piqa' else 1))
        gold = int(row['label']) if row['task'] == 'piqa' else (0 if row['label'] else 1)
        prediction = 0 if scores[0] >= scores[1] else 1
        predictions.append({'task': row['task'], 'split': split_name, 'id': row['id'], 'gold': gold,
                            'prediction': prediction, 'scores': scores,
                            'gold_nll_sum': sums[gold], 'gold_loss_tokens': counts[gold]})
    for task in ('piqa', 'boolq'):
        selected = [r for r in predictions if r['task'] == task]
        if not selected:
            raise ValueError('both tasks need nonempty dev rows')
        nll = sum(r['gold_nll_sum'] for r in selected)
        tokens = sum(r['gold_loss_tokens'] for r in selected)
        metrics.update({f'{split_name}.{task}.accuracy': sum(r['prediction'] == r['gold'] for r in selected)/len(selected),
                        f'{split_name}.{task}.examples': len(selected), f'{split_name}.{task}.nll': nll/tokens,
                        f'{split_name}.{task}.loss_tokens': tokens})
    metrics[f'{split_name}.nll'] = sum(metrics[f'{split_name}.{t}.nll'] for t in ('piqa', 'boolq'))/2
    if not all(math.isfinite(float(x)) for x in metrics.values()):
        raise EvaluationNonfinite('nonfinite segmented evaluation aggregate')
    return metrics, predictions, {'elapsed_s': time.monotonic()-start, 'seams': audit, 'donors': donor_map,
                                 'forward_seconds': segment_seconds, 'donor_forward_seconds': donor_seconds,
                                 **({} if writes is None else {'segment1_writes': writes})}
