"""Shared V2 data verification, deterministic sample order and optimizer schedule."""
import hashlib
import json
import math
from pathlib import Path
import random

from v1_data import derive_seed

DATA_HASHES = {
    'train': 'a2acb5451bc8aade1c274ebe225b4411bb3063e81bf57ba73c35101d9ac807d2',
    'calibration': 'c08753365165216b5a9f4c3d629271b2183e1323973886256867e826c1712f18',
    'dev': 'c6583cf8f7cd9d5652ed57093432b278b0600c611076fac2fe6c6b7d9e7715f4',
}
DATA_COUNTS = {'train': 16384, 'calibration': 2048, 'dev': 1000}


def load_data(path, split):
    if split not in DATA_HASHES:
        raise ValueError('only train/calibration/dev are authorized')
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATA_HASHES[split]:
        raise ValueError(f'{split} data hash mismatch')
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(rows) != DATA_COUNTS[split] or any(r['identity']['split'] != split for r in rows):
        raise ValueError('data split/count mismatch')
    if len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate episode identity')
    return rows


ORDER_HASH_ENCODING = 'UTF-8 episode_id followed by LF, including final LF, three epochs'


def _cost(row):
    return len(row['support_input_ids']) + len(row['question_input_ids'])


def train_order(rows, phase='V2'):
    if phase not in ('V2', 'V3'):
        raise ValueError('unsupported data phase')
    if not rows or len(rows) % 8:
        raise ValueError('complete batches of eight required')
    order = []
    for epoch in range(3):
        indices = list(range(len(rows)))
        random.Random(derive_seed(20261029, 'concat_loop_v2', phase, f'train-order-{epoch}')).shuffle(indices)
        order.extend(indices)
    digest = hashlib.sha256(''.join(rows[i]['episode_id']+'\n' for i in order).encode('utf-8')).hexdigest()
    return order, digest


def make_schedule(rows, budget, *, phase='V2'):
    if phase not in ('V2', 'V3'):
        raise ValueError('unsupported data phase')
    if budget <= 0 or not rows or len(rows) % 8:
        raise ValueError('positive budget and complete batches of eight required')
    order, digest = train_order(rows, phase)
    batches, tokens = [], 0
    for offset in range(0, len(order), 8):
        batch = order[offset:offset+8]
        batches.append(batch)
        tokens += sum(_cost(rows[i]) for i in batch)
        if tokens >= budget:
            break
    return {'order': order, 'batches': batches, 'steps': len(batches), 'tokens': tokens,
            'warmup_steps': math.ceil(.05*len(batches)), 'order_sha256': digest,
            'budget': budget, 'order_hash_encoding': ORDER_HASH_ENCODING}


def make_step_schedule(rows, steps, *, phase='V2'):
    """The first ``steps`` batches of the same three-epoch order (E-0014 step budget).

    On rows where the token budget stops at step N, steps=N gives the same batches.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError('positive integer step budget required')
    order, digest = train_order(rows, phase)
    if steps * 8 > len(order):
        raise ValueError('step budget exceeds three epochs')
    batches = [order[offset:offset+8] for offset in range(0, steps*8, 8)]
    return {'order': order, 'batches': batches, 'steps': steps,
            'tokens': sum(_cost(rows[i]) for batch in batches for i in batch),
            'warmup_steps': math.ceil(.05*steps), 'order_sha256': digest,
            'step_budget': steps, 'order_hash_encoding': ORDER_HASH_ENCODING}


def make_curriculum_schedule(segments):
    """One schedule over concatenated row sets: ``segments`` is [(label, rows, steps)].

    Segment k uses the first ``steps`` batches of its own rows' step schedule, so
    steps are contiguous in segment order. Batch indices refer to the returned
    concatenated ``rows``; warmup/cosine spans the total step count.
    """
    if not segments:
        raise ValueError('empty curriculum')
    rows, batches, parts, offset, start = [], [], [], 0, 0
    for label, segment_rows, steps in segments:
        plan = make_step_schedule(segment_rows, steps)
        batches.extend([[offset + i for i in batch] for batch in plan['batches']])
        parts.append({'label': label, 'first_step': start + 1, 'last_step': start + steps,
                      'steps': steps, 'tokens': plan['tokens'], 'order_sha256': plan['order_sha256'],
                      'row_offset': offset, 'rows': len(segment_rows)})
        rows.extend(segment_rows)
        offset += len(segment_rows)
        start += steps
    if len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('curriculum row sets share episode identities')
    digest = hashlib.sha256(''.join(rows[i]['episode_id']+'\n' for batch in batches for i in batch)
                            .encode('utf-8')).hexdigest()
    return rows, {'batches': batches, 'steps': start, 'tokens': sum(p['tokens'] for p in parts),
                  'warmup_steps': math.ceil(.05*start), 'step_budget': start,
                  'curriculum': parts, 'order_sha256': digest,
                  'order_hash_encoding': 'UTF-8 episode_id followed by LF for every scheduled example'}


def lr_multiplier(step, total, warmup):
    """Zero-based step; first update 1/warmup, warmup endpoint 1, last 0."""
    if not 0 <= step < total or not 1 <= warmup <= total:
        raise ValueError('invalid schedule position')
    completed = step + 1
    if completed <= warmup:
        return completed / warmup
    return .5 * (1 + math.cos(math.pi * (completed-warmup)/(total-warmup)))


def select_learning_rate(candidates):
    """Minimum finite NLL, preferring smaller LR within 0.5% of that minimum."""
    valid = [r for r in candidates if math.isfinite(r['nll']) and r.get('finite_loss', True)]
    if not valid:
        raise ValueError('all learning rates failed')
    best = min(r['nll'] for r in valid)
    return min((r for r in valid if r['nll']-best < .005*abs(best) or r['nll'] == best), key=lambda r: r['lr'])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = load_data(args.train, 'train')
    report = {str(b): {k: v for k, v in make_schedule(rows, b).items() if k not in ('order', 'batches')}
              for b in (200000, 4000000)}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
