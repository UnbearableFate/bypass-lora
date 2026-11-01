"""Offline RB read distributions; the actual shared LatentReader is observed.

These diagnostics never substitute a read or change its epsilon denominator.
As in v2_memory.read_metrics, probability is renormalized to sum to one before
computing entropy and effective item count. Empty reads have all-zero metrics.
"""
import torch


# Per-position vectors carried in read records; ReadSummary never averages them.
VECTOR_FIELDS = ('read_vector',)

@torch.no_grad()
def prefix_read_metrics(query, keys, valid, *, kind=None, donor=None, window=None,
                        chunk_size=64):
    batch, length, rank = query.shape
    if keys.shape != query.shape or valid.shape != (batch, length):
        raise ValueError('read diagnostic feature/mask shape mismatch')
    if chunk_size < 1:
        raise ValueError('read diagnostic chunk size must be positive')
    if kind not in (None, 'iv_window', 'iv_window1', 'iv_window8', 'iv_strict',
                    'iv_swap', 'iv_permute'):
        raise ValueError('no read distribution is defined for this intervention')
    if kind and kind.startswith('iv_window'):
        window = window if kind == 'iv_window' else int(kind[9:])
        if type(window) is not int or window < 1:
            raise ValueError('history window must be positive')
    query = query.detach().float()
    keys = keys.detach().float()
    valid = valid.bool()
    if kind == 'iv_swap':
        if donor is None:
            raise ValueError('swap diagnostics require donor keys and valid mask')
        history = donor['keys'].detach().to(query.device).float()
        history_valid = donor['valid'].to(query.device).bool()
        if history.shape[:2] != history_valid.shape or history.shape[0] != batch or history.shape[2] != rank:
            raise ValueError('swap diagnostic donor shape mismatch')
    else:
        history, history_valid = keys, valid
    positions = torch.arange(history.shape[1], device=query.device)
    pieces = {name: [] for name in ('current_weight', 'n_eff', 'entropy',
                                   'relative_n_eff', 'readable_positions',
                                   'share0', 'share0_2', 'argmax0', 'uniform0', 'uniform0_2')}
    for start in range(0, length, chunk_size):
        end = min(length, start+chunk_size)
        current = torch.arange(start, end, device=query.device)
        eligible = history_valid[:, None] & (positions[None, None] <= current[None, :, None])
        if kind in ('iv_strict', 'iv_swap'):
            eligible = eligible & (positions[None, None] < current[None, :, None])
        if kind and kind.startswith('iv_window'):
            eligible = eligible & (positions[None, None] > current[None, :, None]-window)
        scores = torch.einsum('btr,bnr->btn', query[:, start:end], history)
        scores = torch.where(eligible, scores, torch.zeros_like(scores))
        if kind == 'iv_swap':
            # Donor past plus recipient current; no donor future, even when the
            # donor is longer. A short donor contributes its entire valid past.
            own = (query[:, start:end]*keys[:, start:end]).sum(-1)
            own = torch.where(valid[:, start:end], own, torch.zeros_like(own))
            scores = torch.cat((scores, own[..., None]), -1)
            readable = eligible.sum(-1) + valid[:, start:end].long()
            own_score = own
            slots = torch.cat((eligible, valid[:, start:end, None]), -1)
        else:
            own_score = (query[:, start:end]*keys[:, start:end]).sum(-1)
            own_score = torch.where(valid[:, start:end] & (kind != 'iv_strict'), own_score,
                                    torch.zeros_like(own_score))
            readable = eligible.sum(-1)
            slots = eligible
        total = scores.sum(-1)
        probability = scores / total[..., None].clamp_min(1e-30)
        active = valid[:, start:end] & (readable > 0)
        neff = torch.where(active, 1/probability.square().sum(-1).clamp_min(1e-30), 0)
        entropy = -(probability*probability.clamp_min(1e-30).log()).sum(-1)
        values = {'current_weight': own_score/total.clamp_min(1e-30),
                  'n_eff': neff, 'entropy': entropy,
                  'relative_n_eff': neff/readable.clamp_min(1),
                  'readable_positions': readable.float(),
                  # History index 0 is the sequence's first token (the donor's
                  # first token under iv_swap). Uniform shares use the same slots.
                  'share0': probability[..., 0],
                  'share0_2': probability[..., :3].sum(-1),
                  'argmax0': (probability.argmax(-1) == 0).float(),
                  'uniform0': slots[..., 0].float()/readable.clamp_min(1),
                  'uniform0_2': slots[..., :3].sum(-1).float()/readable.clamp_min(1)}
        for name, value in values.items():
            pieces[name].append(torch.where(active, value, torch.zeros_like(value)))
    return {name: torch.cat(value, 1) for name, value in pieces.items()}


class ReadSummary:
    """Token-weighted means, per task/layer/read step and query-position scope.

    Only the gold candidate is aggregated, after scoring both candidates with
    identical model information. Answer scope refers to positions predicting
    answer tokens, including the final prompt position predicting its first token.
    """
    def __init__(self):
        self.totals = {}

    def add(self, task, diagnostics, attention_mask, labels):
        valid = attention_mask.bool()
        answer = torch.zeros_like(valid)
        answer[:, :-1] = labels[:, 1:] != -100
        for layer, diagnostic in (diagnostics or {}).items():
            for record in diagnostic.get('read_metrics', []):
                for scope, mask in (('all_valid_input', valid), ('answer_prediction', valid & answer)):
                    key = (task, layer, record['step'], scope)
                    total = self.totals.setdefault(key, {'count': 0, 'empty_reads': 0, 'sums': {}})
                    count = int(mask.sum())
                    total['count'] += count
                    total['empty_reads'] += int((mask & (record['readable_positions'] == 0)).sum())
                    for name, value in record.items():
                        if name not in ('step', *VECTOR_FIELDS):
                            total['sums'][name] = total['sums'].get(name, 0.) + float(value[mask].double().sum())

    def finish(self):
        summaries = []
        for (task, layer, step, scope), value in sorted(self.totals.items()):
            nonempty_count = value['count'] - value['empty_reads']
            summaries.append({
                'task': task, 'layer': layer, 'step': step, 'scope': scope,
                'count': value['count'], 'empty_reads': value['empty_reads'],
                # Keep the original means for report compatibility. Empty reads
                # stay zero here, including for current_weight.
                'means': {name: total/value['count'] if value['count'] else None
                          for name, total in value['sums'].items()},
                'nonempty_means': {
                    name: total/nonempty_count if nonempty_count else None
                    for name, total in value['sums'].items()
                },
            })
        return summaries
