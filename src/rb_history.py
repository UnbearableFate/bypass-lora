"""Evaluation-only F-line side-history interventions; backbone KV is untouched."""
import random

import torch

from bypass_core import (MemoryState, SequenceResult, kernel_prefix_states,
                         read_kernel_prefix)
from v1_data import derive_seed


KINDS = ("iv_off", "iv_window1", "iv_window8", "iv_window", "iv_strict", "iv_swap", "iv_permute")


def donor_indices(rows, phase):
    """Uniform rejection-sampled derangement separately within each task."""
    result = list(range(len(rows)))
    for task in sorted({row['task'] for row in rows}):
        positions = [i for i, row in enumerate(rows) if row['task'] == task]
        if len(positions) < 2:
            raise ValueError('iv_swap needs at least two examples per task')
        donors = positions.copy()
        rng = random.Random(derive_seed(20261101, 'residual_la_v21', phase, 'iv_swap'))
        while True:
            rng.shuffle(donors)
            if all(a != b for a, b in zip(positions, donors)):
                break
        for recipient, donor in zip(positions, donors):
            result[recipient] = donor
    return result


def run_history(q, k, v, reader, valid, config, options):
    """Full-sequence diagnostic path. Interventions never accept cached state.

    The donor contains projected positive keys and raw values from a normal
    forward of that donor, at this same adapter layer. Histories use physical
    sequence positions; invalid writes contribute zero.
    """
    kind = options.get('kind')
    if kind is not None and kind not in KINDS:
        raise ValueError('unknown F-line history intervention')
    keys, values, matrices, masses, counts, final = kernel_prefix_states(k, v, reader, valid)
    normal_m, normal_c, normal_n = matrices, masses, counts
    zero_m = torch.zeros_like(matrices[:, :1])
    zero_c = torch.zeros_like(masses[:, :1])
    zero_n = torch.zeros_like(counts[:, :1])
    prior_m = torch.cat((zero_m, matrices[:, :-1]), 1)
    prior_c = torch.cat((zero_c, masses[:, :-1]), 1)
    prior_n = torch.cat((zero_n, counts[:, :-1]), 1)
    current_m = keys[..., :, None] * values[..., None, :]
    if kind and kind.startswith('iv_window'):
        window = options.get('window') if kind == 'iv_window' else int(kind[9:])
        if type(window) is not int or window < 1:
            raise ValueError('history window must be a positive integer')
        if window < q.shape[1]:
            matrices = matrices - torch.cat((torch.zeros_like(matrices[:, :window]), matrices[:, :-window]), 1)
            masses = masses - torch.cat((torch.zeros_like(masses[:, :window]), masses[:, :-window]), 1)
            counts = counts - torch.cat((torch.zeros_like(counts[:, :window]), counts[:, :-window]), 1)
    elif kind == 'iv_swap':
        donor = options.get('donor')
        if not isinstance(donor, dict) or set(donor) != {'keys', 'values', 'valid'}:
            raise ValueError('iv_swap requires a normal donor history for this layer')
        dk, dv, dm = donor['keys'], donor['values'], donor['valid']
        if dk.ndim != 3 or dk.shape != dv.shape or dk.shape[0] != q.shape[0] or dk.shape[2] != q.shape[2] or dm.shape != dk.shape[:2]:
            raise ValueError('donor history shape mismatch')
        dm = dm.to(q.device)
        dk = torch.where(dm[..., None], dk.to(q.device), torch.zeros_like(dk, device=q.device))
        dv = torch.where(dm[..., None], dv.to(q.device), torch.zeros_like(dv, device=q.device))
        index = torch.arange(q.shape[1], device=q.device).clamp_max(dk.shape[1])
        matrices = torch.cat((zero_m, (dk[..., :, None]*dv[..., None, :]).cumsum(1)), 1).index_select(1, index) + current_m
        masses = torch.cat((zero_c, dk.cumsum(1)), 1).index_select(1, index) + keys
        counts = torch.cat((zero_n, dm.to(q.device).long().cumsum(1)), 1).index_select(1, index) + valid.long()
    elif kind == 'iv_permute':
        identities = options.get('example_ids')
        phase = options.get('phase')
        if not isinstance(identities, (tuple, list)) or len(identities) != q.shape[0] or not isinstance(phase, str):
            raise ValueError('iv_permute requires phase and one example_id per row')
        # Keep keys and c fixed; only valid past values are reassigned. The
        # current token is never included in the permutation.
        rows = []
        for b in range(q.shape[0]):
            positions = []
            for t in range(q.shape[1]):
                past = torch.nonzero(valid[b, :t], as_tuple=False).flatten()
                indices = list(range(past.numel()))
                if not options.get('identity_permutation', False):
                    random.Random(derive_seed(20261101, 'residual_la_v21', phase,
                        'iv_permute', identities[b], t)).shuffle(indices)
                perm = torch.tensor(indices, dtype=torch.long, device=q.device)
                paired = values[b].index_select(0, past.index_select(0, perm))
                old_keys = keys[b].index_select(0, past)
                positions.append(old_keys.transpose(0, 1) @ paired + current_m[b, t])
            rows.append(torch.stack(positions))
        matrices = torch.stack(rows)
    if kind == 'iv_strict' or (config.mode == 'strict' and kind is None):
        matrices, masses, counts = prior_m, prior_c, prior_n
    elif config.mode != 'inclusive':
        raise ValueError('history interventions other than strict require an inclusive model')
    outputs, weights = read_kernel_prefix(q, reader, matrices, masses, counts, valid,
        steps=config.steps, update=config.update, readout=config.readout,
        fixed_query=config.fixed_query, current_keys=keys if options.get('trace_reads') else None)
    if kind == 'iv_off':
        outputs = torch.zeros_like(outputs)
    diagnostics = {}
    if options.get('capture_history'):
        diagnostics['history'] = {'keys': keys.detach().clone(), 'values': values.detach().clone(),
                                  'valid': valid.detach().clone()}
    if weights is not None:
        diagnostics['current_weight'] = weights
    return SequenceResult(outputs, final,
        {'valid_tokens': valid.sum().detach(), 'writes': valid.sum().detach(),
         'reads': valid.sum().detach()*config.steps, 'reader_steps': config.steps}, diagnostics)


def summarize_current_weights(diagnostics, valid):
    """Report per-layer/per-step means and quantiles, excluding padding rows."""
    result = {}
    for layer, values in diagnostics.items():
        weights = values.get('current_weight')
        if weights is None:
            continue
        result[layer] = []
        for step in range(weights.shape[-1]):
            active = weights[..., step][valid].float()
            result[layer].append({'step': step+1, 'count': active.numel(),
                'mean': active.mean().item() if active.numel() else None,
                'quantiles': torch.quantile(active, torch.tensor([0., .25, .5, .75, 1.], device=active.device)).tolist()
                             if active.numel() else []})
    return result
