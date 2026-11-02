"""E-0029 paired Q/K and gradient-subspace initialization. Run only in PBS."""
from contextlib import contextmanager
import hashlib
import random
import time

import torch

from bypass_core import elu_plus_one, rmsnorm_no_affine
from rb_data import BASE_SEED, STUDY, canonical
from v1_data import derive_seed, encode_training_example
from v1_run import batch_tensors, example_mean_objective, token_losses


IG_SEED = derive_seed(BASE_SEED, STUDY, 'R1', 'ig', 0)


class InitializationCheckFailed(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        summary = {index: {'RMS_error': row['relative_RMS_error'],
            'orthogonality_error': row['unscaled_orthogonality_max_error']}
            for index, row in diagnostics['layers'].items()}
        super().__init__(f'I-G scale or orthogonality check failed: {summary}')


def calibration_examples(splits, tok, excluded, count=256):
    """One pooled, without-replacement train sample, shared by all three taps."""
    excluded = {(row['task'], row['id']) for row in excluded}
    population = [row for task in ('piqa', 'boolq')
                  for row in splits['splits'][task]['train']
                  if (task, row['id']) not in excluded]
    selected = random.Random(IG_SEED).sample(population, count)
    examples = [encode_training_example(tok, row, max_length=512) for row in selected]
    identities = [{'task': row['task'], 'id': row['id'], 'input_tokens': len(encoded['input_ids'])}
                  for row, encoded in zip(selected, examples)]
    record = {'seed': IG_SEED, 'sampling': 'pooled RB train; task/source order; sample without replacement',
              'examples': identities, 'input_tokens': sum(row['input_tokens'] for row in identities),
              'examples_sha256': hashlib.sha256(canonical(identities).encode()).hexdigest(),
              'loss_normalization': 'per-example answer-token mean / 2 (formal batch size)',
              'diagnostic_subset': 'first 64 sampled examples; same IDs at step0 and final'}
    return examples, record


def grouped_query_projection(A_k, query_heads, kv_heads, head_dim):
    """G_group A_k without materializing a dense, mostly zero G_group."""
    if query_heads % kv_heads or A_k.shape[0] != kv_heads * head_dim:
        raise ValueError('invalid GQA projection dimensions')
    group = query_heads // kv_heads
    return A_k.reshape(kv_heads, head_dim, -1).repeat_interleave(group, dim=0).reshape(
        query_heads * head_dim, -1) / group ** .5


@torch.no_grad()
def pair_qk(model):
    for layer in model.adapters().values():
        side = layer.side
        projection = side.projection
        if model.config.tap == 'direct_hidden':
            value = projection.A_k
        elif model.config.tap == 'qknorm_pre_rope':
            attention = layer.original.self_attn
            dim = attention.head_dim
            value = grouped_query_projection(projection.A_k,
                attention.q_proj.out_features // dim, attention.k_proj.out_features // dim, dim)
        else:
            raise ValueError('paired initialization supports T and N taps only')
        projection.A_q.copy_(value)
        if projection.A_q is projection.A_k or projection.A_q.data_ptr() == projection.A_k.data_ptr():
            raise ValueError('Q/K parameters must remain independent')


@torch.no_grad()
def full_value_prefix(qbar, kbar, values, reader, valid, chunk_size=64):
    """Inclusive p_t via full-value sufficient statistics, bounded prefix storage.

    Q/K use the actual S1 initial reader. The uncompressed value dimension is
    independent of the key dimension; padding neither writes nor reads.
    """
    if qbar.shape != kbar.shape or values.shape[:2] != qbar.shape[:2] or valid.shape != qbar.shape[:2]:
        raise ValueError('full-value prefix shapes differ')
    if chunk_size < 1:
        raise ValueError('positive prefix chunk required')
    q = elu_plus_one(rmsnorm_no_affine(qbar.float(), reader.eps_norm) @ reader.Rq)
    k = elu_plus_one(rmsnorm_no_affine(kbar.float(), reader.eps_norm)) * valid[..., None]
    values = values.float() * valid[..., None]
    batch, length, rank = k.shape
    matrix = k.new_zeros(batch, rank, values.shape[-1])
    mass = k.new_zeros(batch, rank)
    result = []
    for start in range(0, length, chunk_size):
        stop = min(length, start + chunk_size)
        keys, features = k[:, start:stop], values[:, start:stop]
        matrices = (keys[..., :, None] * features[..., None, :]).cumsum(1) + matrix[:, None]
        masses = keys.cumsum(1) + mass[:, None]
        queries = q[:, start:stop]
        p = torch.einsum('btr,btrd->btd', queries, matrices) / (
            (queries * masses).sum(-1, keepdim=True) + reader.eps_read)
        result.append(p * valid[:, start:stop, None])
        matrix, mass = matrices[:, -1], masses[:, -1]
    return torch.cat(result, 1) if result else values.new_zeros(values.shape)


def mixed_gradient(p, d, valid):
    """All valid input positions, including prompt positions with indirect loss."""
    return p[valid].float().T @ d[valid].float()


def basis_orthogonality_error(basis):
    """Validate stored columns in FP64, independent of TF32 matmul settings."""
    columns = basis.double()
    return (columns.T @ columns - torch.eye(columns.shape[1], device=columns.device,
                                            dtype=torch.float64)).abs().max().item()


def scaled_subspace(gradient, covariance, random_z_sum_squares, reader, rank):
    if not torch.isfinite(gradient).all() or not torch.isfinite(covariance).all():
        raise ValueError('nonfinite initialization statistics')
    # CUDA's default FP32 Jacobi decomposition returned non-orthogonal left
    # vectors on the observed task gradients. Use the accuracy-oriented QR
    # driver in FP64, and only then round the retained vectors to storage dtype.
    decomposition_input = gradient.double()
    options = {'driver': 'gesvd'} if gradient.is_cuda else {}
    left, singular, right = torch.linalg.svd(decomposition_input, full_matrices=False, **options)
    precise_basis = left[:, :rank]
    precise_orth_error = basis_orthogonality_error(precise_basis)
    basis = precise_basis.float()
    if not singular.square().sum() > 0:
        raise ValueError('zero I-G task gradient')
    orth_error = basis_orthogonality_error(basis)
    fp32_product_error = (basis.T @ basis - torch.eye(rank, device=basis.device)).abs().max().item()
    projected = basis @ reader.Ro.detach()
    candidate_sum = reader.eta ** 2 * (projected * (covariance @ projected)).sum().item()
    if candidate_sum <= 0 or random_z_sum_squares <= 0:
        raise ValueError('zero I-G readout energy')
    scale = (random_z_sum_squares / candidate_sum) ** .5
    energy = singular.square()
    return basis * scale, {'unscaled_orthogonality_max_error': orth_error,
        'precast_orthogonality_max_error': precise_orth_error,
        'svd_input_dtype': str(decomposition_input.dtype),
        'svd_driver': 'gesvd' if gradient.is_cuda else 'CPU LAPACK',
        'condition_number_descriptive': float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
        'top_rank_condition_number_descriptive': float(singular[0]/singular[rank-1]) if singular[rank-1] > 0 else None,
        'leading_singular_residual_relative': float((decomposition_input @ right[:rank].T -
            precise_basis * singular[:rank]).norm()/decomposition_input.norm()),
        'orthogonality_check_dtype': 'torch.float64',
        'FP32_orthogonality_product_max_error_descriptive': fp32_product_error,
        'singular_values': singular.tolist(), 'top_rank_energy_fraction': float(energy[:rank].sum()/energy.sum()),
        'scale': scale, 'random_Z_sum_squares': random_z_sum_squares,
        'unscaled_Z_sum_squares': candidate_sum}


class GradientCollector:
    """Activation hooks do not unfreeze any backbone weight or take optimizer steps."""
    def __init__(self, model):
        self.model, self.valid, self.pending, self.stats = model, None, {}, {}
        self.handles = []
        for index, layer in model.adapters().items():
            value = layer.side.projection.A_v
            width = layer.original.mlp.up_proj.out_features
            self.stats[index] = {'G': value.new_zeros(value.shape[0], width),
                'covariance': value.new_zeros(value.shape[0], value.shape[0]), 'Z2': 0., 'count': 0,
                'direct_Z2': 0., 'direct_Z_error2': 0.}
            self.handles.extend([
                layer.side.projection.register_forward_hook(self._projection(index, layer.side.reader)),
                layer.side.register_forward_hook(self._readout(index)),
                layer.original.mlp.up_proj.register_forward_hook(self._up(index))])

    def _projection(self, index, reader):
        def hook(module, args, result):
            p = full_value_prefix(result[0].detach(), result[1].detach(), args[2].detach(), reader, self.valid)
            self.pending[index] = p
            selected = p[self.valid]
            self.stats[index]['covariance'].add_(selected.T @ selected)
        return hook

    def _readout(self, index):
        def hook(module, args, result):
            selected = result[0].detach()[self.valid]
            self.stats[index]['Z2'] += selected.double().square().sum().item()
            self.stats[index]['count'] += selected.numel()
            side = self.model.adapters()[index].side
            ideal = side.reader.eta * (self.pending[index] @ side.projection.A_v.detach() @ side.reader.Ro.detach())
            ideal = ideal[self.valid]
            self.stats[index]['direct_Z2'] += ideal.double().square().sum().item()
            self.stats[index]['direct_Z_error2'] += (ideal.double()-selected.double()).square().sum().item()
        return hook

    def _up(self, index):
        def hook(module, args, output):
            # First adapted up output can have no graph because the backbone is frozen.
            # Making only this activation require gradients exposes the true addition-point d_t.
            output = output.requires_grad_(True)
            p, valid = self.pending[index], self.valid
            def accumulate(d):
                self.stats[index]['G'].add_(mixed_gradient(p, d, valid))
            output.register_hook(accumulate)
            return output
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.pending.clear()


@contextmanager
def readout_sums(model):
    sums = {index: {'sum_squares': 0., 'count': 0} for index in model.adapters()}
    handles = []
    for index, layer in model.adapters().items():
        def hook(module, args, output, index=index):
            selected = output[0].detach()[args[3].bool()]
            sums[index]['sum_squares'] += selected.double().square().sum().item()
            sums[index]['count'] += selected.numel()
        handles.append(layer.side.register_forward_hook(hook))
    try:
        yield sums
    finally:
        for handle in handles:
            handle.remove()


def initialize_gradient(model, examples, tok):
    if any(torch.count_nonzero(layer.side.fusion.C_u) for layer in model.adapters().values()):
        raise ValueError('I-G requires zero exits')
    tick = time.monotonic()
    model.eval()
    collector = GradientCollector(model)
    try:
        for example in examples:
            model.zero_grad(set_to_none=True)
            batch = batch_tensors([example], tok.pad_token_id)
            collector.valid = batch['attention_mask'].bool()
            losses, _ = token_losses(model, batch)
            if not torch.isfinite(losses).all():
                raise ValueError('nonfinite I-G calibration loss')
            example_mean_objective(losses, 2).backward()
    finally:
        collector.close()
    collection_seconds = time.monotonic() - tick
    records = {}
    with torch.no_grad():
        for index, layer in model.adapters().items():
            stats = collector.stats[index]
            projection = layer.side.projection
            value, record = scaled_subspace(stats['G'], stats['covariance'], stats['Z2'],
                layer.side.reader, model.config.rank)
            projection.A_v.copy_(value)
            record['random_Z_rms'] = (stats['Z2']/stats['count']) ** .5
            record['gradient_shape'] = list(stats['G'].shape)
            record['gradient_dtype'] = str(stats['G'].dtype)
            record['float32_matmul_precision'] = torch.get_float32_matmul_precision()
            record['cuda_matmul_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
            record['gradient_frobenius_norm'] = stats['G'].norm().item()
            record['full_value_random_Z_sum_squares'] = stats['direct_Z2']
            record['full_value_relative_L2_error_vs_actual_Z'] = (stats['direct_Z_error2']/stats['Z2']) ** .5
            effective = projection.A_v.detach() @ layer.side.reader.Ro.detach()
            # This uses the newly assigned basis. Record the covariance-implied
            # scaled energy alongside the independent actual forward check.
            record['covariance_scaled_Z_sum_squares'] = layer.side.reader.eta ** 2 * (
                effective * (stats['covariance'] @ effective)).sum().item()
            records[str(index)] = record
        with readout_sums(model) as measured:
            for example in examples:
                batch = batch_tensors([example], tok.pad_token_id)
                model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False)
    for index, values in measured.items():
        record = records[str(index)]
        record['scaled_Z_rms'] = (values['sum_squares']/values['count']) ** .5
        record['relative_RMS_error'] = abs(record['scaled_Z_rms']/record['random_Z_rms'] - 1)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    passed = all(row['relative_RMS_error'] <= 1e-3 and row['unscaled_orthogonality_max_error'] <= 1e-4
                 for row in records.values())
    result = {'layers': records, 'collection_seconds': collection_seconds,
        'total_seconds': time.monotonic() - tick, 'optimizer_steps': 0,
        'gradient_positions': 'all nonpadding; includes indirect prompt gradients', 'passed': passed}
    if not passed:
        raise InitializationCheckFailed(result)
    return result
