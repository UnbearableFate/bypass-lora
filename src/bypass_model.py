"""Qwen3 side adapters with explicit per-request memory and backbone-cache ownership.

Pinned to transformers 4.57.1. The original attention and MLP execute once;
short-lived hooks observe pre-RoPE features and inject gate/up residuals.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch
from torch import nn
import torch.nn.functional as F

from bypass_core import LatentReader, process_sequence, process_kernel_tokenwise


FUSIONS = ('F_out', 'F_pre', 'F_attnres', 'C_gate', 'C_up', 'C_dual')
# E-0014 R6: initial tau of the learnable centered-cosine address temperature.
CENTERED_COS_TEMPERATURE_INIT = 10.


@dataclass(frozen=True)
class AdapterConfig:
    layers: tuple = (6, 13, 20, 27)
    rank: int = 32
    branch: str = 'tokenwise_silu'
    fusion: str = 'C_dual'
    tap: str = 'qknorm_pre_rope'
    backend: str = 'kernel_matrix'
    steps: int = 1
    mode: str = 'inclusive'
    update: str = 'accumulate'
    readout: str = 'increment'
    eta: float = .25
    read_gain: float = 1.
    eps_norm: float = 1e-6
    eps_read: float = 1e-6
    write_scope: str = 'visible_fact_delimiter'
    slot_capacity: int = 16
    # E-0014 R4: rank of a separate query-only tokenwise F_out prior path; 0 = absent.
    prior_rank: int = 0
    model_revision: str = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
    fixed_query: bool = False
    impl: str = 'reference'

    def __post_init__(self):
        if self.fusion not in FUSIONS or self.branch not in ('tokenwise_silu', 'tokenwise_linear', 'memory', 'kernel_tokenwise'):
            raise ValueError('unknown branch or fusion')
        if self.tap not in ('qknorm_pre_rope', 'raw_qkv', 'direct_hidden'):
            raise ValueError('unknown feature tap')
        if self.steps < 1 or self.rank < 1 or len(set(self.layers)) != len(self.layers):
            raise ValueError('invalid rank, steps or duplicate layers')
        if self.mode not in ('inclusive', 'strict') or self.update not in ('accumulate', 'anchored'):
            raise ValueError('unknown recurrence')
        if self.readout not in ('increment', 'full'):
            raise ValueError('unknown readout')
        if self.write_scope not in ('visible_fact_delimiter', 'all_support_tokens') or self.slot_capacity < 1:
            raise ValueError('invalid support write scope or slot capacity')
        if type(self.prior_rank) is not int or self.prior_rank < 0:
            raise ValueError('prior_rank must be a nonnegative integer')
        if self.prior_rank and self.branch != 'memory':
            raise ValueError('the prior path is defined only beside an M-line memory branch')
        if self.impl not in ('reference', 'prefix'):
            raise ValueError('unknown memory implementation')
        if self.fixed_query and self.branch != 'memory':
            raise ValueError('fixed_query is defined only for memory')
        if self.impl == 'prefix' and self.backend != 'kernel_matrix':
            raise ValueError('prefix requires KernelMatrix')
        if self.branch == 'kernel_tokenwise' and (self.update != 'accumulate' or self.readout != 'increment' or self.read_gain != 1.):
            raise ValueError('T0 requires accumulating increment recurrence')


class LowDimProjection(nn.Module):
    def __init__(self, dims, rank):
        super().__init__()
        self.A_q = nn.Parameter(torch.randn(dims[0], rank) / dims[0] ** .5)
        self.A_k = nn.Parameter(torch.randn(dims[1], rank) / dims[1] ** .5)
        self.A_v = nn.Parameter(torch.randn(dims[2], rank) / dims[2] ** .5)

    def forward(self, q, k, v):
        return q.float() @ self.A_q, k.float() @ self.A_k, v.float() @ self.A_v


class FeatureTap:
    def __init__(self, attention, kind):
        self.attention, self.kind = attention, kind

    @contextmanager
    def capture(self, normalized_hidden):
        values, handles = {}, []
        if self.kind == 'direct_hidden':
            values.update(q=normalized_hidden, k=normalized_hidden, v=normalized_hidden)
        else:
            modules = {'q': self.attention.q_norm, 'k': self.attention.k_norm,
                       'v': self.attention.v_proj} if self.kind == 'qknorm_pre_rope' else {
                           'q': self.attention.q_proj, 'k': self.attention.k_proj, 'v': self.attention.v_proj}
            for key, module in modules.items():
                def hook(module, inputs, output, key=key):
                    values[key] = output.flatten(2)
                handles.append(module.register_forward_hook(hook))
        try:
            yield values
        finally:
            for handle in handles:
                handle.remove()


class Fusion(nn.Module):
    def __init__(self, kind, rank, width, intermediate):
        super().__init__()
        self.kind = kind
        if kind.startswith('F_'):
            self.P = nn.Parameter(torch.zeros(rank, width))
        if kind in ('C_gate', 'C_dual'):
            self.C_g = nn.Parameter(torch.zeros(rank, intermediate))
        if kind in ('C_up', 'C_dual'):
            self.C_u = nn.Parameter(torch.zeros(rank, intermediate))

    def forward(self, h, u, z, mlp, norm):
        if self.kind == 'F_out':
            return h + mlp(u) + (z @ self.P).to(h.dtype)
        if self.kind == 'F_pre':
            return h + mlp(u + (z @ self.P).to(u.dtype))
        if self.kind == 'F_attnres':
            h_new = h + (z @ self.P).to(h.dtype)
            return h_new + mlp(norm(h_new))
        handles = []
        if hasattr(self, 'C_g'):
            delta = z @ self.C_g
            handles.append(mlp.gate_proj.register_forward_hook(
                lambda module, inputs, output: output + delta.to(output.dtype)))
        if hasattr(self, 'C_u'):
            delta_u = z @ self.C_u
            handles.append(mlp.up_proj.register_forward_hook(
                lambda module, inputs, output: output + delta_u.to(output.dtype)))
        try:
            return h + mlp(u)
        finally:
            for handle in handles:
                handle.remove()


class PriorPath(nn.Module):
    """Tokenwise SiLU F_out increment ``SiLU(u A) P`` added to the layer output.

    P starts at zero, so the path is an exact identity at initialization.
    """

    def __init__(self, width, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(width, rank) / width ** .5)
        self.P = nn.Parameter(torch.zeros(rank, width))

    def forward(self, u, valid):
        return (F.silu(u.float() @ self.A) * valid.unsqueeze(-1)) @ self.P


class SideBranch(nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.config = config
        d = layer.self_attn.q_proj.in_features
        if config.branch in ('memory', 'kernel_tokenwise'):
            dims = (d, d, d) if config.tap == 'direct_hidden' else (
                layer.self_attn.q_proj.out_features, layer.self_attn.k_proj.out_features,
                layer.self_attn.v_proj.out_features)
            self.projection = LowDimProjection(dims, config.rank)
            self.reader = LatentReader(config.rank, eta=config.eta, read_gain=config.read_gain,
                                       eps_norm=config.eps_norm, eps_read=config.eps_read,
                                       temperature_init=CENTERED_COS_TEMPERATURE_INIT
                                       if config.backend == 'centered_cos_slot' else None)
        else:
            self.A = nn.Parameter(torch.randn(d, config.rank) / d ** .5)
        self.fusion = Fusion(config.fusion, config.rank, d, layer.mlp.gate_proj.out_features)
        # Created last so that memory-path initialization is unchanged by its presence.
        self.prior = PriorPath(d, config.prior_rank) if config.prior_rank else None
        self.diagnostics = None

    def forward(self, u, features, state, valid, phase, write_mask=None, runtime=None):
        cfg = self.config
        self.diagnostics = None
        if cfg.branch not in ('memory', 'kernel_tokenwise'):
            z = u.float() @ self.A
            if cfg.branch == 'tokenwise_silu':
                z = F.silu(z)
            return z * valid.unsqueeze(-1), state, None
        q, k, v = self.projection(features['q'], features['k'], features['v'])
        if cfg.branch == 'kernel_tokenwise':
            if phase != 'F':
                raise ValueError('T0 is an F-line branch')
            z = process_kernel_tokenwise(q, k, v, self.reader, valid, cfg.steps)
            if runtime and runtime.get('kind') == 'iv_off':
                z = torch.zeros_like(z)
            return z, None, {'valid_tokens': valid.sum().detach(), 'writes': 0,
                             'reads': 0, 'reader_steps': cfg.steps}
        if runtime:
            from rb_history import run_history
            result = run_history(q, k, v, self.reader, valid, cfg, runtime)
            self.diagnostics = result.diagnostics
            return result.outputs, result.state, result.accounting
        result = process_sequence(q, k, v, self.reader, state=state, valid=valid,
                                  write=(write_mask if phase == 'support' else valid) if phase != 'query' else torch.zeros_like(valid),
                                  read=valid if phase != 'support' else torch.zeros_like(valid),
                                  steps=cfg.steps, mode=cfg.mode, update=cfg.update,
                                  readout=cfg.readout, backend=cfg.backend,
                                  fixed_query=cfg.fixed_query, impl=cfg.impl)
        if phase == 'support' and (result.state.counts > cfg.slot_capacity).any():
            raise ValueError('support slot capacity exceeded')
        return result.outputs, result.state, result.accounting


class AdaptedLayer(nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.original = layer
        self.side = SideBranch(layer, config).to(device=layer.self_attn.q_proj.weight.device)
        self.attention_type = layer.attention_type
        self.request = None
        self.result = None
        self.runtime = None
        self.calls = {'attention': 0, 'ffn': 0}
        # Count executions, not the requested number of executions.
        self.handles = [layer.self_attn.register_forward_hook(self._attention_called),
                        layer.mlp.register_forward_hook(self._ffn_called)]

    def _attention_called(self, module, inputs, output):
        self.calls['attention'] += 1

    def _ffn_called(self, module, inputs, output):
        self.calls['ffn'] += 1

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
        if self.request is None:
            raise RuntimeError('use BranchModel.forward to supply state ownership and token mask')
        valid, state, phase, write_mask = self.request
        layer = self.original
        n = layer.input_layernorm(hidden_states)
        with FeatureTap(layer.self_attn, self.side.config.tap).capture(n) as features:
            o, _ = layer.self_attn(hidden_states=n, attention_mask=attention_mask,
                                  position_ids=position_ids, past_key_values=past_key_values,
                                  use_cache=use_cache, cache_position=cache_position,
                                  position_embeddings=position_embeddings, **kwargs)
        h = hidden_states + o
        u = layer.post_attention_layernorm(h)
        z, state, counts = self.side(u, features, state, valid, phase, write_mask, self.runtime)
        self.result = (state, counts)
        if phase == 'support':
            return h + layer.mlp(u)
        out = self.side.fusion(h, u, z, layer.mlp, layer.post_attention_layernorm)
        if self.side.prior is not None:
            if phase != 'query':
                raise ValueError('the prior path is defined only for M-line question forwards')
            out = out + self.side.prior(u, valid).to(out.dtype)
        return out


@dataclass(frozen=True)
class BranchCache:
    memory: dict
    backbone: object = None
    phase: str = 'F'

    def reorder(self, indices):
        # HF cache reordering is in-place. Clone it so the source owner stays usable.
        from copy import deepcopy
        backbone = deepcopy(self.backbone) if self.backbone is not None else None
        if backbone is not None:
            backbone.reorder_cache(indices)
        return BranchCache({i: s.reorder(indices) if s is not None else None
                            for i, s in self.memory.items()}, backbone, self.phase)


@dataclass
class BranchOutput:
    logits: torch.Tensor
    cache: BranchCache
    accounting: dict
    diagnostics: object = None


class BranchModel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        model.requires_grad_(False)
        self.model, self.config = model, config
        for index in config.layers:
            if isinstance(model.model.layers[index], AdaptedLayer):
                raise ValueError('model already adapted')
            model.model.layers[index] = AdaptedLayer(model.model.layers[index], config)

    def adapters(self):
        return {i: self.model.model.layers[i] for i in self.config.layers}

    def adapter_parameters(self):
        for layer in self.adapters().values():
            yield from layer.side.parameters()

    def forward(self, input_ids, attention_mask=None, cache=None, memory=None, phase='F', use_cache=False,
                write_mask=None, intervention=None, capture_history=False, trace_reads=False):
        if phase not in ('F', 'support', 'query'):
            raise ValueError('unknown protocol phase')
        runtime = None
        if intervention is not None or capture_history or trace_reads:
            if self.training or torch.is_grad_enabled() or phase != 'F' or cache is not None or memory is not None or use_cache:
                raise ValueError('history diagnostics require no-grad uncached F-line evaluation')
            if self.config.branch not in ('memory', 'kernel_tokenwise') or self.config.backend != 'kernel_matrix':
                raise ValueError('history diagnostics require a KernelMatrix/T0 branch')
            from rb_history import KINDS
            if intervention is not None and (not isinstance(intervention, dict) or intervention.get('kind') not in KINDS):
                raise ValueError('unknown history intervention')
            runtime = dict(intervention or {})
            runtime.update(capture_history=capture_history, trace_reads=trace_reads)
        if self.model.is_gradient_checkpointing:
            raise ValueError('checkpoint replay requires an explicit replay-safe state protocol')
        if cache is not None and cache.phase != phase:
            raise ValueError('backbone cache may not cross protocol phases')
        if phase == 'support' and (cache is not None or memory is not None or use_cache):
            raise ValueError('support builds fresh memory and never exports backbone KV')
        if phase != 'query' and memory is not None:
            raise ValueError('support snapshots are only accepted for read-only query')
        if cache is not None and memory is not None:
            raise ValueError('provide cache or memory, not both')
        if phase == 'query' and cache is None and memory is None:
            raise ValueError('query requires explicit support memory')
        if phase != 'support' and write_mask is not None:
            raise ValueError('write_mask is only accepted in support phase')
        if attention_mask is None:
            if cache is not None:
                raise ValueError('cached forward needs the complete attention mask')
            attention_mask = torch.ones_like(input_ids)
        valid = attention_mask[:, -input_ids.shape[1]:].bool()
        if phase == 'support':
            if write_mask is None:
                if self.config.write_scope == 'visible_fact_delimiter':
                    raise ValueError('visible_fact_delimiter requires an explicit write_mask')
                write_mask = valid
            if write_mask.shape != valid.shape or write_mask.dtype != torch.bool:
                raise ValueError('write_mask must be boolean with the input shape')
            if (write_mask & ~valid).any():
                raise ValueError('write_mask must be a subset of valid tokens')
        if cache is not None:
            if cache.backbone is None or cache.backbone.get_seq_length() != attention_mask.shape[1] - input_ids.shape[1]:
                raise ValueError('stale or reused cache; reorder/copy before reuse and request use_cache=True')
        states = cache.memory if cache else (memory if memory is not None else {})
        if phase == 'query' and self.config.branch == 'memory':
            if any(states.get(i) is None for i in self.config.layers):
                raise ValueError('query is missing support memory for an adapter layer')
        if runtime is not None and runtime.get('kind') == 'iv_swap' and self.config.branch == 'memory':
            donors = runtime.get('donor')
            if not isinstance(donors, dict) or any(i not in donors for i in self.config.layers):
                raise ValueError('iv_swap missing donor history for an adapter layer')
        before = {i: dict(layer.calls) for i, layer in self.adapters().items()}
        for i, layer in self.adapters().items():
            layer.request = (valid, states.get(i), phase, write_mask)
            layer.runtime = dict(runtime) if runtime is not None else None
            if runtime is not None and runtime.get('kind') == 'iv_swap' and self.config.branch == 'memory':
                layer.runtime['donor'] = runtime['donor'][i]
        try:
            position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)[:, -input_ids.shape[1]:]
            output = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                past_key_values=cache.backbone if cache else None,
                                use_cache=use_cache)
            new_states = {i: layer.result[0] for i, layer in self.adapters().items()}
            counts = {i: {'base': {k: layer.calls[k] - before[i][k] for k in layer.calls},
                          'side': layer.result[1]} for i, layer in self.adapters().items()}
            diagnostics = {i: layer.side.diagnostics or {} for i, layer in self.adapters().items()} if runtime is not None else None
            return BranchOutput(output.logits, BranchCache(new_states, output.past_key_values if use_cache else None, phase), counts, diagnostics)
        finally:
            for layer in self.adapters().values():
                layer.request = None
                layer.result = None
                layer.runtime = None
                layer.side.diagnostics = None

    def export_adapter(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'config.json').write_text(json.dumps(asdict(self.config), indent=2) + '\n')
        torch.save({str(i): layer.side.state_dict() for i, layer in self.adapters().items()}, directory / 'adapter.pt')

    @classmethod
    def load_adapter(cls, model, directory):
        directory = Path(directory)
        data = json.loads((directory / 'config.json').read_text())
        data['layers'] = tuple(data['layers'])
        result = cls(model, AdapterConfig(**data))
        weights = torch.load(directory / 'adapter.pt', map_location='cpu', weights_only=True)
        for i, layer in result.adapters().items():
            layer.side.load_state_dict(weights[str(i)], strict=True)
        return result
