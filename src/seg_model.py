"""E-0025 two independent backbone forwards; only four M,c,counts cross.

No backbone KV is ever requested or transferred. The second forward computes
its own positions from its own causal mask, starting at zero. States remain
in the autograd graph during training and are fresh for every example.
"""
from contextlib import contextmanager
from dataclasses import replace

import torch

from bypass_model import BranchModel
from calibrate import load_base
from rb_data import BASE_SEED, STUDY
from rb_train import adapter_config, LAYERS
from v1_data import derive_seed
from v1_lora import LoRAConfig, LoRAModel

ARMS = tuple(f'SEG-{i:02}' for i in range(8))
KEEP = ('SEG-02', 'SEG-05')
RESET = ('SEG-03', 'SEG-06')
T0 = ('SEG-01', 'SEG-07')
FAMILIES = {**{a: 'RBc-R' for a in (*KEEP, *RESET)}, **{a: 'RBc-T' for a in T0}, 'SEG-04': 'RBc-L'}
PARAMETERS = {'SEG-00': 0, **{f'SEG-{i:02}': 663552 for i in (1, 2, 3)},
              'SEG-04': 630784, **{f'SEG-{i:02}': 2752512 for i in (5, 6, 7)}}
# E-0027 trains only the r128 triplet and LoRA_seg at new seeds; seed 0 is E-0025.
SEED_ARMS = {0: ARMS, **{s: ('SEG-04', 'SEG-05', 'SEG-06', 'SEG-07') for s in (1, 2, 3, 4, 5)}}


def contract(seed):
    if seed not in SEED_ARMS:
        raise ValueError('unknown segmented seed')
    return 'E-0025@1' if seed == 0 else 'E-0027@1' if seed in (1, 2) else 'E-0030@1'


def config_for(arm):
    return replace(adapter_config('RB1-01' if arm in T0 else 'RB1-02'),
                   rank=128 if arm in ('SEG-05', 'SEG-06', 'SEG-07') else 32)


def make_model(arm, checkpoint=None, seed=0):
    if arm not in ARMS:
        raise ValueError('unknown segmented arm')
    if arm not in SEED_ARMS.get(seed, ()):
        raise ValueError('unregistered segmented seed/arm')
    base = load_base()
    torch.manual_seed(derive_seed(BASE_SEED, STUDY, 'R1', 'init', seed))
    if arm == 'SEG-00':
        if checkpoint is not None:
            raise ValueError('Base_seg has no adapter checkpoint')
        base.requires_grad_(False)
        return base
    if arm == 'SEG-04':
        config = LoRAConfig(layers=LAYERS, rank=7, alpha=7)
        model = LoRAModel.load_adapter(base, checkpoint) if checkpoint else LoRAModel(base, config)
    else:
        config = config_for(arm)
        model = BranchModel.load_adapter(base, checkpoint) if checkpoint else BranchModel(base, config)
    if model.config != config:
        raise ValueError('segmented checkpoint config/arm mismatch')
    params = list(model.adapter_parameters())
    if sum(p.numel() for p in params) != PARAMETERS[arm] or any(
            not p.requires_grad or p.dtype != torch.float32 for p in params):
        raise ValueError('segmented parameter contract mismatch')
    return model


def parameter_digest(model):
    """SHA256 over the ordered adapter tensors (shape, dtype, bytes); E-0027 rule 1(e)."""
    import hashlib
    digest = hashlib.sha256()
    for p in model.adapter_parameters():
        x = p.detach().cpu().contiguous()
        digest.update(repr((tuple(x.shape), str(x.dtype))).encode())
        digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def zero_exits(model, enabled):
    if enabled and (model.training or torch.is_grad_enabled()):
        raise ValueError('iv_off is a frozen evaluation intervention')
    handles = []
    if enabled:
        if not isinstance(model, BranchModel):
            raise ValueError('iv_off requires a side branch')
        def off(side, inputs, output):
            z, state, accounting = output
            return torch.zeros_like(z), state, accounting
        handles = [layer.side.register_forward_hook(off) for layer in model.adapters().values()]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def first_forward(model, first_ids, first_mask):
    out = model(input_ids=first_ids, attention_mask=first_mask, use_cache=False)
    if out.cache.backbone is not None:
        raise ValueError('segment 1 exported backbone KV')
    return out


def forward(model, arm, first_ids, first_mask, second_ids, second_mask,
            kind='normal', donor_memory=None, force_first=False, timing_events=None):
    if kind not in ('normal', 'iv_off', 'iv_reset', 'iv_swap'):
        raise ValueError('unknown segmented intervention')
    if kind != 'normal' and arm not in KEEP:
        raise ValueError('only R_keep has segmented interventions')
    if kind != 'normal' and (model.training or torch.is_grad_enabled()):
        raise ValueError('segmented interventions require frozen no-grad evaluation')
    if (kind == 'iv_swap') != (donor_memory is not None):
        raise ValueError('iv_swap requires an explicit donor-only side state')
    memory, parts = None, {}
    def call(phase, ids, mask, **kwargs):
        if timing_events is None:
            return model(input_ids=ids, attention_mask=mask, use_cache=False, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = model(input_ids=ids, attention_mask=mask, use_cache=False, **kwargs)
        end.record()
        timing_events.append((phase, start, end))
        return result
    with zero_exits(model, kind == 'iv_off'):
        if arm in KEEP or force_first:
            first = call('segment1', first_ids, first_mask)
            if isinstance(model, BranchModel):
                if first.cache.backbone is not None:
                    raise ValueError('segment 1 exported backbone KV')
                parts['segment1'] = first.accounting
                if arm in KEEP and kind != 'iv_reset':
                    memory = first.cache.memory
        if kind == 'iv_swap':
            memory = donor_memory
        kwargs = {'initial_memory': memory} if memory is not None else {}
        output = call('segment2', second_ids, second_mask, **kwargs)
    if isinstance(model, BranchModel):
        if output.cache.backbone is not None:
            raise ValueError('segment 2 exported backbone KV')
        parts['segment2'] = output.accounting
    return output, parts


def state_description(states):
    return {str(i): {'M.shape': list(s.M.shape), 'c.shape': list(s.c.shape),
                    'counts.shape': list(s.counts.shape), 'M.dtype': str(s.M.dtype),
                    'c.dtype': str(s.c.dtype), 'counts.dtype': str(s.counts.dtype),
                    'M_c_bytes': (s.M.numel()+s.c.numel())*s.M.element_size(),
                    'counts_bytes': s.counts.numel()*s.counts.element_size()}
            for i, s in states.items() if s is not None}
