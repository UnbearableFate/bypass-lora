"""Stable objectives for marginalizing over latent memory slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def slot_marginal(
    log_weights: torch.Tensor,
    slot_logits: torch.Tensor,
    labels: torch.Tensor,
    slot_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean answer NLL and per-row log probabilities marginalized over slots.

    ``log_weights`` contains normalized log address probabilities with shape
    ``[B, N]``. ``slot_logits`` contains the answer vocabulary logits from
    reading each slot independently and has shape ``[B, N, V]``. Invalid
    slots are excluded from the mixture without renormalizing the remaining
    address weights. Every row must have at least one valid slot.
    """
    if log_weights.ndim != 2:
        raise ValueError("log_weights must have shape [B, N]")
    if slot_logits.ndim != 3:
        raise ValueError("slot_logits must have shape [B, N, V]")
    batch, slots = log_weights.shape
    if slots < 1 or slot_logits.shape[:2] != (batch, slots):
        raise ValueError("slot_logits must match log_weights batch and slot dimensions")
    vocab = slot_logits.shape[2]
    if vocab < 1:
        raise ValueError("slot_logits must contain at least one vocabulary item")
    if labels.shape != (batch,):
        raise ValueError("labels must have shape [B]")
    if labels.dtype != torch.long:
        raise ValueError("labels must have dtype torch.long")

    if slot_valid is None:
        valid = torch.ones((batch, slots), dtype=torch.bool, device=log_weights.device)
    else:
        if slot_valid.shape != (batch, slots):
            raise ValueError("slot_valid must have shape [B, N]")
        if slot_valid.dtype != torch.bool:
            raise ValueError("slot_valid must have dtype torch.bool")
        valid = slot_valid.to(device=log_weights.device)
    if not bool(valid.any(dim=1).all()):
        raise ValueError("each row must contain at least one valid slot")

    # Neutralize invalid logits before log_softmax, since an all -inf invalid
    # slot would otherwise produce NaNs that can leak through autograd.
    safe_logits = torch.where(valid[..., None], slot_logits.float(), 0.0)
    slot_log_probs = F.log_softmax(safe_logits, dim=-1)
    joint_log_probs = log_weights.float()[..., None] + slot_log_probs
    joint_log_probs = joint_log_probs.masked_fill(~valid[..., None], -torch.inf)
    log_mixture_probabilities = torch.logsumexp(joint_log_probs, dim=1)

    target_log_probs = log_mixture_probabilities.gather(1, labels.to(log_mixture_probabilities.device)[:, None])
    mean_nll = -target_log_probs.mean()
    return mean_nll, log_mixture_probabilities
