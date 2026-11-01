"""Pure PyTorch reference core for the low-dimensional bypass side branch.

All memory transitions are functional: returned states own fresh tensors, so
reordered beam rows (including duplicate indices) can be advanced independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def rmsnorm_no_affine(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Last-dimension RMS normalization without learned scale or bias."""
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Positive ELU feature map, ``ELU(x) + 1``."""
    return F.elu(x) + 1.0


class KernelMatrix(nn.Module):
    """Normalized positive-kernel read from the sufficient statistics M,c."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, q: torch.Tensor, state: "MemoryState") -> torch.Tensor:
        if state.M is None or state.c is None:
            raise ValueError("KernelMatrix requires M and c")
        numerator = torch.einsum("br,brv->bv", q.float(), state.M)
        denominator = (q.float() * state.c).sum(dim=-1, keepdim=True) + self.eps
        out = numerator / denominator
        return torch.where(state.counts[:, None] > 0, out, torch.zeros_like(out))


class KernelSlotReference(nn.Module):
    """Explicit-slot reference implementation of the same positive-kernel read."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, q: torch.Tensor, state: "MemoryState") -> torch.Tensor:
        if state.slot_keys is None or state.slot_values is None or state.slot_valid is None:
            raise ValueError("KernelSlotReference requires stored slots")
        scores = torch.einsum("br,bnr->bn", q.float(), state.slot_keys.float())
        weights = scores * state.slot_valid.to(scores.dtype)
        denominator = weights.sum(dim=-1, keepdim=True) + self.eps
        out = torch.einsum("bn,bnv->bv", weights / denominator, state.slot_values.float())
        return torch.where(state.counts[:, None] > 0, out, torch.zeros_like(out))


class SoftmaxSlot(nn.Module):
    """Explicit-slot softmax reader with fixed ``sqrt(rank)`` score scale."""

    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, state: "MemoryState") -> torch.Tensor:
        if state.slot_keys is None or state.slot_values is None or state.slot_valid is None:
            raise ValueError("SoftmaxSlot requires stored slots")
        scores = torch.einsum("br,bnr->bn", q.float(), state.slot_keys.float())
        scores = scores / (state.rank ** 0.5)
        scores = scores.masked_fill(~state.slot_valid, -torch.inf)
        empty = state.counts == 0
        scores = torch.where(empty[:, None], torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(state.slot_valid, weights, torch.zeros_like(weights))
        out = torch.einsum("bn,bnv->bv", weights, state.slot_values.float())
        return torch.where(empty[:, None], torch.zeros_like(out), out)


@dataclass(frozen=True)
class MemoryState:
    """Batched, differentiable memory state. M and c are always FP32."""

    M: Optional[torch.Tensor]
    c: Optional[torch.Tensor]
    counts: torch.Tensor
    backend: str
    rank: int
    slot_keys: Optional[torch.Tensor] = None
    slot_values: Optional[torch.Tensor] = None
    slot_valid: Optional[torch.Tensor] = None

    @classmethod
    def empty(cls, batch_size: int, rank: int, backend: str = "kernel_matrix",
              device: Optional[torch.device] = None) -> "MemoryState":
        if backend not in {"kernel_matrix", "kernel_slot_reference", "soft_slot"}:
            raise ValueError(f"unknown backend: {backend}")
        M = torch.zeros(batch_size, rank, rank, dtype=torch.float32, device=device)
        c = torch.zeros(batch_size, rank, dtype=torch.float32, device=device)
        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        if backend == "kernel_matrix":
            return cls(M, c, counts, backend, rank)
        slots = torch.empty(batch_size, 0, rank, dtype=torch.float32, device=device)
        valid = torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        return cls(M, c, counts, backend, rank, slots, slots.clone(), valid)

    @property
    def batch_size(self) -> int:
        return int(self.counts.shape[0])

    @property
    def slot_count(self) -> Optional[int]:
        return None if self.slot_keys is None else int(self.slot_keys.shape[1])

    def write(self, k: torch.Tensor, v: torch.Tensor,
              valid: Optional[torch.Tensor] = None) -> "MemoryState":
        """Functionally add one token per row; masked rows are unchanged."""
        if k.shape != (self.batch_size, self.rank):
            raise ValueError(f"k must have shape {(self.batch_size, self.rank)}")
        if v.shape != (self.batch_size, self.rank):
            raise ValueError(f"v must have shape {(self.batch_size, self.rank)}")
        if valid is None:
            valid = torch.ones(self.batch_size, dtype=torch.bool, device=k.device)
        valid = valid.to(device=k.device, dtype=torch.bool)
        kf, vf = k.float(), v.float()
        m_add = torch.einsum("br,bv->brv", kf, vf)
        m_add = torch.where(valid[:, None, None], m_add, torch.zeros_like(m_add))
        k_add = torch.where(valid[:, None], kf, torch.zeros_like(kf))
        M = self.M + m_add
        c = self.c + k_add
        counts = self.counts + valid.to(self.counts.dtype)
        if self.slot_keys is None:
            return MemoryState(M, c, counts, self.backend, self.rank)
        slot_keys = torch.cat((self.slot_keys, k_add[:, None, :]), dim=1)
        v_add = torch.where(valid[:, None], vf, torch.zeros_like(vf))
        slot_values = torch.cat((self.slot_values, v_add[:, None, :]), dim=1)
        slot_valid = torch.cat((self.slot_valid, valid[:, None]), dim=1)
        return MemoryState(M, c, counts, self.backend, self.rank,
                           slot_keys, slot_values, slot_valid)

    def reset(self, rows: Optional[torch.Tensor] = None) -> "MemoryState":
        """Clear every row, or selected rows, without mutating this snapshot."""
        if rows is None:
            return MemoryState.empty(self.batch_size, self.rank, self.backend,
                                     device=self.counts.device)
        rows = rows.to(device=self.counts.device, dtype=torch.bool)
        if bool(rows.all()):
            return MemoryState.empty(self.batch_size, self.rank, self.backend,
                                     device=self.counts.device)
        M = torch.where(rows[:, None, None], torch.zeros_like(self.M), self.M)
        c = torch.where(rows[:, None], torch.zeros_like(self.c), self.c)
        counts = torch.where(rows, torch.zeros_like(self.counts), self.counts)
        if self.slot_valid is None:
            return MemoryState(M, c, counts, self.backend, self.rank)
        slot_valid = self.slot_valid & ~rows[:, None]
        slot_keys = torch.where(rows[:, None, None], torch.zeros_like(self.slot_keys), self.slot_keys)
        slot_values = torch.where(rows[:, None, None], torch.zeros_like(self.slot_values), self.slot_values)
        return MemoryState(M, c, counts, self.backend, self.rank,
                           slot_keys, slot_values, slot_valid)

    def reorder(self, indices: torch.Tensor) -> "MemoryState":
        """Return selected rows; duplicate beam indices have independent storage."""
        indices = indices.to(device=self.counts.device, dtype=torch.long)
        args = [self.M.index_select(0, indices).clone(),
                self.c.index_select(0, indices).clone(),
                self.counts.index_select(0, indices).clone()]
        if self.slot_keys is None:
            return MemoryState(*args, self.backend, self.rank)
        return MemoryState(*args, self.backend, self.rank,
                           self.slot_keys.index_select(0, indices).clone(),
                           self.slot_values.index_select(0, indices).clone(),
                           self.slot_valid.index_select(0, indices).clone())

    def read(self, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if self.backend == "kernel_matrix":
            return KernelMatrix(eps)(q, self)
        if self.backend == "kernel_slot_reference":
            return KernelSlotReference(eps)(q, self)
        return SoftmaxSlot()(q, self)


class LatentReader(nn.Module):
    """Shared Rq/Ro latent-memory loop with fixed eta and read gain defaults."""

    def __init__(self, rank: int, eta: float = 0.25, read_gain: float = 1.0,
                 eps_norm: float = 1e-6, eps_read: float = 1e-6):
        super().__init__()
        self.rank = rank
        self.eta = float(eta)
        self.read_gain = float(read_gain)
        self.eps_norm = eps_norm
        self.eps_read = eps_read
        self.last_read_calls = 0
        self.Rq = nn.Parameter(torch.eye(rank))
        self.Ro = nn.Parameter(torch.eye(rank))

    def forward(self, qbar: torch.Tensor, state: MemoryState, steps: int = 1,
                update: str = "accumulate", readout: str = "increment",
                fixed_query: bool = False) -> torch.Tensor:
        if qbar.ndim != 2 or qbar.shape[-1] != self.rank:
            raise ValueError("qbar must have shape [batch, rank]")
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if update not in {"accumulate", "anchored"}:
            raise ValueError("update must be 'accumulate' or 'anchored'")
        if readout not in {"increment", "full"}:
            raise ValueError("readout must be 'increment' or 'full'")
        self.last_read_calls = 0
        z0 = qbar
        z = z0
        fixed_q = elu_plus_one(rmsnorm_no_affine(z0, self.eps_norm) @ self.Rq) if fixed_query else None
        for _ in range(steps):
            query = fixed_q if fixed_query else elu_plus_one(
                rmsnorm_no_affine(z, self.eps_norm) @ self.Rq)
            a = state.read(query, self.eps_read) * self.read_gain
            self.last_read_calls += 1
            candidate = z0 if update == "anchored" else z
            z = candidate + self.eta * (a @ self.Ro)
        return z - z0 if readout == "increment" else z


@dataclass(frozen=True)
class SequenceResult:
    outputs: torch.Tensor
    state: MemoryState
    accounting: dict[str, int]


def process_sequence(qbar: torch.Tensor, kbar: torch.Tensor, vbar: torch.Tensor,
                     reader: LatentReader, state: Optional[MemoryState] = None,
                     valid: Optional[torch.Tensor] = None,
                     reset: Optional[torch.Tensor] = None,
                     write: Optional[torch.Tensor] = None,
                     read: Optional[torch.Tensor] = None,
                     steps: int = 1, mode: str = "inclusive",
                     update: str = "accumulate", readout: str = "increment",
                     fixed_query: bool = False,
                     backend: str = "kernel_matrix") -> SequenceResult:
    """Process [B,T,r] features causally, writing at most once per token.

    ``mode='inclusive'`` writes before reading; ``mode='strict'`` reads first.
    Set ``write=False`` per position for read-only M-line queries.
    """
    if qbar.ndim != 3 or kbar.shape != qbar.shape or vbar.shape != qbar.shape:
        raise ValueError("qbar, kbar, vbar must share [batch, time, rank] shape")
    if mode not in {"inclusive", "strict"}:
        raise ValueError("mode must be 'inclusive' or 'strict'")
    batch, time, rank = qbar.shape
    if rank != reader.rank:
        raise ValueError("feature rank does not match reader")
    if state is None:
        state = MemoryState.empty(batch, rank, backend=backend, device=qbar.device)
    if state.batch_size != batch or state.rank != rank:
        raise ValueError("initial state shape does not match features")
    valid = torch.ones(batch, time, dtype=torch.bool, device=qbar.device) if valid is None else valid.bool()
    reset = torch.zeros(batch, time, dtype=torch.bool, device=qbar.device) if reset is None else reset.bool()
    write = torch.ones(batch, time, dtype=torch.bool, device=qbar.device) if write is None else write.bool()
    read = valid if read is None else read.bool()
    if (valid.shape != (batch, time) or reset.shape != (batch, time)
            or write.shape != (batch, time) or read.shape != (batch, time)):
        raise ValueError("valid/reset/write/read masks must have shape [batch, time]")
    outs = []
    reads = writes = valid_count = 0
    for t in range(time):
        if bool(reset[:, t].any()):
            state = state.reset(reset[:, t])
        active = valid[:, t]
        do_write = active & write[:, t]
        do_read = active & read[:, t]
        valid_count += int(active.sum().item())
        if mode == "inclusive" and bool(do_write.any()):
            state = state.write(
                elu_plus_one(rmsnorm_no_affine(kbar[:, t], reader.eps_norm)),
                vbar[:, t], do_write)
        # Skip the reader entirely for a fully padded timestep.
        read_rows = torch.nonzero(do_read, as_tuple=False).flatten()
        if read_rows.numel():
            read_values = reader(qbar[:, t].index_select(0, read_rows),
                                 state.reorder(read_rows), steps, update, readout, fixed_query)
            z = torch.zeros_like(qbar[:, t]).index_copy(0, read_rows, read_values)
            reads += reader.last_read_calls * int(read_rows.numel())
        else:
            z = torch.zeros_like(qbar[:, t])
        outs.append(z)
        if mode == "strict" and bool(do_write.any()):
            state = state.write(
                elu_plus_one(rmsnorm_no_affine(kbar[:, t], reader.eps_norm)),
                vbar[:, t], do_write)
        writes += int(do_write.sum().item())
    outputs = torch.stack(outs, dim=1) if outs else qbar.new_zeros(batch, 0, rank)
    return SequenceResult(outputs, state, {
        "valid_tokens": valid_count,
        "writes": writes,
        "reads": reads,
        "reader_steps": steps,
    })


__all__ = [
    "elu_plus_one", "rmsnorm_no_affine", "MemoryState", "KernelMatrix",
    "KernelSlotReference", "SoftmaxSlot", "LatentReader", "SequenceResult",
    "process_sequence",
]

