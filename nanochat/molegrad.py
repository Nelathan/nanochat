"""
MoleGrad: Low-rank gradient updates tunnel through subspace.

Core principle: Weight updates live in low-rank subspace. For each large Linear,
estimate top-k singular modes from activations A and output grads B without
forming full gradient G = BᵀA. Update only along those dominant modes.

Like moles tunneling underground: minimal surface disruption, efficient paths.
All math is GEMMs with A and B, no giant intermediates. Model-agnostic.
"""

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

power_iters: int = 1
eps: float = 1e-6
use_trust_scaling: bool = True
reduce_orthogonalization: bool = True
hutch_probes: int = 2
warm_start_decay: float = 0.95
column_norm_threshold: float = 0.1

class MoleGrad:
    """Low-rank gradient in factored form: G ≈ U @ diag(S) @ V.T

    This is what gets stored in weight.grad for MoleLinear layers.
    The optimizer knows to use addmm_ instead of materializing.
    """
    __slots__ = ('U', 'S', 'V', 'shape')

    def __init__(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, shape: tuple):
        self.U = U  # [out_dim, rank]
        self.S = S  # [rank]
        self.V = V  # [in_dim, rank]
        self.shape = shape  # (out_dim, in_dim) - for compatibility checks

    def materialize(self) -> torch.Tensor:
        """Reconstruct full gradient (only for debugging - defeats the purpose!)"""
        return (self.U * self.S.unsqueeze(0)) @ self.V.T

    def to(self, dtype):
        """Convert to different dtype"""
        return MoleGrad(
            self.U.to(dtype),
            self.S.to(dtype),
            self.V.to(dtype),
            self.shape
        )


def _lowrank_grad(A: torch.Tensor, B: torch.Tensor, Q_prev: Optional[torch.Tensor], rank_k: int):
    """
    Compute factored low-rank gradient of G = B^T A using blockwise power iteration.
    """
    in_dim = A.shape[-1]
    out_dim = B.shape[-1]
    dtype = torch.bfloat16 if torch.cuda.is_available() and hasattr(torch, 'bfloat16') else A.dtype
    A_flat = A.reshape(-1, in_dim)
    B_flat = B.reshape(-1, out_dim)

    assert min(in_dim, out_dim) > 0, f"Invalid dimensions: in_dim={in_dim}, out_dim={out_dim}"

    k = min(rank_k, in_dim, out_dim)

    # Initialize or warm-start Q (keep in bf16 for fast matmuls)
    if Q_prev is None or Q_prev.shape != (in_dim, k):
        Q = torch.randn(in_dim, k, device=A_flat.device, dtype=dtype)
        Q = Q / (Q.norm(dim=0, keepdim=True) + eps)
    else:
        # Warm-start Q with decay (convert to bf16 if needed)
        Q = warm_start_decay * Q_prev.to(device=A_flat.device, dtype=dtype)
        Q = Q + (1 - warm_start_decay) * torch.randn(in_dim, k, device=A_flat.device, dtype=dtype)

        # Smart orthogonalization: column-norm check with QR fallback
        column_norms = Q.norm(dim=0)
        min_norm = column_norms.min()
        max_norm = column_norms.max()

        needs_qr = (min_norm / (max_norm + eps)) <= column_norm_threshold
        if reduce_orthogonalization and not needs_qr:
            # Fast column-norm normalization when columns are well-behaved
            Q = Q / (Q.norm(dim=0, keepdim=True) + eps)
        else:
            # Lightweight QR: do in fp32 for stability, convert back to bf16
            Q_fp32, _ = torch.linalg.qr(Q.float(), mode="reduced")
            Q = Q_fp32.to(dtype)

    # Single Blockwise power iteration: process in chunks to reduce memory and improve cache locality
    # Ensure Q is in the right dtype before entering loop
    Q = Q.to(dtype)
    # Forward: Y = B.T @ (A @ Q) but blockwise
    Y = B_flat.T @ (A_flat @ Q)
    # Backward: Z = A.T @ (B @ Y)
    Z = A_flat.T @ (B_flat @ Y)
    # column re-orthogonalization
    Q = Z / (Z.norm(dim=0, keepdim=True) + eps)

    # Final projection and small SVD (only SVD in fp32, rest in bf16)
    Y = B_flat.T @ (A_flat @ Q)              # [out,k] in bf16
    U_block, S, V_block_T = torch.linalg.svd(Y.float(), full_matrices=False)
    r = min(k, S.shape[0])

    if r == 0:
        # Degenerate case: return empty rank-0 factors in bf16
        U = torch.zeros((out_dim, 0), device=A_flat.device, dtype=dtype)
        S = torch.zeros((0,), device=A_flat.device, dtype=dtype)
        V = torch.zeros((in_dim, 0), device=A_flat.device, dtype=dtype)
        return U, S, V, Q

    # Extract factors and return in bf16
    U = U_block[:, :r].to(dtype)    # [out,r]
    S = S[:r].to(dtype)              # [r]
    V = (Q.float() @ V_block_T.T)[:, :r].to(dtype)   # [in,r]

    # Return factored form; trust scaling is handled in backward/optimizer
    return U, S, V, Q


class _MoleGradFn(torch.autograd.Function):
    """Custom autograd function that computes low-rank gradient updates."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, state: SimpleNamespace, rank_k: int):
        ctx.save_for_backward(x, weight)
        ctx.state = state
        ctx.rank_k = rank_k
        return x.matmul(weight.t())

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        state: SimpleNamespace = ctx.state
        rank_k: int = ctx.rank_k

        if grad_out.dtype is not weight.dtype:
            grad_input = grad_out.to(weight.dtype).matmul(weight).to(x.dtype)
        else:
            grad_input = grad_out.matmul(weight)

        prev_Q = getattr(state, "Q", None)
        U, S, V, Q_next = _lowrank_grad(x, grad_out, prev_Q, rank_k)
        state.Q = Q_next
        state.last_USV = (U, S, V)
        return grad_input, None, None, None, None


class MoleLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with low-rank gradient updates.

    Core concept: Weight updates live in low-rank subspace estimated from
    activations A and output gradients B without forming full G = BᵀA.

    Key features:
    - Top-k singular modes estimated via block power iteration
    - Warm-start subspace tracking across steps
    - Memory scales with rank_k, not sequence length
    - Bias disabled for clean gradient mathematics
    """

    def __init__(self, in_features: int, out_features: int, rank_k: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.rank_k = rank_k
        self._state = SimpleNamespace(Q=None, last_USV=None)
        setattr(self.weight, "_mole_state", self._state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = _MoleGradFn.apply(x, self.weight, self._state, self.rank_k)
        assert isinstance(result, torch.Tensor)
        return result

    def extra_repr(self) -> str:
        return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, module_type={self.module_type}, rank_k={self.rank_k}'
