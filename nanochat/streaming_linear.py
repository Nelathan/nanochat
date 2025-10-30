"""
MoleGrad: Low-rank gradient updates tunnel through subspace.

Core principle: Weight updates live in low-rank subspace. For each large Linear,
estimate top-k singular modes from activations A and output grads B without
forming full gradient G = BᵀA. Update only along those dominant modes.

Like moles tunneling underground: minimal surface disruption, efficient paths.
All math is GEMMs with A and B, no giant intermediates. Model-agnostic.
"""

import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn


class MoleGradConfig:
    """
    Configuration for MoleGrad low-rank updates.

    Optimized defaults:
    - rank_k=32: Dominant modes to track per layer
    - power_iters=1: Fast block power iteration
    - adaptive orthogonalization: Column-norm fallback
    - trust_scaling: Hutchinson norm for stability
    """

    def __init__(
        self,
        *,
        rank_k: int = 32,
        power_iters: int = 1,
        eps: float = 1e-6,
        use_trust_scaling: bool = True,
        reduce_orthogonalization: bool = True,
        hutch_probes: int = 2,
        min_layer_utilization: float = 2.0,
        warm_start_decay: float = 0.95,
        column_norm_threshold: float = 0.1,
    ):
        # Validate and set core parameters
        if rank_k < 1:
            raise ValueError(f"rank_k must be >= 1, got {rank_k}")
        if power_iters < 0:
            raise ValueError(f"power_iters must be >= 0, got {power_iters}")
        if not (0.0 < warm_start_decay <= 1.0):
            raise ValueError(f"warm_start_decay must be in (0, 1], got {warm_start_decay}")

        self.rank_k = int(rank_k)
        self.power_iters = int(power_iters)
        self.eps = float(eps)
        self.use_trust_scaling = bool(use_trust_scaling)
        self.reduce_orthogonalization = bool(reduce_orthogonalization)
        self.hutch_probes = int(hutch_probes)
        self.min_layer_utilization = float(min_layer_utilization)

        # Advanced features
        self.warm_start_decay = float(warm_start_decay)
        self.column_norm_threshold = float(column_norm_threshold)

  
    
    def __repr__(self) -> str:
        return (f"StreamingSVDConfig(rank_k={self.rank_k}, power_iters={self.power_iters}, "
                f"use_trust_scaling={self.use_trust_scaling}, "
                f"warm_start_decay={self.warm_start_decay})")


def _hutch_fro_norm_estimate(A: torch.Tensor, B: torch.Tensor, probes: int = 2, eps: float = 1e-8) -> float:
    """Estimate ||G||_F where G = B^T A using Hutchinson estimator for norm matching."""
    in_dim = A.shape[1]
    est_sq = 0.0
    for _ in range(probes):
        z = torch.randn(in_dim, 1, device=A.device, dtype=A.dtype)
        Gz = B.T @ (A @ z)
        est_sq += torch.sum(Gz * Gz).item()
    est_sq /= probes
    return math.sqrt(est_sq + eps)


def _lowrank_grad(A: torch.Tensor, B: torch.Tensor, cfg: MoleGradConfig, Q_prev: Optional[torch.Tensor] = None):
    """
    Compute low-rank approximation of gradient G = B^T A using block power iteration.

    Args:
        A: Input activations [N, in_dim]
        B: Output gradients [N, out_dim]
        cfg: MoleGrad configuration
        Q_prev: Previous subspace basis for warm-start [in_dim, k]

    Returns:
        G_hat: Low-rank gradient approximation [out_dim, in_dim]
        Q_next: Updated subspace basis for next step [in_dim, k]
    """
    # A: [N,in], B: [N,out]
    in_dim = A.shape[1]
    out_dim = B.shape[1]

    if cfg.rank_k <= 0 or min(in_dim, out_dim) <= 0:
        return torch.zeros((out_dim, in_dim), device=A.device, dtype=A.dtype), None

    k = min(cfg.rank_k, in_dim, out_dim)
    eps = cfg.eps

    # Initialize or warm-start Q
    if (Q_prev is None or Q_prev.shape != (in_dim, k) or
        Q_prev.device != A.device or Q_prev.dtype != A.dtype):
        Q = torch.randn(in_dim, k, device=A.device, dtype=A.dtype)
        Q = Q / (Q.norm(dim=0, keepdim=True) + eps)
    else:
        # Warm-start Q with decay for stability across steps
        Q = cfg.warm_start_decay * Q_prev
        # Add small random component for exploration
        Q = Q + (1 - cfg.warm_start_decay) * torch.randn_like(Q)

        # Smart orthogonalization: column-norm check with QR fallback
        column_norms = Q.norm(dim=0)
        min_norm_ratio = column_norms.min() / column_norms.max()

        if cfg.reduce_orthogonalization and min_norm_ratio > cfg.column_norm_threshold:
            # Fast column-norm normalization when columns are well-behaved
            Q = Q / (Q.norm(dim=0, keepdim=True) + eps)
        else:
            # Full QR when columns are degenerate or threshold is breached
            Q, _ = torch.linalg.qr(Q, mode="reduced")

    # Power iterations to find top-k subspace
    for i in range(cfg.power_iters):
        Y = B.T @ (A @ Q)          # [out,k]
        Z = A.T @ (B @ Y)          # [in,k]
        if cfg.reduce_orthogonalization and i < cfg.power_iters - 1:
            Q = Z / (Z.norm(dim=0, keepdim=True) + eps)
        else:
            Q, _ = torch.linalg.qr(Z, mode="reduced")

    # Final projection and small SVD
    Y = B.T @ (A @ Q)              # [out,k]
    U_block, S, V_block_T = torch.linalg.svd(Y, full_matrices=False)
    r = min(k, S.shape[0])

    if r == 0:
        return torch.zeros((out_dim, in_dim), device=A.device, dtype=A.dtype), Q

    U = U_block[:, :r]             # [out,r]
    S = S[:r]                      # [r]
    V = (Q @ V_block_T.T)[:, :r]   # [in,r]

    # Assemble low-rank gradient
    G_hat = (U * S.unsqueeze(0)) @ V.T  # [out,in]

    # Optional trust scaling to preserve gradient norm
    if cfg.use_trust_scaling:
        true_norm = _hutch_fro_norm_estimate(A, B, probes=cfg.hutch_probes, eps=eps)
        comp_norm = torch.norm(G_hat)
        if comp_norm > eps:
            G_hat = G_hat * (true_norm / comp_norm)

    return G_hat, Q


class _MoleGradFn(torch.autograd.Function):
    """Custom autograd function that computes low-rank gradient updates."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, cfg: MoleGradConfig, state: SimpleNamespace):
        ctx.save_for_backward(x, weight)
        ctx.cfg = cfg
        ctx.state = state
        return x.matmul(weight.t())

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        cfg: MoleGradConfig = ctx.cfg
        state: SimpleNamespace = ctx.state

        # Standard input gradient
        grad_input = grad_out.matmul(weight)

        # Low-rank gradient update
        G_hat, Q_next = _lowrank_grad(x, grad_out, cfg, getattr(state, "Q", None))
        state.Q = Q_next  # warm-start for next step

        return grad_input, G_hat, None, None


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

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False, cfg: MoleGradConfig):
        super().__init__()
        assert not bias, "Bias disabled to keep gradient math crisp."
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.cfg = cfg
        self._state = SimpleNamespace(Q=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _MoleGradFn.apply(x, self.weight, self.cfg, self._state)

    def extra_repr(self) -> str:
        return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, rank_k={self.cfg.rank_k}'


def replace_linears_with_molegrad(module: nn.Module, cfg: MoleGradConfig, *,
                                   size_multiplier: float = None,
                                   target_layers: str = "ffn_only") -> nn.Module:
    """
    Replace nn.Linear layers with MoleLinear based on heuristics.

    Args:
        module: Model to modify
        cfg: MoleGrad configuration
        size_multiplier: Minimum size threshold
        target_layers: "ffn_only", "attention_only", or "all"

    Returns:
        Modified module
    """
    size_multiplier = size_multiplier or cfg.min_layer_utilization

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            out_f, in_f = child.weight.shape

            # Skip tiny layers where compression adds overhead
            base_thresh = max(cfg.rank_k * (in_f + out_f), cfg.rank_k * cfg.rank_k)
            if (out_f * in_f) < size_multiplier * base_thresh:
                continue

            # Skip if rank too close to full rank
            if cfg.rank_k >= min(out_f, in_f):
                continue

            # Layer type filtering based on naming conventions
            layer_name = name.lower()
            should_replace = False

            if target_layers == "ffn_only":
                # FFN layers typically contain 'mlp', 'fc', 'linear2', etc.
                ffn_indicators = ['mlp', 'fc', 'linear2', 'ffn', 'feed_forward']
                should_replace = any(ind in layer_name for ind in ffn_indicators)
            elif target_layers == "attention_only":
                # Attention projections contain 'q_proj', 'k_proj', 'v_proj', 'o_proj'
                attn_indicators = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention', 'attn']
                should_replace = any(ind in layer_name for ind in attn_indicators)
            else:  # "all"
                should_replace = True

            if should_replace:
                new = MoleLinear(in_f, out_f, bias=False, cfg=cfg)
                with torch.no_grad():
                    new.weight.copy_(child.weight)
                setattr(module, name, new)
        else:
            # Recursively process child modules
            replace_linears_with_streaming(child, cfg, size_multiplier=size_multiplier, target_layers=target_layers)

    return module