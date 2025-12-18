"""
MoleGrad: Low-rank gradient updates tunnel through subspace.

Core principle: Weight updates live in low-rank subspace. For each large Linear,
estimate top-k singular modes from activations A and output grads B without
forming full gradient G = BᵀA. Update only along those dominant modes.

Like moles tunneling underground: minimal surface disruption, efficient paths.
The mole follows where it found "food" (signal) last time, plus exploration.

Key optimization: Store A@Q (small) instead of A (large) in forward pass.
Backward only needs A@Q and B, never full A. Massive activation memory savings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.amp.autocast_mode import custom_fwd, custom_bwd


class MoleGradFn(torch.autograd.Function):
    """Low-rank gradient via projected activations."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        Q: torch.Tensor,
        mole_grad: torch.Tensor,
    ) -> torch.Tensor:
        x_flat = x.view(-1, x.shape[-1])
        Q_cast = Q.to(dtype=x.dtype)
        AQ = x_flat @ Q_cast
        ctx.save_for_backward(weight, AQ, mole_grad)
        output = F.linear(x, weight, None)
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        weight, AQ, mole_grad = ctx.saved_tensors

        # 1. Gradient w.r.t Input
        grad_output_flat = grad_output.view(-1, grad_output.shape[-1])
        grad_input = grad_output.matmul(weight)

        # 2. Gradient w.r.t Weight (Low Rank)
        mole_grad_update = grad_output_flat.transpose(0, 1) @ AQ

        # Accumulate gradient in-place (Dynamo-friendly)
        mole_grad.add_(mole_grad_update)

        return grad_input, None, None, None


class MoleLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with low-rank gradient updates.

    Core concept: Weight updates live in low-rank subspace estimated from
    activations A and output gradients B without forming full G = BᵀA.

    Key optimization: Store A@Q (small) not A (large) during forward.
    Memory scales with rank_k, not input dimension or sequence length.

    The "mole" metaphor: Q is the tunnel direction. Each step, it follows
    where signal was found (V from SVD) plus random exploration. Over many
    steps, the mole covers the full weight space while staying efficient.

    Features:
    - Activation memory: O(n × k) instead of O(n × d)
    - Gradient memory: O(d × k) instead of O(d × d)
    - V from current step becomes Q for next step (signal following)
    - Exploration noise prevents collapse to fixed subspace
    """

    def __init__(self, in_features: int, out_features: int, rank_k: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank_k = rank_k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = None
        # Register buffer with placeholder — will be initialized properly in init_weights
        self.register_buffer("Q", torch.empty(in_features, rank_k))
        # Gradient accumulator buffer (persistent=False so it's not saved in checkpoints)
        self.register_buffer(
            "mole_grad", torch.zeros(out_features, rank_k), persistent=False
        )

    def init_Q(self):
        """Initialize Q with random orthonormal columns. Call after model is on device."""
        device = self.weight.device
        dtype = self.weight.dtype
        Q = torch.randn(
            self.in_features, self.rank_k, device=device, dtype=torch.float32
        )
        Q, _ = torch.linalg.qr(Q, mode="reduced")

        # Check for NaNs in initialization
        if torch.isnan(Q).any():
            print(
                f"WARNING: NaN detected in Q initialization for {self.extra_repr()}. Retrying with noise."
            )
            Q = (
                torch.randn(
                    self.in_features, self.rank_k, device=device, dtype=torch.float32
                )
                + 1e-6
            )
            Q, _ = torch.linalg.qr(Q, mode="reduced")

        self.Q.copy_(Q.to(dtype))

        # Reset gradient accumulator
        self.mole_grad.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return MoleGradFn.apply(x, self.weight, self.Q, self.mole_grad)
        else:
            return F.linear(x, self.weight, None)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank_k={self.rank_k}"
