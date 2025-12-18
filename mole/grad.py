"""
Low-rank autograd function for MoleGrad.

Core principle: Weight updates live in a low-rank subspace. For each large
Linear, we estimate the dominant singular modes from activations `A` and
output grads `B` without forming the dense gradient `G = BᵀA`.

Key optimization: store only the projected activations `A @ Q` (small) in the
forward pass. Backward then needs `A @ Q` and `B`, never the full activations.
This preserves the descriptive context that motivated MoleGrad while keeping
the new Gradient-Gated subspace evolution flow.
"""

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import custom_fwd, custom_bwd

__all__ = ["MoleGradFn"]


class MoleGradFn(torch.autograd.Function):
    """Connects dense forward with low-rank backward."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, AQ: torch.Tensor, mole_grad: torch.Tensor) -> torch.Tensor:
        # AQ is pre-computed in MoleLinear to allow fusion with Z-sampling
        ctx.save_for_backward(weight, AQ, mole_grad)
        return F.linear(x, weight, None)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        weight, AQ, mole_grad = ctx.saved_tensors

        # Dense gradient for inputs
        grad_input = grad_output.matmul(weight)

        # Low-rank gradient accumulation for weights
        grad_output_flat = grad_output.view(-1, grad_output.shape[-1])
        torch.addmm(input=mole_grad, mat1=grad_output_flat.T, mat2=AQ, beta=1.0, alpha=1.0, out=mole_grad)

        return grad_input, None, None, None
