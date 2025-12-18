"""
MoleLinear with Gradient-Gated Subspace Evolution.

Concept: weight updates live in a low-rank subspace estimated from activations
`A` and output gradients `B` without forming the full `G = BᵀA`. Store `A @ Q`
(small) instead of `A` (large) to cut activation memory. The mole follows where
it found signal (gradient utility) while using fresh data pull `Z = Xᵀ (X Q)`
to overwrite stale directions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grad import MoleGradFn

__all__ = ["MoleLinear"]


class MoleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank_k: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank_k = rank_k

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = None

        # Subspace state and accumulators
        self.register_buffer("Q", torch.empty(in_features, rank_k))
        self.register_buffer("mole_grad", torch.zeros(out_features, rank_k), persistent=False)
        # subspace_z removed

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self._reset_subspace()

    def _reset_subspace(self):
        Q_init = torch.randn(self.in_features, self.rank_k, device=self.weight.device, dtype=torch.float32)
        Q_init, _ = torch.linalg.qr(Q_init, mode="reduced")
        Q_init = F.normalize(Q_init, dim=0)
        self.Q.copy_(Q_init.to(self.weight.dtype))
        self.mole_grad.zero_()

    def init_Q(self):
        """Compatibility alias for resetting the subspace buffers."""
        self._reset_subspace()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 1. Compute AQ for backward (needed for gradient projection)
            x_flat = x.view(-1, x.shape[-1])
            Q_cast = self.Q if self.Q.dtype == x.dtype else self.Q.to(x.dtype)
            AQ = x_flat @ Q_cast

            # Z-estimation removed: We now use "Scout Renewal" strategy in optimizer
            # which relies purely on gradient utility (eigenvalues of Y^T Y)

            return MoleGradFn.apply(x, self.weight, AQ, self.mole_grad)
        return nn.functional.linear(x, self.weight, None)

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, k={self.rank_k}'
