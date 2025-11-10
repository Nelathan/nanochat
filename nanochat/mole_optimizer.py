"""
MoleOptimizer: Apply factored low-rank gradients efficiently (no materialization).

Design:
- MoleLinear.backward() stashes (U, S, V) and trust_scale in parameter._mole_state
- This optimizer reads those and applies weight.addmm_ without forming dense grads
"""

import torch
from typing import Optional

class MoleOptimizer(torch.optim.Optimizer):
    """
    Minimal optimizer for MoleLinear layers that applies factored gradients.

    When weight.grad is a MoleGrad object (U, S, V), applies the update:
        weight -= lr * scale * (U @ diag(S) @ V.T)
    using torch.addmm_ without materializing the full gradient.

    No momentum, no preconditioning - the low-rank structure IS the preconditioning.

    Args:
        params: iterable of MoleLinear parameters
        lr: learning rate
        aspect_ratio_scaling: if True, scale by sqrt(max(1, out_dim/in_dim))
    """

    def __init__(self, params, lr: float = 0.02, aspect_ratio_scaling: bool = True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, aspect_ratio_scaling=aspect_ratio_scaling)

        # Group params by shape for efficient batched operations
        params = list(params)
        assert all(p.ndim == 2 for p in params), "MoleOptimizer expects MoleLinear weights"

        param_groups = []
        shapes = sorted({p.shape for p in params})
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            param_groups.append(dict(params=group_params))

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        """Apply factored low-rank gradients efficiently from parameter state."""
        for group in self.param_groups:
            lr = group["lr"]
            aspect_ratio_scaling = group["aspect_ratio_scaling"]

            for p in group["params"]:
                # Optional aspect ratio scaling (matching Muon's strategy)
                if aspect_ratio_scaling:
                    scale = (max(1.0, p.size(0) / p.size(1)) ** 0.5)
                else:
                    scale = 1.0

                state = getattr(p, "_mole_state", None)
                if state is None or getattr(state, "last_USV", None) is None:
                    # Nothing to apply for this parameter (could be non-MoleLinear)
                    continue

                U, S, V = state.last_USV
                trust_scale: Optional[torch.Tensor] = getattr(state, "trust_scale", None)
                tscale = trust_scale if trust_scale is not None else torch.tensor(1.0, device=p.device, dtype=torch.float32)

                # Bring factors to param device/dtype
                Ud = U.to(device=p.device, dtype=p.dtype)
                Sd = S.to(device=p.device, dtype=p.dtype) * tscale.to(device=p.device, dtype=p.dtype)
                Vd = V.to(device=p.device, dtype=p.dtype)

                # Efficient factored update: weight -= lr * scale * (U @ diag(S) @ V.T)
                US = Ud * Sd.unsqueeze(0)
                p.addmm_(US, Vd.T, alpha=-lr * scale)

                # Clear state for next step
                state.last_USV = None
                state.trust_scale = None
