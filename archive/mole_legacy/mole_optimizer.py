"""
MoleOptimizer: Apply factored low-rank gradients via eigh decomposition.

Design:
- MoleLinear.backward() stores Y (gradient in subspace) and Q (subspace basis)
- This optimizer does eigh on Y.T @ Y to get singular vectors/values
- Then applies: W -= lr * US @ V.T via addmm_ (no dense grad materialization)
- Updates Q to follow the signal (V-following with exploration)

Why eigh instead of SVD:
- For Y of shape (out_dim, k), SVD is O(out_dim * k²)
- eigh on Y.T @ Y is O(k³) plus two GEMMs - faster for small k
- Math: Y = U S Vh => Y.T Y = V S² V.T => eigh gives V and S²
- Key identity: U S = Y @ V (avoids computing U explicitly)

Batching optimization:
- Groups parameters by (out_dim, in_dim) shape
- Batches eigh across same-shaped layers (e.g., all 12 c_q layers together)
- Single batched eigh call instead of 72 individual calls
"""

import torch
from collections import defaultdict

# Exploration noise scale for Q evolution
# Small noise prevents Q from collapsing to a fixed subspace
exploration_noise: float = 0.1


def _batched_eigh_update(
    modules: list,
    Ys: list,
    Qs: list,
    lr: float,
    scales: list,
):
    """
    Batched eigh update for modules with identical weight shapes.

    All modules must have same (out_dim, in_dim) weight shape.
    Batches the k×k eigh calls for massive speedup.
    """
    # Stack Ys and Qs: [batch, out_dim, k] and [batch, in_dim, k]
    Y_batch = torch.stack(Ys)  # (B, out_dim, k)
    Q_batch = torch.stack(Qs)  # (B, in_dim, k)

    # 1. Batched covariance: C = Y.T @ Y for each in batch
    C_batch = torch.bmm(Y_batch.transpose(-2, -1), Y_batch)  # (B, k, k)

    # 2. Batched eigendecomposition in float32
    S_sq_batch, V_eig_batch = torch.linalg.eigh(C_batch.float())  # (B, k), (B, k, k)

    # 3. Singular values (unused for now, but available)
    # S_batch = torch.sqrt(torch.clamp(S_sq_batch, min=0))  # (B, k)

    # 4. Cast V_eig back to compute dtype
    V_eig_batch = V_eig_batch.to(Y_batch.dtype)  # (B, k, k)

    # 5. Batched US = Y @ V_eig and V = Q @ V_eig
    US_batch = torch.bmm(Y_batch, V_eig_batch)  # (B, out_dim, k)
    V_batch = torch.bmm(Q_batch.to(Y_batch.dtype), V_eig_batch)  # (B, in_dim, k)

    # 6. Apply updates and update Q for each module
    sum_update_norm = 0.0
    max_update_norm = 0.0

    for i, (m, scale) in enumerate(zip(modules, scales)):
        US = US_batch[i].to(m.weight.dtype)
        V = V_batch[i].to(m.weight.dtype)

        # W -= lr * scale * US @ V.T
        update = torch.mm(US, V.T)
        m.weight.data.add_(update, alpha=-lr * scale)

        # Log stats occasionally (e.g. if we had a logger passed in, but for now just print if huge)
        update_norm = update.norm().item()
        sum_update_norm += update_norm
        if update_norm > max_update_norm:
            max_update_norm = update_norm

        if update_norm > 10.0:
            print(
                f"MoleOptimizer: Large update norm {update_norm:.2f} on {m.extra_repr()}"
            )

        # Update Q: V + noise, then QR orthonormalize
        Q_next = V + exploration_noise * torch.randn_like(V)
        Q_next, _ = torch.linalg.qr(Q_next.float(), mode="reduced")
        m.Q.copy_(Q_next.to(V.dtype))

        # Clear mole_grad accumulator
        m.mole_grad.zero_()

    return sum_update_norm, max_update_norm


class MoleOptimizer(torch.optim.Optimizer):
    """
    Optimizer for MoleLinear modules using eigh-based gradient factorization.

    Receives MoleLinear modules. Reads Y and Q from module,
    applies low-rank update via eigendecomposition of Y.T @ Y.

    Batches eigh calls across modules with same weight shape for efficiency.

    Args:
        modules: iterable of MoleLinear modules
        lr: learning rate
        aspect_ratio_scaling: if True, scale by sqrt(max(1, out_dim/in_dim))
    """

    def __init__(self, modules, lr: float = 0.02, aspect_ratio_scaling: bool = True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        self.modules = list(modules)
        self.metrics = {}

        # Collect weight params for the Optimizer base class
        params = [m.weight for m in self.modules]
        defaults = dict(lr=lr, aspect_ratio_scaling=aspect_ratio_scaling, initial_lr=lr)

        super().__init__([{"params": params}], defaults)

    @torch.no_grad()
    def step(self):
        """
        Apply low-rank gradient update using batched eigh decomposition.

        Groups modules by weight shape and processes each group in a single batched call.
        """
        total_Y_norm = 0.0
        total_update_norm = 0.0
        max_update_norm = 0.0
        count = 0

        for group in self.param_groups:
            lr = group["lr"]
            aspect_ratio_scaling = group["aspect_ratio_scaling"]
            shape_to_data = defaultdict(
                lambda: {"modules": [], "Ys": [], "Qs": [], "scales": []}
            )

            for m in self.modules:
                Y = m.mole_grad
                Q = m.Q

                if Y is None or Q is None:
                    raise RuntimeError(
                        f"MoleLinear gradient incomplete — missing mole_grad or Q on {m}"
                    )

                # Track Y norm
                total_Y_norm += Y.norm().item()
                count += 1

                out_dim, in_dim = m.out_features, m.in_features
                scale = (
                    (max(1.0, out_dim / in_dim) ** 0.5) if aspect_ratio_scaling else 1.0
                )

                shape = (out_dim, in_dim)
                shape_to_data[shape]["modules"].append(m)
                shape_to_data[shape]["Ys"].append(Y)
                shape_to_data[shape]["Qs"].append(Q)
                shape_to_data[shape]["scales"].append(scale)

            # Process each shape group with batched eigh
            for shape, data in shape_to_data.items():
                sum_norm, max_norm = _batched_eigh_update(
                    data["modules"],
                    data["Ys"],
                    data["Qs"],
                    lr,
                    data["scales"],
                )
                total_update_norm += sum_norm
                max_update_norm = max(max_update_norm, max_norm)

        # Store metrics
        self.metrics = {
            "mole/Y_norm": total_Y_norm / max(1, count),
            "mole/update_norm": total_update_norm / max(1, count),
            "mole/max_update_norm": max_update_norm,
        }
