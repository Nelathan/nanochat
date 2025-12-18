"""
MoleOptimizer: Apply factored low-rank gradients via eigh decomposition.

Design:
- MoleLinear.backward() stores Y (gradient in subspace) and Q (subspace basis)
- This optimizer does eigh on Y.T @ Y to get singular vectors/values
- Then applies: W -= lr * US @ V.T via addmm_ (no dense grad materialization)
- Updates Q to follow the signal (Scout Renewal strategy)

Why eigh instead of SVD:
- For Y of shape (out_dim, k), SVD is O(out_dim * k²)
- eigh on Y.T @ Y is O(k³) plus two GEMMs - faster for small k
- Math: Y = U S Vh => Y.T Y = V S² V.T => eigh gives V and S²

Subspace Evolution (Scout Renewal):
- Elites (Top k-1): Kept and aligned with gradient signal.
- Scout (Bottom 1): Replaced with random orthogonal vector to explore new directions.
"""

import torch
import torch.nn.functional as F
from collections import defaultdict

__all__ = ["MoleOptimizer"]

# Subspace evolution hyperparameters
QR_EVERY_N_STEPS = 10

# Pointwise subspace modulation
# Effective per-mode scale: erf(alpha * s) / (alpha * s)
# where s = sqrt(eigenvalue) is the singular value of Y in that mode.
ERF_ALPHA = 1.0
ERF_EPS = 1e-20


# If batched eigh ever fails, retry once with diagonal jitter scaled to trace.
EIGH_JITTER_RETRY_SCALE = 1e-6


def _mole_eigh_core(C_batch: torch.Tensor, Y_batch: torch.Tensor, Q_batch: torch.Tensor):
    """Core math for MoleGrad update: eigh + two GEMMs + erf modulation.

    Kept as a standalone function so it can be optionally wrapped by torch.compile.
    """

    S_eig_batch, V_eig_batch = torch.linalg.eigh(C_batch)

    V_eig_batch = V_eig_batch.to(Y_batch.dtype)
    S_f = S_eig_batch.float().clamp(min=0.0)

    US_batch = torch.bmm(Y_batch, V_eig_batch)
    V_batch = torch.bmm(Q_batch.to(Y_batch.dtype), V_eig_batch)

    s = torch.sqrt(S_f + ERF_EPS)
    scale_cols = torch.special.erf(ERF_ALPHA * s) / (ERF_ALPHA * s + ERF_EPS)
    US_batch = US_batch * scale_cols.to(US_batch.dtype).unsqueeze(1)

    update_batch = torch.bmm(US_batch, V_batch.transpose(-2, -1))
    return update_batch, V_batch


def _mole_batched_update(modules, Ys, Qs, lr, scales, step_count, state_dict, compiled_core=None):
    """Process a batch of same-shaped layers with eigendecomposition and Q evolution."""

    # Fast path: batched eigh on (B, k, k). If it fails, retry with small jitter and log.
    Y_batch = torch.stack(Ys)  # (B, out, k)
    Q_batch = torch.stack(Qs)  # (B, d, k)

    # Compute Y norms in-batch to avoid per-module .item() sync overhead.
    y_norms = Y_batch.float().norm(dim=(-2, -1))
    sum_Y_norm = float(y_norms.sum().item())
    min_Y_norm = float(y_norms.min().item())
    count = int(Y_batch.shape[0])

    C_batch = torch.bmm(Y_batch.transpose(-2, -1), Y_batch).float()  # (B, k, k)
    C_batch = 0.5 * (C_batch + C_batch.transpose(-2, -1))

    core = compiled_core if compiled_core is not None else _mole_eigh_core

    try:
        update_batch, V_batch = core(C_batch, Y_batch, Q_batch)
    except torch._C._LinAlgError as exc:  # type: ignore[attr-defined]
        # Log minimal diagnostics to console, then retry with jitter.
        try:
            diag = C_batch.diagonal(dim1=-2, dim2=-1)
            print("[MoleGrad] batched eigh failed; retrying with jitter")
            print(f"  error={exc}")
            print(f"  batch={C_batch.shape[0]} k={C_batch.shape[-1]}")
            print(f"  diag_min={float(diag.min().item()):.3e} diag_max={float(diag.max().item()):.3e}")
        except Exception:
            pass

        # First retry: tiny jitter proportional to trace.
        k = C_batch.shape[-1]
        eye = torch.eye(k, device=C_batch.device, dtype=C_batch.dtype).unsqueeze(0)
        trace = C_batch.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)  # (B, 1)
        base = (trace / max(1, k)).clamp(min=1e-12)
        jitter = (EIGH_JITTER_RETRY_SCALE * base).unsqueeze(-1) * eye
        update_batch, V_batch = core(C_batch + jitter, Y_batch, Q_batch)

    update_norms = update_batch.float().norm(dim=(-2, -1))
    sum_update_norm = float(update_norms.sum().item())
    max_update_norm = float(update_norms.max().item())

    # "Update norm" is the norm of the unscaled update matrix (pre lr * aspect scaling).
    # For LR/"effective step size" comparisons, also track the true applied deltaW norms.
    scale_t = torch.tensor(scales, device=update_norms.device, dtype=update_norms.dtype)
    deltaW_norms = update_norms * (float(lr) * scale_t)
    sum_deltaW_norm = float(deltaW_norms.sum().item())
    max_deltaW_norm = float(deltaW_norms.max().item())

    # Apply weight updates (per-module due to separate Parameter storages)
    for i, (m, scale) in enumerate(zip(modules, scales)):
        update = update_batch[i].to(m.weight.dtype)
        m.weight.data.add_(update, alpha=-lr * scale)

    # Update Q (Scout Renewal) in batch
    Q_next_batch = V_batch.to(Q_batch.dtype).clone()

    R = torch.randn(
        Q_next_batch.shape[0], Q_next_batch.shape[1], 1,
        device=Q_next_batch.device, dtype=Q_next_batch.dtype,
    )
    V_elites = Q_next_batch[:, :, 1:]
    R_proj = torch.bmm(V_elites, torch.bmm(V_elites.transpose(-2, -1), R))
    R_perp = F.normalize(R - R_proj, dim=1, eps=1e-12)
    Q_next_batch[:, :, 0:1] = R_perp

    if step_count % QR_EVERY_N_STEPS == 0:
        Q_next_batch, _ = torch.linalg.qr(Q_next_batch.float(), mode="reduced")
        Q_next_batch = Q_next_batch.to(Q_batch.dtype)
    else:
        Q_next_batch = F.normalize(Q_next_batch, dim=1, eps=1e-12)

    for i, m in enumerate(modules):
        m.Q.copy_(Q_next_batch[i])
        m.mole_grad.zero_()

    return sum_Y_norm, min_Y_norm, count, sum_update_norm, max_update_norm, sum_deltaW_norm, max_deltaW_norm


class MoleOptimizer(torch.optim.Optimizer):
    def __init__(self, modules, lr: float = 0.02, aspect_ratio_scaling: bool = True):
        self.modules = list(modules)
        params = [m.weight for m in self.modules]
        defaults = dict(lr=lr, aspect_ratio_scaling=aspect_ratio_scaling)
        super().__init__([{"params": params}], defaults)
        self.metrics = {}
        self.step_count = 0

        self._shape_groups_cache = {}

        # Always try to compile the core (if available). This keeps step() clean and fast.
        # If compile isn't available (or fails), we silently fall back to eager.
        self._compiled_core = None
        if hasattr(torch, "compile"):
            try:
                self._compiled_core = torch.compile(_mole_eigh_core, fullgraph=True)
            except Exception:
                self._compiled_core = None

    def _get_shape_groups(self, ar_scale: bool):
        cached = self._shape_groups_cache.get(ar_scale)
        if cached is not None:
            return cached

        shape_groups = defaultdict(lambda: {"modules": [], "scales": []})
        for m in self.modules:
            out_d, in_d = m.out_features, m.in_features
            scale = (max(1.0, out_d / in_d) ** 0.5) if ar_scale else 1.0
            shape_groups[(out_d, in_d)]["modules"].append(m)
            shape_groups[(out_d, in_d)]["scales"].append(scale)

        shape_groups = dict(shape_groups)
        self._shape_groups_cache[ar_scale] = shape_groups
        return shape_groups

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        total_Y_norm = 0.0
        min_Y_norm = float("inf")
        total_update_norm = 0.0
        max_update_norm = 0.0
        total_deltaW_norm = 0.0
        max_deltaW_norm = 0.0
        count = 0

        if len(self.param_groups) != 1:
            raise RuntimeError("MoleOptimizer expects a single param_group")

        group = self.param_groups[0]
        lr = group["lr"]
        ar_scale = group["aspect_ratio_scaling"]

        shape_groups = self._get_shape_groups(ar_scale)

        compiled_core = self._compiled_core

        for shape, data in shape_groups.items():
            modules = data["modules"]
            Ys = [m.mole_grad for m in modules]
            Qs = [m.Q for m in modules]
            scales = data["scales"]

            sum_Y, min_Y, n, sum_norm, max_norm, sum_dW, max_dW = _mole_batched_update(
                modules,
                Ys,
                Qs,
                lr,
                scales,
                self.step_count,
                self.state,
                compiled_core=compiled_core,
            )

            total_Y_norm += sum_Y
            min_Y_norm = min(min_Y_norm, min_Y)
            count += n

            total_update_norm += sum_norm
            max_update_norm = max(max_update_norm, max_norm)

            # Track true applied deltaW norms (includes lr + aspect scaling)
            # This is the closest analogue to how "big" an optimizer step was.
            total_deltaW_norm += sum_dW
            max_deltaW_norm = max(max_deltaW_norm, max_dW)

        avg_Y_norm = total_Y_norm / max(1, count)
        avg_update_norm = total_update_norm / max(1, count)
        avg_deltaW_norm = total_deltaW_norm / max(1, count)
        eps = 1e-8
        cond_ratio = avg_update_norm / (avg_Y_norm + eps)

        self.metrics = {
            "mole/Y_norm": avg_Y_norm,
            "mole/min_Y_norm": min_Y_norm if count > 0 else 0.0,
            "mole/update_norm": avg_update_norm,
            "mole/max_update_norm": max_update_norm,
            "mole/deltaW_norm": avg_deltaW_norm,
            "mole/max_deltaW_norm": max_deltaW_norm,
            "mole/cond_ratio": cond_ratio,
        }
