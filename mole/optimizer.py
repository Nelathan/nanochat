"""
MoleOptimizer: Apply factored low-rank gradients via eigh decomposition.

Design:
- MoleLinear.backward() stores Y (gradient in subspace) and Q (subspace basis)
- This optimizer does eigh on Y.T @ Y to get singular vectors/values
- Then applies: W -= lr * US @ V.T via addmm_ (no dense grad materialization)
- Updates Q to follow the signal (Scout Renewal strategy with Mbar-driven transport)

Why eigh instead of SVD:
- For Y of shape (out_dim, k), SVD is O(out_dim * k²)
- eigh on Y.T @ Y is O(k³) plus two GEMMs - faster for small k
- Math: Y = U S Vh => Y.T Y = V S² V.T => eigh gives V and S²

Subspace Evolution (Scout Renewal with Mbar transport):
- Maintain Mbar: EMA of Y^T Y in current Q-basis, transported under basis rotations.
- Diagonalize Mbar via eigh to find stable modes, rotate Q into this basis.
- Scout: Replace weakest mode (lowest Mbar eigenvalue) with random orthogonal vector.
- Scout prior: Initialize new scout's variance entry to median of existing eigenvalues.
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
import math

__all__ = ["MoleOptimizer"]

# Subspace evolution hyperparameters
QR_EVERY_N_STEPS = 50

# subspace modulation (erf compressor)
# We apply erf to the singular values to softly compress the spectrum.
# Since `US = U S`, to obtain `U f(S)` we scale columns by `f(s)/s`.
ERF_ALPHA = 1.0
ERF_EPS = 1e-20

# Mbar EMA decay:
MBAR_BETA = 0.99

# Step size hyperparameter
# Applied step size:
#   alpha = lr * tau * target_std * sqrt(k)
# where
#   fan_out = weight.size(0)
#   fan_in  = weight.size(1)
#   target_std = 1/sqrt(fan_in) * min(1, sqrt(fan_out/fan_in))
TAU = 10.0

# If batched eigh ever fails, retry once with diagonal jitter scaled to trace.
EIGH_JITTER_RETRY_SCALE = 1e-6


def _mole_eigh_core(
    C_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    Q_batch: torch.Tensor,
    Mbar_batch: torch.Tensor,
):
    """Core math for MoleGrad update: eigh + two GEMMs + erf modulation.

    Returns:
        update_batch: (B, out, in) weight update matrices
        Vc_batch: (B, k, k) eigenvectors of C (for Q evolution)
        eig_C_batch: (B, k) eigenvalues of C (for metrics)
        damp: (B, k) dampening factors for metrics
    """
    k = C_batch.shape[-1]
    eig_C_batch, Vc_batch = torch.linalg.eigh(C_batch)

    eig_f = eig_C_batch.clamp(min=0.0)
    US_batch = torch.bmm(Y_batch, Vc_batch)
    V_batch = torch.bmm(Q_batch, Vc_batch)

    # Adafactor-style dampening in subspace (energy-aware):
    # For each current-step eigenmode v (column of Vc), estimate its variance under EMA(Mbar)
    #   var = v^T Mbar v
    # With Mbar ~= EMA(Y^T Y), var has units of s^2. Using 1/sqrt(var*k) makes (s*damp*sqrt(k))
    # dimensionless and near-constant (~1) when Mbar is aligned with the current spectrum.
    T = torch.bmm(Mbar_batch, Vc_batch)  # (B, k, k)
    var_diag = (Vc_batch * T).sum(dim=1).clamp(min=0.0)  # (B, k)
    damp = 1.0 / torch.sqrt(var_diag * float(k) + ERF_EPS)

    s = torch.sqrt(eig_f + ERF_EPS)  # (B, k)

    # erf sees a dimensionless argument; no lr/tau here.
    # For matched scales, s*damp*sqrt(k) ~= 1.
    erf_arg = ERF_ALPHA * (s * damp) * (float(k) ** 0.5)
    scale_cols = torch.special.erf(erf_arg) / (s + ERF_EPS)
    US_batch = US_batch * scale_cols.to(US_batch.dtype).unsqueeze(1)

    update_batch = torch.bmm(US_batch, V_batch.transpose(-2, -1))
    return update_batch, Vc_batch, eig_C_batch, damp


def _mole_batched_update(
    modules, Ys, Qs, lr, step_count, state_dict,
    out_d, in_d, k, compiled_core=None
):
    """Process a batch of same-shaped layers with eigendecomposition and Mbar-driven Q evolution."""
    B = len(modules)
    device = Ys[0].device

    # Stack Y and Q for batched ops
    Y_batch = torch.stack(Ys).float()  # (B, out, k)
    Q_batch = torch.stack(Qs).float()  # (B, in, k)

    # No training wheels: use raw C = Y^T Y
    C_batch = torch.bmm(Y_batch.transpose(-2, -1), Y_batch)  # (B, k, k)
    C_batch = 0.5 * (C_batch + C_batch.transpose(-2, -1))  # symmetrize

    # --- Gather/init per-module state: Mbar ---
    Mbar_list = []
    for m in modules:
        st = state_dict.get(m.weight)
        if st is None:
            st = {}
            state_dict[m.weight] = st
        # Mbar: (k, k) fp32, init to eye(k)/k so trace = 1 (mu_trace starts at 1).
        # This gives var_diag*k = 1 → damp = 1 initially (neutral, neither boost nor dampen).
        # Mbar will adapt to actual C scale within ~100 steps (beta=0.99).
        if "Mbar" not in st:
            st["Mbar"] = torch.eye(k, device=device, dtype=torch.float32) / float(k)
        Mbar_list.append(st["Mbar"])

    Mbar_batch = torch.stack(Mbar_list)  # (B, k, k) fp32

    # --- Compute update using current C (step 2) ---
    core = compiled_core if compiled_core is not None else _mole_eigh_core

    # --- Step size (new definition) ---
    # alpha = lr * tau * target_std * sqrt(k)
    # target_std = 1/sqrt(fan_in) * min(1, sqrt(fan_out/fan_in))
    fan_out = float(out_d)
    fan_in = float(in_d)
    target_std = (1.0 / math.sqrt(fan_in)) * min(1.0, math.sqrt(fan_out / fan_in))
    alpha_scalar = float(lr) * TAU * target_std * math.sqrt(float(k))

    try:
        update_batch, Vc_batch, eig_C_batch, damp = core(C_batch, Y_batch, Q_batch, Mbar_batch)
    except RuntimeError as exc:
        # Retry with jitter
        try:
            diag = C_batch.diagonal(dim1=-2, dim2=-1)
            print("[MoleGrad] batched eigh failed; retrying with jitter")
            print(f"  error={exc}")
            print(f"  batch={B} k={k}")
            print(f"  diag_min={float(diag.min().item()):.3e} diag_max={float(diag.max().item()):.3e}")
        except Exception:
            pass

        eye = torch.eye(k, device=C_batch.device, dtype=C_batch.dtype).unsqueeze(0)
        trace = C_batch.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        base = (trace / max(1, k)).clamp(min=1e-12)
        jitter = (EIGH_JITTER_RETRY_SCALE * base).unsqueeze(-1) * eye
        update_batch, Vc_batch, eig_C_batch, damp = core(C_batch + jitter, Y_batch, Q_batch, Mbar_batch)

    # --- EMA update of Mbar in current coordinates (step 3) ---
    # Keep Mbar magnitude: it should represent EMA energy in the subspace.
    Mbar_batch = MBAR_BETA * Mbar_batch + (1 - MBAR_BETA) * C_batch

    # --- Diagonalize Mbar to define stable modes (step 4) ---
    # Scale Mbar for eigh (eigenvectors unchanged under scalar scaling), to avoid tiny-scale
    # numerical issues when the layer is near-zero-init.
    Mbar_trace = Mbar_batch.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    Mbar_for_eigh = Mbar_batch / Mbar_trace.clamp(min=1e-12)

    try:
        mu_batch, Vm_batch = torch.linalg.eigh(Mbar_for_eigh)  # ascending
    except RuntimeError:
        # Fallback: add jitter
        print("[MoleGrad] Mbar eigh failed; retrying with jitter")
        eye = torch.eye(k, device=Mbar_for_eigh.device, dtype=Mbar_for_eigh.dtype).unsqueeze(0)
        mu_batch, Vm_batch = torch.linalg.eigh(Mbar_for_eigh + 1e-6 * eye)

    # R_batch = Vm_batch is the rotation to apply to Q
    R_batch = Vm_batch  # (B, k, k)

    # Rotate Q into the Mbar eigenbasis
    Q_next_batch = torch.bmm(Q_batch, R_batch)  # (B, in, k)

    # Transport Mbar into new basis: R^T @ Mbar @ R (becomes diagonal by construction)
    Mbar_next_batch = torch.bmm(
        R_batch.transpose(-2, -1),
        torch.bmm(Mbar_batch, R_batch)
    )  # (B, k, k)

    # --- Scout replacement: weakest mode by Mbar eigenvalue (step 5) ---
    # mu_batch is ascending, so index 0 is weakest
    # Replace column 0 with random orthogonal vector

    # Generate random vector and project out all other columns
    R_scout = torch.randn(B, in_d, 1, device=device, dtype=Q_next_batch.dtype)
    V_elites = Q_next_batch[:, :, 1:]  # (B, in, k-1)
    R_proj = torch.bmm(V_elites, torch.bmm(V_elites.transpose(-2, -1), R_scout))
    R_perp = F.normalize(R_scout - R_proj, dim=1, eps=1e-12)
    Q_next_batch[:, :, 0:1] = R_perp

    # Reset Mbar for scout: use the mean of the ACTUAL Mbar diagonal (unnormalized)
    # so the scout starts at the same energy scale as the other modes.
    mbar_diag = Mbar_next_batch.diagonal(dim1=-2, dim2=-1)  # (B, k)
    mu_prior_real = mbar_diag.mean(dim=-1, keepdim=True).clamp(min=1e-12)  # (B, 1)
    Mbar_next_batch[:, 0, :] = 0
    Mbar_next_batch[:, :, 0] = 0
    Mbar_next_batch[:, 0, 0] = mu_prior_real.squeeze(-1)

    # --- Periodic QR for orthonormality ---
    if step_count % QR_EVERY_N_STEPS == 0:
        Q_next_batch, _ = torch.linalg.qr(Q_next_batch, mode="reduced")
    else:
        Q_next_batch = F.normalize(Q_next_batch, dim=1, eps=1e-12)

    alpha = alpha_scalar

    # --- Apply weight updates ---
    for i, m in enumerate(modules):
        m.weight.data.add_(update_batch[i].to(m.weight.dtype), alpha=-alpha)

    # --- Write back state ---
    for i, m in enumerate(modules):
        m.mole_Q.copy_(Q_next_batch[i].to(m.mole_Q.dtype))
        m.mole_Y.zero_()
        state_dict[m.weight]["Mbar"] = Mbar_next_batch[i]

    # --- Compute metrics (minimal set) ---
    # Q orthonormality error
    QtQ = torch.bmm(Q_next_batch.transpose(-2, -1), Q_next_batch)  # (B, k, k)
    eye_k = torch.eye(k, device=device, dtype=QtQ.dtype).unsqueeze(0)
    Q_ortho_err = (QtQ - eye_k).norm(dim=(-2, -1))  # (B,)

    # Effective rank + concentration diagnostics from ACTUAL Mbar diagonal (unnormalized)
    mbar_diag_final = Mbar_next_batch.diagonal(dim1=-2, dim2=-1)  # (B, k)
    mu_sum_m = mbar_diag_final.sum(dim=-1)  # (B,) - this is trace(Mbar)
    mu_sq_sum = (mbar_diag_final ** 2).sum(dim=-1)  # (B,)
    r_eff = (mu_sum_m ** 2) / (mu_sq_sum + 1e-12)  # (B,)
    mu_max = mbar_diag_final.max(dim=-1).values  # (B,)
    mu_max_frac = mu_max / (mu_sum_m + 1e-12)

    # Step multiplier (alpha / lr), constant within this shape group
    step_mult = torch.full((B,), TAU * target_std * math.sqrt(float(k)), device=device, dtype=torch.float32)

    # Useful for debugging scale drift now that we track raw Y.
    y_fro = Y_batch.norm(dim=(-2, -1))  # (B,)

    # Update norms
    update_norms = update_batch.float().norm(dim=(-2, -1))  # (B,)
    deltaW_norms = update_norms * alpha  # (B,)

    metrics = {
        "count": B,
        "Q_ortho_err_sum": Q_ortho_err.sum(),
        "r_eff_sum": r_eff.sum(),
        "mu_sum": mu_sum_m.sum(),
        "mu_max_frac_sum": mu_max_frac.sum(),
        "sum_update_norm": update_norms.sum(),
        "sum_deltaW_norm": deltaW_norms.sum(),
        "step_mult_sum": step_mult.sum(),
        "y_fro_sum": y_fro.sum(),
        "alpha": torch.as_tensor(alpha, device=device, dtype=torch.float32),
        "target_std": torch.as_tensor(target_std, device=device, dtype=torch.float32),
        "ada_damp_sum": damp.float().mean(dim=-1).sum(),
    }
    return metrics


class MoleOptimizer(torch.optim.Optimizer):
    def __init__(self, modules, lr: float = 0.02, aspect_ratio_scaling: bool = True):
        self.modules = list(modules)
        params = [m.weight for m in self.modules]
        defaults = dict(lr=lr)
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

    def _get_shape_groups(self):
        cached = self._shape_groups_cache.get("shape_groups")
        if cached is not None:
            return cached

        shape_groups = defaultdict(lambda: {"modules": []})
        for m in self.modules:
            out_d, in_d = m.out_features, m.in_features
            shape_groups[(out_d, in_d)]["modules"].append(m)

        shape_groups = dict(shape_groups)
        self._shape_groups_cache["shape_groups"] = shape_groups
        return shape_groups

    @torch.no_grad()
    def step(self):
        self.step_count += 1

        if len(self.param_groups) != 1:
            raise RuntimeError("MoleOptimizer expects a single param_group")

        group = self.param_groups[0]
        lr = group["lr"]

        shape_groups = self._get_shape_groups()
        compiled_core = self._compiled_core

        # Collect metrics from all shape groups (tensors, no sync yet)
        all_metrics = []
        shape_keys = []

        for (out_d, in_d), data in shape_groups.items():
            modules = data["modules"]
            k = modules[0].rank_k
            Ys = [m.mole_Y for m in modules]
            Qs = [m.mole_Q for m in modules]

            metrics = _mole_batched_update(
                modules, Ys, Qs, lr,
                self.step_count, self.state,
                out_d, in_d, k,
                compiled_core=compiled_core,
            )
            all_metrics.append(metrics)
            shape_keys.append((out_d, in_d))

        # Per-shape metrics only
        if not all_metrics:
            self.metrics = {}
            return

        per_shape = {}
        for (out_d, in_d), metrics in zip(shape_keys, all_metrics):
            shape_key = f"mole_{out_d}x{in_d}"
            count_s = max(1, metrics["count"])

            # Single sync per shape
            sync_tensor = torch.stack([
                metrics["Q_ortho_err_sum"],
                metrics["r_eff_sum"],
                metrics["mu_sum"],
                metrics["mu_max_frac_sum"],
                metrics["sum_update_norm"],
                metrics["sum_deltaW_norm"],
                metrics["step_mult_sum"],
                metrics["y_fro_sum"],
                metrics["alpha"], metrics["target_std"],
                metrics["ada_damp_sum"],
            ])
            vals = sync_tensor.tolist()

            per_shape[f"{shape_key}/Q_ortho_err"] = vals[0] / count_s
            per_shape[f"{shape_key}/eff_rank"] = vals[1] / count_s
            per_shape[f"{shape_key}/mu_trace"] = vals[2] / count_s
            per_shape[f"{shape_key}/mu_max_frac"] = vals[3] / count_s
            per_shape[f"{shape_key}/update_norm"] = vals[4] / count_s
            per_shape[f"{shape_key}/deltaW_norm"] = vals[5] / count_s
            per_shape[f"{shape_key}/step_mult"] = vals[6] / count_s
            per_shape[f"{shape_key}/y_fro"] = vals[7] / count_s
            per_shape[f"{shape_key}/alpha"] = vals[8]
            per_shape[f"{shape_key}/target_std"] = vals[9]
            per_shape[f"{shape_key}/ada_damp"] = vals[10] / count_s

        self.metrics = per_shape
