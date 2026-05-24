# Phase 2 — Block AttnRes Integration

**Hypothesis:** Replacing the standard residual stream with Block AttnRes (softmax over block-level cached representations + raw partial states within current block) improves loss by giving every layer global content-aware reach across depth. This is the structural bandwidth claim — narrow towers can scale deeper if the cache pathway becomes the explicit communication channel.

**Variable:** residual stream. Baseline = standard residual. Treatment = Block AttnRes with mean/avg compressor (per Kimi paper default).

**Inherits:** Phase 1 contract. GDN2 in place. All other hyperparameters locked.

## What changes from Phase 1

- Standard residual `x + f(x)` → Block AttnRes (softmax over N block representations + partial state)
- N = 6 blocks (24 layers / 4 layers per block)
- Block compression: simple mean of the 8 sub-block outputs per block (2 sub-blocks per transformer layer × 4 layers)
- 1 pseudo-query per layer (per the paper)
- 1 norm per block representation read (per paper)

## Implementation reference

- Flash-fused kernel: https://github.com/catswe/flash-attention-residuals
- Algorithm: Kimi paper §Block AttnRes
- Pseudocode in paper README

Use the fused Triton kernel directly. Don't reimplement.

## Sub-runs

### 2.1 — Integration smoke test

Short 1000-step run. Verify:
- Kernel runs on sm89 without errors
- VRAM cost matches paper's `O(Nd)` claim (N=6, d=256 → 6×256×B×T bytes per cache, modest)
- Forward and backward pass numerically stable (no NaN, grad norms in expected range)

### 2.2 — Phase 1 baseline re-run

Confirm Phase 1 result still holds with the same seed. If divergence, debug before adding AttnRes.

### 2.3 — AttnRes full run

Full sweep-length run with mean compressor. Same hyperparameters as Phase 1.

### 2.4 — LR sensitivity check (if needed)

AttnRes changes gradient flow significantly. If 2.3 looks undertrained or unstable, narrow LR check (0.5×, 1×, 2×).

## Decision rules

- **Headline metric:** eval loss vs Phase 1 baseline at same FLOPs.
- **Win:** loss improves by >3% (Kimi paper showed 1.25× compute-equivalent improvement at scale; expect smaller at our scale).
- **Marginal:** 1-3% improvement → ship anyway, AttnRes is the foundation for later phases.
- **Loss:** within 1% or worse → debug. Likely compressor (mean is the paper default but may be undertuned at our scale) or LR.

## Phase 2 exit gate

- [ ] Fused Triton kernel integrated
- [ ] VRAM cost matches expectation
- [ ] Loss curve matches or beats Phase 1
- [ ] Activation magnitudes bounded across depth (paper's PreNorm-dilution-fix claim — verify in wandb logs)
- [ ] CHANGELOG entry written

## What this phase does NOT do

- Conv compressor (Phase 3)
- Block composition (Phase 4)
- Block size sweep (locked at 4 layers/block, 6 blocks)
- Depth scaling (Phase 5)

## Open question carried into Phase 3

Block compression via mean treats every sub-block output as equally important. Some sub-blocks build semantic features that should be preserved; others do local refinements that should average out. The mean compressor has no way to express this distinction. Phase 3 swaps it for a learned conv compressor.

## References

- Kimi AttnRes paper: https://arxiv.org/abs/2603.15031
- Flash AttnRes kernel: https://github.com/catswe/flash-attention-residuals
