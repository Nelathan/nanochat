# Phase 3 — Conv Compressor for Block AttnRes

**Hypothesis:** Replacing mean/avg compression of sub-block outputs into the block cache with a learned depthwise 1D conv over the K=8 sub-block outputs improves loss by letting different channels weight different sub-blocks differently. The cache pathway is the dominant bandwidth channel; better compression directly raises capacity.

**Variable:** block compression function. Baseline = mean (Phase 2). Treatment = depthwise 1D conv, kernel size K = block_size × 2 = 8 (attn + mlp sub-block per layer), tied across blocks at pilot scale.

**Inherits:** Phase 2 contract. AttnRes infrastructure in place. All hyperparameters locked.

## What changes from Phase 2

- Block compressor: `mean(stack(sub_block_outputs))` → `einsum('k, k b t d -> b t d', conv_kernel, stack(sub_block_outputs))`
- `conv_kernel` shape: `[K]` (or `[K, D]` if per-channel — see open question below)
- Tied across all blocks (single shared kernel, ~K params total at tied scope)
- Raw partial-state entries in the AttnRes softmax do NOT go through the conv (they're not cacheable; the conv is for the cache compression specifically)

## Open design decisions (active, NOT pre-decided)

### 3.A — Kernel scope: scalar or per-channel?

Option A: scalar `[K]` kernel. ~8 params total. Same weights across all D channels.
Option B: per-channel `[K, D]` kernel. K×D = 2048 params. Each channel learns its own depth-mix.

**Recommendation:** start with scalar `[K]` for pilot. Cheaper, less likely to overfit, validates the basic idea. If wins, expand to per-channel in a follow-up sub-run.

**Daniel decides** before kicking off the run.

### 3.B — Tied vs per-block kernel?

Tied: single `[K]` (or `[K,D]`) kernel shared across all 6 blocks.
Per-block: 6 independent kernels.

**Recommendation:** tied at pilot. Per-block adds variables faster than insight at 24L scale. Middle layers being different is a known phenomenon, but the AttnRes softmax already learns per-block keys downstream — letting the compressor be tied keeps the gradient signal clean.

**Daniel decides** before kicking off the run.

### 3.C — Kernel initialization?

Init to `1/K` per position (equal weights) → conv is identical to mean at step 0. Day-1 loss matches Phase 2 baseline exactly. This is the safe init.

**Locked: init to `1/K`.** Sanity check at step 1 must show loss within 0.1% of Phase 2 step-1 loss. If not, init is wrong.

## Sub-runs

### 3.1 — Smoke test

Short 500-step run with scalar `[K]` conv, init `1/K`.
- Step 1 loss matches Phase 2 step 1 (within 0.1%)
- Kernel values drift from `1/K` during training (otherwise compressor is doing nothing)
- VRAM cost identical to Phase 2 (compressor params are negligible)

### 3.2 — Phase 2 baseline re-run

Same seed, same hyperparameters, mean compressor. Confirm reproducibility.

### 3.3 — Scalar conv full run

Full sweep-length run. Compare eval loss to 3.2.

### 3.4 — Per-channel conv (only if 3.3 wins)

If scalar conv beats mean, run per-channel `[K,D]` for an upside check. Skip if 3.3 doesn't win — adds complexity without justification.

## Decision rules

- **Scalar conv win:** >1.5% eval loss improvement → ship. Move to per-channel.
- **Scalar conv marginal:** 0.5-1.5% → ship, skip per-channel (low confidence the extra params pay).
- **Scalar conv loss/wash:** <0.5% improvement or worse → drop conv, keep mean. Investigate why before declaring failure (kernel might be staying near init = mean, meaning gradient isn't reaching it — log kernel values).

## Phase 3 exit gate

- [ ] Scalar conv compressor integrated, kernel init `1/K`, day-1 loss matches Phase 2
- [ ] Kernel values shown drifting from init during training (wandb scalar log of kernel weights)
- [ ] Eval loss decision made per rules above
- [ ] If per-channel run, decision made on whether to keep
- [ ] CHANGELOG entry written

## What this phase does NOT do

- Block composition ablation (Phase 4)
- Block size variation (locked at 4)
- N blocks variation (locked at 6 for 24L)
- AttnRes softmax modifications (only the compressor input is changed)
- Cross-channel conv (rejected: conflates compression with channel-mixing)

## Open question carried into Phase 4

Once the compressor is in, the optimal block composition (which layers run GDN2 vs GQA) may shift from Daniel's pre-AttnRes finding. Phase 4 re-ablates.

## References

- Phase 2 AttnRes integration
- Flash kernel adaptation point: modify the reduction step in https://github.com/catswe/flash-attention-residuals
