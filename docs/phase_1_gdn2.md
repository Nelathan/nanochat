# Phase 1 — GDN2 Swap

**Hypothesis:** Replacing nanochat's default attention with GDN2 on 3 of every 4 layers (keeping GQA at position 3 of each 4-layer block) maintains or improves loss at lower VRAM and FLOPs cost, enabling the deeper-narrow towers later phases need.

**Variable:** attention layer composition. Baseline = all-GQA. Treatment = `GDN2 GDN2 GQA GDN2` pattern (Daniel's prior ablation winner without AttnRes).

**Inherits:** Phase 0 contract. All hyperparameters locked from Phase 0.6.

## What changes from Phase 0

- 3 of every 4 layers swap from GQA → GDN2
- Block size = 4 layers (forced by 24L / 6 blocks for clean division)
- Position of GQA within block = index 3 (Daniel's prior winner)

## What stays fixed

- B*, β_2, LR, embedding ratio, weight decay — all from Phase 0
- Geometry: 24L, d=256, h=512
- Tokenizer, dataset, schedule
- Logging cadence, eval cadence

## Sub-runs

### 1.1 — Baseline replication

Re-run Phase 0.6 confirmation with same seed. Confirm reproducibility before introducing the variable. If loss curve diverges from Phase 0.6 by >1%, debug determinism before continuing.

### 1.2 — GDN2 swap

Replace 18 of 24 layers with GDN2. Same hyperparameters. Same run length.

### 1.3 — Quick LR sanity (if needed)

If 1.2 trains stably and converges, skip. If grad norms behave differently than Phase 0, run a narrow LR check (0.5×, 1×, 2× of locked LR) to confirm we're not silently undertrained.

## Decision rules

- **Loss equivalence:** GDN2 eval loss within 2% of all-GQA baseline → swap is free, ship it.
- **Loss improvement:** GDN2 eval loss > 2% better → strong signal, ship it.
- **Loss regression:** GDN2 worse by > 2% → investigate. Likely the position of GQA needs reconsideration (Daniel's prior ablation was without AttnRes, so the optimum may shift). Run quick 3-position ablation (GQA at index 1, 2, 3) before declaring failure.
- **VRAM check:** GDN2 should reduce VRAM. If it doesn't, kernel integration issue, investigate.

## Phase 1 exit gate

- [ ] GDN2 integrated, Triton kernel runs cleanly on sm89
- [ ] Loss curve matches or beats Phase 0 baseline
- [ ] VRAM reduction documented
- [ ] CHANGELOG entry written
- [ ] If LR re-tuned, document the new LR and reason

## What this phase does NOT do

- No AttnRes (Phase 2)
- No block composition ablation (Phase 4)
- No conv compressor (Phase 3)
- No depth scaling (Phase 5)
- No optimizer changes

## Open question carried into Phase 2

AttnRes (Phase 2) adds per-layer checkpoints. The block pattern of `GDN2 GDN2 GQA GDN2` was Daniel's winner without AttnRes. Once AttnRes provides global reach via the softmax over block representations, the local pattern may matter less — this is the Phase 4 ablation. For Phase 2, hold the pattern fixed.

## References

- GDN2: https://github.com/NVlabs/GatedDeltaNet-2
- Daniel's prior block-pattern ablation: see chat history (Chimera v9–v16)
