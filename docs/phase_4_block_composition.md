# Phase 4 — Block Composition Ablation

**Hypothesis:** With AttnRes + conv compressor in place, the optimal block composition pattern may differ from Daniel's pre-AttnRes finding (`GDN2 GDN2 GQA GDN2`). AttnRes provides global reach, so the local composition matters less than before — but exactly how much less is empirical.

**Variable:** which sub-block positions are GDN2 vs GQA within the 4-layer block. Block size and count are fixed.

**Inherits:** Phase 3 contract. AttnRes + conv compressor in place. All hyperparameters locked.

## Variants to test

Three candidates from prior design discussion:

1. **Uniform middle:** `GDN2 GDN2 GQA GDN2` repeated × 6 blocks (Daniel's pre-AttnRes winner)
2. **Endcap-pure + uniform middle:** first and last blocks pure GDN2, middle 4 blocks uniform pattern
3. **Endcap-pure + GQA-heavy middle:** first and last blocks pure GDN2, middle blocks with 2 GQA per block

All variants have same total layer count (24) and roughly similar FLOPs. GQA count varies: variant 1 has 6 GQA layers, variant 2 has 4, variant 3 has 8.

## What changes from Phase 3

Only the per-block composition. Everything else fixed.

## Sub-runs

### 4.1 — Variant 1 (Phase 3 baseline)

Reproduces Phase 3 best result. Confirms reproducibility.

### 4.2 — Variant 2

Endcap-pure GDN2 (blocks 0 and 5 are 4× GDN2). Middle 4 blocks uniform `GDN2 GDN2 GQA GDN2`.

### 4.3 — Variant 3

Endcap-pure GDN2. Middle 4 blocks `GDN2 GQA GDN2 GQA`.

All three runs use identical seed, identical token budget, identical schedule.

## Decision rules

- **Lowest eval loss wins.** Need >1.5% margin to be considered meaningfully better than Variant 1 (which is the simpler default).
- **FLOP-aware:** if a variant wins by margin <1.5% but uses fewer FLOPs (less GQA), it still wins (efficiency at equal loss).
- **VRAM tie-break:** at equal loss and FLOPs, lower VRAM wins.

## Phase 4 exit gate

- [ ] All three variants run with same seed
- [ ] Winner picked per decision rules
- [ ] Block composition locked for Phase 5
- [ ] CHANGELOG entry written

## What this phase does NOT do

- Block size variation (still locked at 4)
- N blocks variation
- Adding new layer types
- Re-tuning hyperparameters
- Depth scaling (Phase 5)

## Open question carried into Phase 5

The winner at 24L may not be the winner at 64L. Phase 5 should sanity-check the chosen composition at 48L and 64L before locking forever. If composition needs to change with depth, that's a discovery, not a failure.

## References

- Phase 3 winner config
- Daniel's prior block-pattern ablation (pre-AttnRes)
