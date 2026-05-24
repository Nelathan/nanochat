# Phase 5 — Depth Scaling

**Hypothesis:** Narrow-deep tower (d=256, 64L) with AttnRes + conv compressor + chosen block composition scales monotonically — eval loss improves with depth at fixed d. The whole project rests on this. If it fails, depth wasn't the right bet.

**Variable:** number of layers. 24 → 48 → 64. d=256 held constant per μP transfer logic.

**Inherits:** Phase 4 contract. All hyperparameters locked. Architecture locked.

## What changes from Phase 4

- Layer count 24 → 48 (Phase 5.1) → 64 (Phase 5.2)
- N blocks adjusts: 12 blocks at 48L, 16 blocks at 64L (block size stays 4)
- Total steps may need adjusting per token budget at larger param count

## What does NOT change

- d=256, h=512 (μP transfer requires width fixed)
- All hyperparameters from Phase 4
- Block composition from Phase 4 winner
- All architecture from Phase 4
- Tokenizer, dataset, schedule shape

## Sub-runs

### 5.1 — 48L scale-up

- Adjust step count to maintain ~Chinchilla-optimal tokens-per-param for this size
- Same hyperparameters
- Verify training stability through full run

### 5.2 — μP sanity check

If 5.1 trains stably, no need for full μP sweep. If unstable or grad norms misbehave, narrow LR check (0.5×, 1×, 2× of Phase 4 LR) at 48L.

### 5.3 — 64L scale-up

Final target geometry. Full Chinchilla-optimal run.

### 5.4 — 64L confirmation

Re-run 5.3 with different seed. If you only do one multi-seed run in the whole project, do it here — the foundational claim that depth pays off justifies the spend.

## Decision rules

- **Monotonic improvement:** eval loss improves from 24L → 48L → 64L. If yes, project hypothesis validated.
- **48L stable, 64L unstable:** narrow LR re-sweep at 64L. Hyperparameter transfer was the gap, not architecture.
- **48L worse than 24L:** stop. Investigate. Possible causes: AttnRes K=8 cache too small for the deeper tower (try increasing block count); LR doesn't transfer (μP-style re-sweep); gradient flow fundamentally limited.
- **64L worse than 48L:** the depth-is-better thesis caps out somewhere between 48 and 64. Investigate but don't necessarily fail.

## Phase 5 exit gate

- [ ] 48L trains stably
- [ ] 64L trains stably
- [ ] Depth scaling story validated or falsified with clear evidence
- [ ] Final config locked
- [ ] CHANGELOG entry written
- [ ] Project-level retrospective: did the architecture deliver vs the Phase 0 baseline at equal FLOPs?

## What this phase does NOT do

- d variation (locked at 256)
- New architecture changes
- New training tricks
- Production deployment (this is a portfolio architecture, not a production model)

## Project completion criteria

By end of Phase 5:
- Eval loss at 64L significantly better than Phase 0 baseline at equal FLOPs (the architecture earns its complexity)
- Architecture documented end-to-end
- Portfolio-worthy: someone reading the contract + phase docs + changelog can reconstruct what was built and why

## References

- All prior phases
- Original deep-narrow hypothesis: see project rationale in CONTRACT.md
