# Phase 0 — Training Calibration

**Hypothesis:** nanochat's hyperparameters, tuned for 8×H100 scale, need rescaling to single-4070-Super scale. The half-life formulation (Marek et al., arxiv 2507.07101) provides the principled rescaling. Once B*, β_2, and LR are locked at our scale, the training stack is the foundation all later phases inherit.

**Variable:** training hyperparameters. No architecture changes.

**Architecture:** raw nanochat-adapted per CONTRACT.md. SWA / nanochat default attention. No GDN2, no AttnRes.

**Geometry:** 24L pilot, d=256, h=512, 16k tokenizer, 2k context, tied embeddings.

## Sub-runs (sequential)

### 0.1 — MFU-vs-batch sweep

Find smallest batch B* that maximizes throughput on 4070 Super. Per Marek et al., this is the optimal operating point. The MFU curve is the decision; eval loss not measured here.

- Sweep B ∈ {1, 2, 4, 8, 16, 32} (or whatever fits VRAM with activation checkpointing on)
- Run 100 steps each at random init, measure tokens/sec
- Decision: pick B* at the knee of the MFU curve (highest tokens/sec with stable VRAM headroom)
- Document B* and tokens-per-step `B* × 2048`

### 0.2 — β_2 derivation (no sweep)

From paper's formula:
```
β_2* = β_2_ref ^ (B* / B_ref)
```
Reference: β_2=0.95 at B=512 (Brown et al. / nanoGPT default).

If B*=4: β_2* = 0.95^(4/512) = 0.95^(1/128) ≈ 0.99960
If B*=8: β_2* = 0.95^(8/512) ≈ 0.99920
If B*=16: β_2* = 0.95^(16/512) ≈ 0.99840

β_1 = 0.9 (paper validates across batch sizes).

Lock derived β_2*. **No β_2 sweep.** The whole point of the half-life parameterization is avoiding the sweep.

### 0.3 — LR sweep at locked B*, β_2*

Starting point per paper's empirical sub-square-root scaling:
```
LR_start ≈ LR_nano × (B* / B_nano)^0.3
```
where B_nano is nanochat's default batch size and LR_nano its default LR.

Sweep: 4 LR points geometrically spaced ±2× around `LR_start`.

- Short runs: 1 file × 1 epoch (~75M tokens). Not full convergence.
- Eval every 10% of run
- Pick lowest eval loss at run end

If lowest LR is at the edge of the sweep range, expand and re-sweep that direction.

### 0.4 — Embedding LR ratio confirmation

Default 0.1×. Quick 2-point check (0.05×, 0.1×, 0.2×) at locked LR from 0.3 to confirm 0.1× still wins at this geometry. If not, lock the winner.

### 0.5 — Weight decay decision

Per paper: weight decay = 0 at B=1. nanochat default = 0.1.

Run two short runs at locked everything-else: wd=0 vs wd=0.1. Pick winner. Lock.

### 0.6 — Final confirmation run

Full 2-epoch run on 3 files (~450M tokens). All hyperparameters locked.

This is the **Phase 0 reference loss**. Every later phase compares its eval loss curve to this one.

## Decision rules

- **MFU sweep:** pick knee (max throughput before diminishing returns).
- **LR sweep:** lowest eval loss at end of short run wins. If two are within 2%, pick smaller LR (safer).
- **Embedding ratio:** if 0.1× wins by >1%, keep it. Otherwise lock the winner.
- **Weight decay:** lowest eval loss wins. Tie-break on simpler value (wd=0).

## Phase 0 exit gate

- [ ] B* locked, documented with MFU number
- [ ] β_2 computed from formula, no sweep done
- [ ] LR locked from 4-point sweep
- [ ] Embedding ratio locked (likely 0.1×)
- [ ] Weight decay locked
- [ ] Reference run completed, eval loss curve in wandb
- [ ] CHANGELOG entry written

## What this phase does NOT do

- No architecture changes (no GDN2, no AttnRes)
- No Muon comparison (excluded per CONTRACT.md decision)
- No Adafactor (excluded per Daniel's decision)
- No multi-seed averaging
- No β_2 sweep (half-life rule replaces it)
- No model size scan (24L only)
- No depth scaling (Phase 5)

## Open question carried into Phase 1

GDN2 weights are dimensionally different from attention weights. FlashOptim AdamW should handle them without issue, but watch for grad-norm anomalies in Phase 1 that might indicate optimizer-architecture mismatch.

## References

- Half-life rule: https://arxiv.org/html/2507.07101v2 §4.3
- LR scaling exponent: same paper, Figure 4
- nanochat baseline: https://github.com/karpathy/nanochat
- FlashOptim: https://github.com/ClashLuke/FlashOptim
