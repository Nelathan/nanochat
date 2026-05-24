# Chimera Experiment Contract

The constitution. Every phase inherits these decisions. Violations must be explicit, justified, and documented as a diff.

## Values

- **The defaults are the science.** "Stay slim" is the antipattern. Skipping normalization, optimizer config, precision, dataloader hygiene invalidates findings. One-variable-at-a-time is the only valid mode.
- **Fail early.** Single-seed, short sweeps. If a candidate is 2× off the AdamW reference at 10%, kill.
- **Composer, not orchestrator.** Each phase has a hypothesis, one variable, a fixed contract reference, and a decision rule. Anything else is drift.
- **A doc that picks for me is a doc that lies to me.** Open questions stay open. Active decisions are surfaced as such.
- **No multi-GPU, no distributed.** Single 4070 Super primary, 5090 derived. All multi-GPU code stripped.

## Hardware

- Primary: RTX 4070 Super (sm89, 12GB VRAM)
- Secondary: RTX 5090 over SSH (Alienware, currently `nvidia-smi` broken, fix expected next week)
- 5090 scaling recipe: 2× batch, β_2 via `β_2^2`, LR via `LR × 2^0.3 ≈ 1.23×`. Same contract, derived at run-time.

## Base implementation

- **nanochat** (`karpathy/nanochat`) as base. Adapted for single-GPU scale, multi-GPU stripped.
- **uv** for package management. Latest PyTorch (2.12+ for `torch.compile` fusion improvements).
- **bf16** compute, fp32 master weights (handled by FlashOptim's bf16 + 8-bit ECC).
- **No FA3** — H100+ only. Falls back to PyTorch FA2 for GQA.

## Locked architecture (Phase 0 baseline)

Inherits nanochat's GPT rewrite features:
- RoPE replaced with NoPE (validated working at our scale, short context)
- QK norm
- Untied embeddings → **tied embeddings** (mandatory at d=256 / V=16k for param budget)
- relu² activation in MLP → **SiLU ungated** (smooth gradients at narrow d, bf16-stable, AttnRes handles depth-axis selection so gated MLP is redundant)
- Norm after token embedding
- No learnable params in RMSNorm
- No bias in linear layers
- GQA support
- No FA3, FA2 via PyTorch

Geometry:
- d = 256, h = 512 (h = 2d, smaller than nanochat's 4d ratio; ungated SiLU needs less expansion than gated)
- 24 layers for pilot (Phase 0–4), scaling to 48 then 64 in Phase 5
- ~42M params at 64L target including tied embeddings

Attention pattern:
- Phase 0: nanochat default (SWA or full GQA per nanochat's choice)
- Phase 1+: GDN2 from NVIDIA (Triton kernel, sm89 compatible) replaces SWA
- Block pattern (post-AttnRes): TBD by Phase 4 ablation. Default carries `GDN2 GDN2 GQA GDN2` as starting point.

References:
- nanochat: https://github.com/karpathy/nanochat
- GDN2: https://github.com/NVlabs/GatedDeltaNet-2
- Flash AttnRes: https://github.com/catswe/flash-attention-residuals
- AttnRes paper: https://arxiv.org/abs/2603.15031 (Kimi)
- Small batch / β_2 half-life: https://arxiv.org/html/2507.07101v2 (Marek et al.)
- FlashOptim: https://github.com/ClashLuke/FlashOptim
- Unsloth: https://github.com/unslothai/unsloth

## Data

- **PleIAs/SYNTH**, shuffled (not curriculum-sorted; SYNTH is hard but noise-free, treat as best-tiny-curriculum data).
- 3 parquet files × ~500MB ≈ 225M tokens. 2 epochs at confirmation runs → ~450M tokens.
- Sweep runs: 1 file × 1 epoch (~75M tokens) for fast iteration.
- 16k BPE tokenizer trained on SYNTH (Phase -1, see below). Locked once trained.

## Tokenizer (Phase -1, setup)

- Vocab size: **16k**
- Training data: same SYNTH subset as model training (or slightly larger sample for better merges)
- Algorithm: BPE
- Digit splitting + byte fallback per nanochat conventions
- Special tokens decided upfront, frozen
- Trained once. Re-tokenizing midstream invalidates everything downstream.

## Optimizer

- **FlashOptim AdamW** for everything (matrices, embeddings, lm_head).
- Muon explicitly NOT used. nanogpt-speedrun favors Muon but operates at much larger effective batch with hyperparameters tuned for it; we don't have either condition. Interaction with GDN2 weights is unknown.
- β_1 = 0.9 (paper validates this across batch sizes).
- β_2 derived from token half-life rule: `β_2* = β_2^(B*/B_ref)` where reference is `β_2=0.95 at B=512·T tokens-per-step`. Computed once B* is locked.
- LR: derived from `LR_nano × (B*/B_nano)^0.3` as starting point, then narrow sweep.
- Embedding LR ratio: 0.1× backbone (validated at d=128 previously, expected to transfer).
- Weight decay: 0 at B=1 per paper; otherwise 0.1 per nanochat default. Decision deferred to Phase 0 once B* is locked.

## Training schedule

- **WSD** (warmup-stable-decay). Cosine rejected — better loss, worse comparability across runs of different step counts.
- Warmup: 5% of total steps
- Decay: 10% of total steps
- Stable: 85%
- Step count: derived from token budget and B*. Sweeps short (loss curves, not full convergence).

## Memory / efficiency tricks

- **Activation checkpointing** on (4070 is memory-traffic bound, never FLOPs bound — checkpointing trades recompute for memory headroom that buys larger batch).
- **Chunked cross-entropy** (Unsloth) — definite win, no behavior change.
- **Fused RMSNorm** (Unsloth) — benchmark against `torch.compile`'d nanochat RMSNorm. Include only if faster.
- **nanochat dataloader** stays. Cross-sequence attention masking enforced — no BOS-trick contamination.
- **torch.compile** on the full model (nanochat default).

## Logging (wandb)

- Cadence: **every 20 steps**, accumulated and averaged within window (not noisy last value).
- ~500 log points over 10k steps target.
- **No GPU syncs inside train loop except at log boundaries.** No `.item()` calls per step.
- Per-step on GPU, sync at log step:
  - loss (train, GPU-accumulated)
  - grad norm (GPU-computed)
  - lr per param group (CPU-side, free)
  - MFU (CPU-side, free)
  - max memory allocated (one sync)
- No histograms.
- Eval every 10% of total steps (10 evals per run).

## Reproducibility manifest

Every run logs:
- git SHA
- exact seed (single seed per run, no averaging — "feel lucky")
- PyTorch version, CUDA version
- GPU model
- contract version (this doc's git SHA)
- phase + iteration name
- full hyperparameter dump

Full determinism: `torch.use_deterministic_algorithms(True)` where possible, deterministic dataloader shuffling, fixed seed for init and data.

## Kill criteria

Abort a run if any of:
- 10% checkpoint eval loss > 2× AdamW reference baseline loss
- NaN loss
- Grad norm spike > 100× rolling median
- MFU drop > 2× from baseline (kernel regression or memory thrashing)
- VRAM OOM (re-tune B and restart, not silent kill)

## Phase exit gates

Each phase ends with explicit go/no-go decision *before* next phase starts:
1. Did the variable's effect match the hypothesis direction?
2. Is the effect size above the decision threshold (defined per phase)?
3. Did anything in the contract change as a result? If yes, document the diff.

No phase-jumping. No "let me just try X while I'm here."

## Changelog discipline

`CHANGELOG.md` carries one entry per verified successful step. Design perspective only, not implementation detail. Commit after each phase exit gate passes.

## What this contract does not cover

Surfaced as open questions in phase docs:
- Exact h ratio (h=2d locked, but the case for h=d if MLPs are shown underutilized stays open)
- Block composition (Phase 4 decides)
- Conv compressor kernel init (Phase 3 decides, default = init to equal weights so day-1 loss matches avg baseline)
- GDN2 vs Mamba-3 — closed in favor of GDN2 on sm89 Triton availability
- μP scope — minimal sanity check at Phase 5 depth scaling, not its own phase
