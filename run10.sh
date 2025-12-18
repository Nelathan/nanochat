#!/bin/bash

# This script is the "Best ChatGPT clone that $10 can buy",
# Optimized for single RTX 5090 with experimental settings:
# - device_batch_size=1 (minimal VRAM)
# - max_seq_len=8192 (long context)
# - total_batch_size=1 (no gradient accumulation)

# Example launch:
# bash run10.sh
# Example launch in a screen session:
# screen -L -Logfile run10.log -S run10 bash run10.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report setup
# uv run -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download dataset - keep full tokenizer data but reduce training data
# Tokenizer: 8 shards (~2B chars) - done once, keep original quality
# Training: 24 shards (~6B chars) - 1/10 of original for our tiny model
# uv run -m nanochat.dataset -n 8
# uv run -m nanochat.dataset -n 24 &
# DATASET_DOWNLOAD_PID=$!

# train tokenizer with original 2B chars but smaller 32k vocab for our tiny model
# check if tokenizer already exists to avoid retraining on multiple runs
# TOKENIZER_PATH="$NANOCHAT_BASE_DIR/tokenizer.model"
# if [ ! -f "$TOKENIZER_PATH" ]; then
#     echo "Training new tokenizer..."
#     uv run -m scripts.tok_train --vocab_size=32768
# else
#     echo "Using existing tokenizer at $TOKENIZER_PATH"
# fi
# uv run -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) - RTX 5090 optimized

# Download eval bundle for CORE metric
# EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
# if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
#     curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
#     unzip -q eval_bundle.zip
#     rm eval_bundle.zip
#     mv eval_bundle $NANOCHAT_BASE_DIR
# fi

# Wait for minimal dataset download
# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID

# Experimental single-GPU training for RTX 4070 super with MoleGrad

#failfast
uv run -m scripts.base_train \
    --depth=8 \
    --max_seq_len=1024 --device_batch_size=64 --grad_accum_steps=1 \
    --num_iterations=10000 \
    --eval_every=500 \
    --eval_tokens=64000 \
    --core_metric_every=-1 --save_every=-1 \
    --mlp_mult=2 --num_kv_heads=1 \
    --run=mole-erf-gqa1-1

# Full training run
uv run -m scripts.base_train \
    --depth=12 \
    --max_seq_len=1024 --device_batch_size=16 --grad_accum_steps=1 \
    --num_iterations=10000 \
    --eval_every=1000 \
    --eval_tokens=128000 \
    --core_metric_every=50000 \
    --sample_every=10000 \
    --run=mini_1

# Quick evaluation on small data subset
uv run -m scripts.base_loss --device_batch_size=1 --split_tokens=8192
uv run -m scripts.base_eval --max-per-task=50
echo "=== run10.sh completed! ==="
