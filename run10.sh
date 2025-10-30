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
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report setup
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download dataset - keep full tokenizer data but reduce training data
# Tokenizer: 8 shards (~2B chars) - done once, keep original quality
# Training: 24 shards (~6B chars) - 1/10 of original for our tiny model
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 24 &
DATASET_DOWNLOAD_PID=$!

# train tokenizer with original 2B chars but smaller 32k vocab for our tiny model
# check if tokenizer already exists to avoid retraining on multiple runs
TOKENIZER_PATH="$NANOCHAT_BASE_DIR/tokenizer.model"
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "Training new tokenizer..."
    python -m scripts.tok_train --max_chars=2000000000 --vocab_size=32768
else
    echo "Using existing tokenizer at $TOKENIZER_PATH"
fi
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) - RTX 5090 optimized

# Download eval bundle for CORE metric
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# Wait for minimal dataset download
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Experimental single-GPU training for RTX 5090 with MVP streaming gradients
python -m scripts.base_train \
    --depth=12 \
    --max_seq_len=8192 \
    --device_batch_size=1 \
    --total_batch_size=1 \
    --eval_every=1000 \
    --eval_tokens=32000 \
    --enable_streaming=True \
    --run=$WANDB_RUN

# Quick evaluation on small data subset
python -m scripts.base_loss --device_batch_size=1 --split_tokens=8192
python -m scripts.base_eval --max-per-task=50
echo "=== run10.sh completed! ==="
