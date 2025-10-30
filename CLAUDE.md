# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a full-stack implementation of a ChatGPT-like LLM with MoleGrad optimization for RTX 5090. This fork focuses on consumer GPU training through low-rank gradient updates, enabling long-context training with minimal VRAM usage. The project includes tokenization, pretraining, evaluation, inference, and web serving.

## Key Commands

### Environment Setup
```bash
# Install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
[ -d ".venv" ] || uv venv
source .venv/bin/activate

# Install dependencies (GPU version)
uv sync --extra gpu

# Build the Rust BPE tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### RTX 5090 Training Pipeline
```bash
# Primary training script - "Best ChatGPT clone that $10 can buy"
bash run10.sh

# Base training only (with custom parameters)
python -m scripts.base_train --enable_streaming=True

# Individual training stages
python -m scripts.base_train
python -m scripts.mid_train
python -m scripts.chat_sft
python -m scripts.chat_rl
```

### Inference and Chat
```bash
# Chat via CLI
python -m scripts.chat_cli

# Chat via web interface (ChatGPT-like UI)
python -m scripts.chat_web

# Chat with specific model checkpoint
python -m scripts.chat_cli -c path/to/checkpoint.pt
```

### Evaluation
```bash
# Tokenizer evaluation
python -m scripts.tok_eval

# Base model evaluation (CORE score)
python -m scripts.base_eval

# Chat model evaluation
python -m scripts.chat_eval

# Loss evaluation
python -m scripts.base_loss
```

### Testing
```bash
# Run tests
python -m pytest tests/test_rustbpe.py -v -s

# Run specific test
python -m pytest tests/test_rustbpe.py::test_rustbpe_train -v
```

### Data Management
```bash
# Download pretraining data shards
python -m nanochat.dataset -n 8  # Download 8 shards

# Train tokenizer
python -m scripts.tok_train --max_chars=2000000000

# Generate report
python -m nanochat.report generate
```

### RTX 5090 Streaming Configuration
```bash
# Manual streaming configuration (run10.sh defaults shown)
python -m scripts.base_train \
    --enable_streaming=True \
    --streaming_rank_k=32 \
    --streaming_power_iters=1 \
    --streaming_target_layers="ffn_only" \
    --device_batch_size=1 \
    --max_seq_len=8192 \
    --total_batch_size=1
```

## Architecture

### Core Components

1. **GPT Model** (`nanochat/gpt.py`): Transformer architecture with rotary embeddings, QK norm, MQA support
2. **MoleGrad** (`nanochat/streaming_linear.py`): Low-rank gradient updates tunnel through subspace
3. **Tokenizer** (`nanochat/tokenizer.py`): BPE tokenizer with special tokens for conversations and tool use
4. **Data Loader** (`nanochat/dataloader.py`): Distributed tokenizing data loader for pretraining
5. **Engine** (`nanochat/engine.py`): Efficient inference engine with KV cache
6. **Execution** (`nanochat/execution.py`): Python code execution tool for assistant

### Training Stages

1. **Base Training** (`scripts/base_train.py`): Pretrain on raw text data
2. **Midtraining** (`scripts/mid_train.py`): Teach conversation special tokens and tool use
3. **SFT** (`scripts/chat_sft.py`): Supervised finetuning on conversation data
4. **RL** (`scripts/chat_rl.py`): Reinforcement learning (currently GSM8K only)

### Configuration

- Uses custom configurator system (`nanochat/configurator.py`) for simple command-line overrides
- Model size controlled by `--depth` parameter (affects all other dimensions automatically)
- Device batch size can be adjusted to fit GPU memory: `--device_batch_size=32/16/8/4/2/1`

### Special Tokens

The tokenizer includes special tokens for structured conversations:
- `<|bos|>`: Document boundary
- `<|user_start|>` / `<|user_end|>`: User message boundaries
- `<|assistant_start|>` / `<|assistant_end|>`: Assistant message boundaries
- `<|python_start|>` / `<|python_end|>`: Python tool execution
- `<|output_start|>` / `<|output_end|>`: Tool output

### Hardware Requirements

- **Primary**: RTX 5090 (32GB VRAM) with streaming gradients
- **Alternative**: Single GPU with reduced batch size (uses gradient accumulation)
- **CPU/MPS**: Supported for development/testing (see `dev/runcpu.sh`)
- **Legacy**: Multi-GPU support retained but not optimized for this fork

### Key Design Principles

- **Consumer GPU focus**: Optimized for RTX 5090 with streaming gradients
- **Minimal dependencies**: Uses uv for Python management, maturin for Rust extension
- **Single script execution**: End-to-end training via run10.sh
- **User control**: Full parameter exposure, no auto-tuning
- **MoleGrad updates**: Low-rank gradient compression enables long-sequence training on 32GB VRAM
- **Evaluation integrated**: Automatic metrics and report generation

### MoleGrad Features

- **Memory efficient**: O(rank_k) memory instead of O(sequence_length × model_dim)
- **Warm-start tracking**: Subspace basis persists across training steps
- **Targeted compression**: FFN layers only (configurable to attention if needed)
- **Adaptive subspace**: Power iterations find dominant gradient directions
- **No full gradients**: Custom autograd prevents dense gradient materialization

### Important Notes

- Set `WANDB_RUN` environment variable to enable logging (defaults to "dummy")
- Intermediate artifacts stored in `$HOME/.cache/nanochat`
- Model checkpoints are saved automatically during training
- The system auto-detects available compute (GPU/CPU/MPS)
- Streaming gradients enable `device_batch_size=1` with long sequences (8192+ tokens)
- Configure `streaming_rank_k` to balance memory savings vs gradient quality
- `run10.sh` is the primary training script for this RTX 5090-optimized fork
- Tokenizer training is cached to avoid retraining on multiple runs