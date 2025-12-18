# AGENDA (Archived)

> **Note**: This file is archived. See `mole/README.md` for the current project status and vision.

## Vision (Original)

**"The best ChatGPT that $10 can buy"** - Enable training large context language models on consumer RTX 5090 hardware through MoleGrad low-rank gradient updates.

### Core Innovation
MoleGrad treats weight updates as living in low-rank subspace. For each large Linear layer, estimate top-k singular modes from activations A and output gradients B without forming full gradient G = BᵀA. Update only along those dominant modes.

Key benefits:
- Memory scales with rank_k, not sequence length
- Warm-start subspace tracking across steps
- Enables `device_batch_size=1` with `max_seq_len=8192+` on 32GB VRAM

## Phase 1: Foundation ✅ COMPLETE

### Core Implementation
- ✅ **MoleLinear**: Drop-in replacement for nn.Linear with custom autograd
- ✅ **MoleGradConfig**: Modern configuration with validation
- ✅ **Low-rank gradient computation**: Block power method without forming full G
- ✅ **Warm-start tracking**: Subspace basis persistence across training steps
- ✅ **Trust scaling**: Hutchinson norm estimation for gradient stability
- ✅ **Smart orthogonalization**: Column-norm check with QR fallback
- ✅ **Integration**: Seamless nanochat integration with CLI flags
- ✅ **run10.sh**: Complete training pipeline optimized for RTX 5090

### Key Files Created/Modified
- `nanochat/streaming_linear.py` - Core MoleGrad implementation
- `nanochat/gpt.py` - Added `enable_molegrad()` method
- `scripts/base_train.py` - MoleGrad configuration and initialization
- `run10.sh` - RTX 5090 optimized training script
- `CLAUDE.md` - Updated documentation

### Current Defaults (racing car setup)
```bash
--enable_streaming=True
--streaming_rank_k=32
--streaming_power_iters=1
--streaming_target_layers="ffn_only"
--streaming_warm_start_decay=0.95
```

## Phase 2: Validation & Debugging 🎯 IN PROGRESS

### MUST HAVE - Core Validation
- 🎯 **Muon compatibility**: Verify polar updates work with low-rank Ĝ
- 🎯 **Gradient quality tracking**: Energy capture and norm ratios
- 🎯 **Basic metrics**: Memory usage, timing, stability indicators
- 🎯 **Loss convergence**: Verify MoleGrad doesn't hurt training dynamics
- 🎯 **Checkpoint compatibility**: Ensure save/load works with MoleGrad state

### MUST HAVE - Debugging Infrastructure
- 🎯 **Wandb integration**: Track key metrics
  - `energy_capture`: ||Ĝ||_F / ||G_true||_F
  - `gradient_norm_compression_ratio`: full/compressed norm ratio
  - `subspace_quality`: Q_t vs Q_{t-1} cosine similarity
  - `layer_memory_usage`: Per-layer VRAM tracking
  - `trust_scaling_factor`: Applied scaling per layer
- 🎯 **Periodic true G comparison**: Shadow layer validation
- 🎯 **Health checks**: Detect degenerate subspaces automatically

### Tracking Metrics to Implement
```python
# In _lowrank_grad return additional metrics:
metrics = {
    'energy_capture': energy_ratio,
    'trust_factor': scaling_factor,
    'condition_number': cond_number,
    'subspace_drift': cosine_sim,
    'memory_saved_mb': memory_savings,
}
```

## Phase 3: Performance Optimization 🏁 OPTIONAL

### Speed Optimizations (Future)
- 🏁 **Kernel fusion**: A@Q and Bᵀ@(...) operations
- 🏁 **Memory pooling**: Preallocated scratch buffers per layer
- 🏁 **cuBLASLt integration**: Grouped GEMMs for batch processing
- 🏁 **FP8 matmuls**: E4M3/E5M2 for heavy computations

### Advanced Features (Future)
- 🏁 **Sequence streaming**: True O(1) memory in sequence length
- 🏁 **Adaptive rank k**: Dynamic sizing based on energy capture
- 🏁 **Advanced optimizers**: Adafactor coupling with low-rank gradients
- 🏁 **Attention streaming**: Compress QKV projections too

## Design Philosophy

### Racing Car, Not Autopilot
- **Full user control**: All parameters exposed via CLI flags
- **No auto-magic**: Removed auto-tuning features
- **Transparent behavior**: Clear logging of what's happening
- **Fast iteration**: Quick reruns without retraining tokenizer

### Success Criteria
1. **Memory efficiency**: 4x-8x reduction in gradient memory
2. **Training stability**: Loss curves comparable to full gradients
3. **Performance**: Minimal overhead (<20% step time increase)
4. **Quality**: Model performance within 5% of baseline
5. **Usability**: Simple CLI interface, clear debugging output

## Current Status

### Ready to Test
- Core MoleGrad implementation is complete
- run10.sh provides end-to-end training pipeline
- Basic validation infrastructure in place

### Next Steps
1. **Test Muon + Ĝ compatibility**: Verify optimizer works with low-rank gradients
2. **Add tracking metrics**: Implement debugging metrics
3. **Run validation**: Execute full training pipeline and monitor convergence
4. **Iterate**: Based on results, tune parameters and fix issues

### Blocking Issues
- None identified yet - ready for first validation run

## Testing Strategy

### Phase 2 Testing Plan
1. **Smoke test**: 100 steps to verify no crashes
2. **Short run**: 1K steps to check convergence
3. **Full run**: 5K steps for validation
4. **Comparison**: Run identical training without streaming for baseline
5. **Metrics analysis**: Review wandb logs for stability indicators

### Success Gates
- ✅ No crashes during training
- ✅ Memory usage within expected bounds
- ✅ Loss decreases monotonically
- ✅ Gradient quality metrics stable
- ✅ Final model generates coherent text
