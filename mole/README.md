# MoleGrad: Low-Rank Subspace Optimization

**Vision**: Enable training large context language models on consumer hardware (e.g., RTX 5090) by treating weight updates as living in a low-rank subspace.

## Core Concept
Instead of computing the full gradient $G = B^T A$ (which requires storing large activations $A$), we:
1.  **Forward**: Store projected activations $AQ$ (size $N \times k$) instead of $A$ (size $N \times d$).
2.  **Backward**: Compute projected gradients $Y = B^T (AQ)$.
3.  **Update**: Compute the update direction using the eigendecomposition of $Y^T Y$.

This reduces activation memory from $O(N \cdot d)$ to $O(N \cdot k)$, where $k \ll d$ (e.g., $k=16$ vs $d=768$).

## Current Strategy: "Scout Renewal" + Eigen-EMA

We use a dynamic subspace $Q$ that evolves during training to chase the gradient signal.

### 1. Subspace Evolution ("Scout Renewal")
We treat the $k$ directions in $Q$ as a set of tunnels being dug by the mole.
*   **Main Tunnels (Elites)**: The top $k-1$ directions with the strongest gradient signal (eigenvalues of $Y^T Y$) are kept and aligned with the signal.
*   **Test Tunnel (Scout)**: The weakest direction (index 0) is **abandoned** and replaced by a random orthogonal vector (a new test tunnel).
*   **Result**: We continuously explore 1 new direction per step while exploiting the $k-1$ best known directions.

### 2. Optimizer: Hybrid SGD/Adafactor (Eigen-EMA)
We scale the updates based on the signal strength (eigenvalues $S$) using an Exponential Moving Average (EMA).
*   **Scaling Factor**: $1 / \max(\sqrt{\text{EMA}(S)}, 1.0)$
*   **Elites ($S > 1$)**: Scaled down (Adafactor/RMSProp behavior) to stabilize strong features.
*   **Scouts ($S < 1$)**: Unit scaling (SGD behavior). They must "earn" their variance.
    *   *Note*: Earlier experiments showed that boosting Scouts (allowing denominator < 1.0) led to faster convergence but potential instability.

## Non-Standard Techniques & Design Decisions

1.  **Low-Rank Gradient Approximation**: We assume $\nabla W \approx B^T (AQ)$. This is exact if the gradient lies in the subspace $Q$, but an approximation otherwise.
2.  **Subspace Evolution via Eigendecomposition**: We use `eigh(Y^T Y)` to find the optimal update directions within the current subspace. This is cheaper than SVD on the full gradient.
3.  **Scout Renewal**: Explicitly forcing exploration by replacing the lowest-energy eigenvector with noise. This solves "Mode Collapse" where the subspace would otherwise get stuck.
4.  **Eigen-EMA**: Applying Adafactor-style scaling to the *eigenvalues* of the gradient covariance. This is rotation-invariant and robust.
5.  **Jitter Regularization**: Adding `1e-6 * I` to $Y^T Y$ before `eigh`. Essential for stability when the subspace is degenerate (e.g., a new Scout has 0 signal).

## Shadow Full-Gradient Logger (Design)

To quantify how well MoleGrad's low-rank updates approximate a standard dense gradient $G$, we plan a "shadow" full-gradient logger built around a reference Linear layer.

Goals:
*  Measure how much gradient mass and geometry the subspace captures, without changing the main training dynamics.
*  Compare MoleGrad's update $G_{\text{mole}}$ against a one-step dense gradient $G_{\text{full}}$ on the same data.

Planned design:
*  **Shadow Linear Pair**: For a selected layer shape $(\text{out}, \text{in})$, instantiate a standard `nn.Linear` alongside the existing `MoleLinear` with shared inputs.
*  **One-Step Capture**: On a small debug run or at scheduled steps, run a single batch where the shadow Linear uses a dense optimizer (e.g. AdamW) purely to materialize $G_{\text{full}}$, while MoleGrad produces its low-rank update direction $G_{\text{mole}} = USV^\top$.
*  **Metrics (Logged, Not Applied)**:
    *  Frobenius norm ratio: $\|G_{\text{mole}}\|_F / \|G_{\text{full}}\|_F$.
    *  Cosine similarity: $\langle G_{\text{mole}}, G_{\text{full}} \rangle / (\|G_{\text{mole}}\|_F \cdot \|G_{\text{full}}\|_F)$.
    *  Spectral comparison: eigenvalue spectra of $G_{\text{full}}^T G_{\text{full}}$ vs $G_{\text{mole}}^T G_{\text{mole}}$ in a common basis.
*  **Isolation**: The shadow logger runs under a debug flag, does not apply its own updates, and is only used to emit metrics to WandB / logs for offline analysis.

This experiment will tell us not only how much gradient mass we capture in the subspace, but how faithfully we preserve the gradient's spectrum and directions (Muon-like "steepest descent" behavior) before committing to long pretraining or continual finetuning runs.

## Variance Tracking via Cholesky (Design)

Goal: introduce true second-order smoothing inside the $k$-dimensional subspace without tracking unstable eigenvector identities across steps.

Design sketch:
* Maintain a full $k\times k$ EMA "variance" matrix in subspace coordinates:
    * $M \leftarrow \beta M + (1-\beta)\,(Y^T Y)$.
* Use a small diagonal regularizer only when needed for numerical stability: $M_\epsilon = M + \epsilon I$.
* Compute $M_\epsilon = LL^T$ via Cholesky (fast for small $k$).
* Use $L$ to build a variance-normalized signal matrix, e.g. $\tilde C = L^{-1}(Y^T Y)L^{-T}$.
* Then use `eigh(\tilde C)` to obtain directions that maximize "signal / variance" (a generalized eigenproblem) rather than pure signal magnitude.

This approach keeps the variance estimate in a consistent coordinate system (the current $Q$ basis) and avoids the headache of trying to track per-step eigenvectors when eigenvalues are clustered or degenerate.

## Failed Experiments (What didn't work)

*   **Data Covariance ($Z$) Mixing**:
    *   *Idea*: Mix the data covariance $Z = X^T (XQ)$ into $Q$ to align with data distribution.
    *   *Failure*: $Z$ is computed *through* $Q$, so it lies largely within the existing subspace. It failed to find orthogonal directions, leading to mode collapse.
*   **Unclamped Eigen-Scaling**:
    *   *Idea*: Scale by $1/\sqrt{S}$ even for small $S$.
    *   *Failure*: Caused massive updates ($30\times$ boost) for new Scouts. While it accelerated learning, it caused numerical instability and crashes without Jitter.

## Status (Dec 11, 2025)

### Performance
*   **Memory**: ~1.5GB peak for 135M param model (vs >9GB baseline).
*   **Speed**: ~58k-72k tokens/sec on RTX 4070 Super.
*   **Convergence**:
    *   **Run 2 (Accidental High LR)**: Validation bpb **1.47**. (Scouts were boosted 30x).
    *   **Run 3 (Safe LR)**: Validation bpb **1.76**. (Scouts clamped to unit update).
    *   *Insight*: The aggressive boosting of random scouts might be necessary for them to find signal quickly.

### Key Files
*   `mole/linear.py`: `MoleLinear` layer. Stores $Q$ and handles projection.
*   `mole/grad.py`: Custom autograd function.
*   `mole/optimizer.py`: `MoleOptimizer`. Implements `_batched_eigh_update`, "Scout Renewal" logic, and Eigen-EMA.

## Historical Context
*   **Phase 1**: Foundation (MoleLinear, basic low-rank grad).
*   **Phase 2**: Debugging Mode Collapse.
    *   Attempt 1: Data Covariance ($Z$) mixing. Failed.
    *   Attempt 2: "Scout Renewal" (Explicit random exploration). **Success**.

## Usage
```bash
# Train with MoleGrad
uv run -m scripts.base_train --run=mole-test --matrix_lr=0.002
```
