# MoleGrad Migration Guide (Legacy -> Current)

This document tracks the evolution of the MoleGrad optimizer and the rationale behind changes from the legacy implementation (`archive/mole_legacy`) to the current stable version (`mole/`).

## 1. Strategy Shift: "Z-Mixing" to "Scout Renewal"

### Legacy Approach (Z-Mixing)
*   **Concept**: Attempted to mix the data covariance $Z = X^T (XQ)$ into the subspace $Q$ to align it with the data distribution.
*   **Implementation**: `Q_next = (1 - alpha) * V + alpha * Z_ortho`
*   **Failure Mode**: "Mode Collapse". Since $Z$ is computed *through* $Q$, it lies largely within the existing subspace. The optimizer failed to find new orthogonal directions, causing the subspace to stagnate.

### Current Approach (Scout Renewal)
*   **Concept**: Explicitly force exploration by replacing the weakest direction with random noise.
*   **Implementation**:
    *   **Elites**: Top $k-1$ eigenvectors of gradient covariance ($Y^T Y$) are kept.
    *   **Scout**: Bottom 1 eigenvector is replaced by a random orthogonal vector.
*   **Result**: Continuous exploration of the null space. Solved mode collapse.

## 2. Numerical Stability

### Legacy Issues
*   `linalg.eigh` would crash on singular matrices (when a direction had 0 gradient signal).
*   No protection against division by zero when scaling updates.

### Current Fixes
*   **Jitter**: Added `1e-6 * I` to the covariance matrix $C = Y^T Y$ before decomposition. This ensures positive definiteness.
*   **Clamping**: Eigen-EMA denominator is clamped to `min=1.0` (or similar) to prevent exploding updates for new scouts.

## 3. Optimizer Dynamics (Eigen-EMA)

### Legacy
*   Standard SGD-like updates on the projected gradient.

### Current
*   **Eigen-EMA**: Tracks the exponential moving average of the eigenvalues ($S$).
*   **Hybrid Scaling**:
    *   **Strong Directions ($S > 1$)**: Scaled by $1/\sqrt{S}$ (Adafactor/RMSProp behavior) to stabilize learning.
    *   **Weak Directions ($S < 1$)**: Scaled by 1.0 (SGD behavior) to allow them to grow without noise amplification.

## 4. Code Cleanup
*   Removed `Z` calculation from `MoleLinear` (saved compute).
*   Removed `exploration_noise` parameter (replaced by explicit Scout replacement).
