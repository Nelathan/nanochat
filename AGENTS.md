# MoleGrad Project - Agent Context

> See [NOVA.md](./NOVA.md) for agent persona and interaction style.

## The Project: MoleGrad

**See [mole/README.md](./mole/README.md) for the latest technical documentation and strategy.**

### Core Concept
A novel low-rank gradient method for training neural networks. Instead of computing full gradient $G = B^T A$, we store $AQ$ (small) and compute gradients in a subspace $Q$.

### Current State (Dec 11, 2025)
- **Strategy**: "Scout Renewal" (formerly "Kill the Runt") + Eigen-EMA.
- **Status**: Working, stable, memory efficient.
- **Latest Result**: Validation bpb 1.47 (Run 2, with high LR boost).

### Key Files
- `mole/linear.py`: Core layer logic.
- `mole/optimizer.py`: Optimizer logic (Scout Renewal, Eigen-EMA).
- `nanochat/gpt.py`: Model definition.
- `scripts/base_train.py`: Training script.

## How to Work with Me (User Preferences)

1.  **Rigorous Documentation**:
    *   The user intends to publish this work. Documentation must be scientific, precise, and honest.
    *   **Document Failures**: Always document *why* something didn't work (e.g., Z-mixing). Negative results are valuable.
    *   **Explain Non-Standard Techniques**: If we use a trick (like Jitter or Eigen-EMA), explain *why* it's there and what problem it solves.

2.  **Metaphor Consistency**:
    *   Use metaphors that fit the project theme ("Mole", "Tunnels", "Scouts") but don't force them if they are confusing.
    *   Avoid violent or crude names (e.g., "Kill the Runt" was rejected).

3.  **Communication**:
    *   **Cut Ambiguity**: If a request is vague, ask for clarification or propose a specific plan.
    *   **Context Awareness**: Before starting a task, check `mole/README.md` for the latest technical state.

4.  **Respect for Legacy & Continuity**:
    *   **Don't Erase History**: When refactoring, create migration guides (like `mole/MIGRATION.md`) instead of just overwriting.
    *   **Coherent Comments**: Avoid adding messy, repetitive comments. Ensure new comments match the quality and depth of the legacy code.
    *   **Finish the Job**: Do not leave documentation in a half-baked state. Clean up after refactors.

---
*This file helps AI agents understand the project context when starting a new chat.*
