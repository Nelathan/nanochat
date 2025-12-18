
import torch
import torch.nn as nn
from mole import MoleLinear

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

print(f"Testing MoleGrad with torch.compile on {device}...")

# 1. Create Model
model = MoleLinear(32, 32, rank_k=4).to(device)
model.init_Q() # Important: Initialize the buffers

# 2. Compile with fullgraph=True (strict mode)
print("Compiling...")
opt_model = torch.compile(model, fullgraph=True)

# 3. Run Forward/Backward
x = torch.randn(16, 32, device=device)
print("Running Forward...")
y = opt_model(x)
loss = y.sum()

print("Running Backward...")
loss.backward()

# 4. Verify Gradients
print("Verifying Gradients...")
if model.mole_grad is None:
    print("❌ FAIL: mole_grad is None")
elif model.mole_grad.abs().sum() == 0:
    print("⚠️ WARNING: mole_grad is all zeros (might be correct if inputs were zero, but unlikely)")
else:
    print(f"✅ SUCCESS: mole_grad populated. Shape: {model.mole_grad.shape}")
    print("Graph capture successful!")
