#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
python3 << 'PYEOF'
import torch
print("=== TF32 test ===")
torch.set_float32_matmul_precision("high")
print("high -> TF32:", torch.backends.cuda.matmul.allow_tf32, "precision:", torch.get_float32_matmul_precision())
torch.set_float32_matmul_precision("medium")
print("medium -> TF32:", torch.backends.cuda.matmul.allow_tf32, "precision:", torch.get_float32_matmul_precision())
torch.backends.cuda.matmul.allow_tf32 = True
print("forced -> TF32:", torch.backends.cuda.matmul.allow_tf32)
PYEOF
