#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
grep -n "tf32\|TF32\|float32_matmul\|allow_tf32" /root/autodl-tmp/yuaz_sgr/train/train.py
echo "---python---"
python3 -c "import torch; print('TF32:', torch.backends.cuda.matmul.allow_tf32); print('cudnn:', torch.backends.cudnn.allow_tf32); print('precision:', torch.get_float32_matmul_precision())"
echo "---gpu---"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "---step---"
grep "Step" /root/autodl-tmp/logs/train_nohup.out | tail -3
