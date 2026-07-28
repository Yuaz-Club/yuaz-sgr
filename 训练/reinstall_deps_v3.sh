#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

echo "=== 强制重装 PyTorch (cu126) ==="
pip install torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu126 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --force-reinstall 2>&1

echo "=== 安装其他库 ==="
pip install torchcrepe soundfile pyyaml pretty_midi textgrid tqdm numpy librosa \
  -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1

echo "=== 验证 ==="
DISABLE_ITT=1 python -c "
import torch
print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')
import torch.nn
print('torch.nn OK')
import torchcrepe
import librosa
print('all deps OK')
"
echo "=== DONE ==="
