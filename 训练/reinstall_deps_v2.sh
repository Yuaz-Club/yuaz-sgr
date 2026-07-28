#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

echo "=== pip version ==="
pip --version

echo "=== 安装 PyTorch (详细输出) ==="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1

echo "=== 验证 ==="
python -c "import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')" 2>&1
python -c "import torch.nn; print('torch.nn OK')" 2>&1
python -c "import torchcrepe; import librosa; print('all deps OK')" 2>&1
