#!/bin/bash
# 重装 yuaz 环境的 PyTorch 和依赖
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

echo "=== 卸载旧 PyTorch ==="
pip uninstall -y torch torchaudio torchvision 2>/dev/null

echo "=== 安装 PyTorch (清华源 + CUDA 12.6) ==="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple -q

echo "=== 安装其他依赖 ==="
pip install torchcrepe soundfile pyyaml pretty_midi textgrid tqdm numpy librosa \
  -i https://pypi.tuna.tsinghua.edu.cn/simple -q

echo "=== 验证 ==="
python -c "import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')"
python -c "import torchcrepe; import librosa; print('all OK')"
echo "=== DONE ==="
