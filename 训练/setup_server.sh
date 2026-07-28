#!/bin/bash
# setup_server.sh — 在李老师服务器上安装环境
# 用法: ssh -p 45349 root@connect.bjb1.seetacloud.com "bash -s" < setup_server.sh

set -e

echo "=== 安装 Miniconda ==="
if [ ! -d "/root/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
fi
export PATH="/root/miniconda3/bin:$PATH"

echo "=== 创建环境 ==="
conda create -n yuaz python=3.10 -y 2>/dev/null || echo "env exists"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

echo "=== 安装依赖 ==="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 -q
pip install librosa torchcrepe soundfile pyyaml pretty_midi textgrid tqdm numpy -q

echo "=== 验证 ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torchcrepe; print('torchcrepe OK')"
python -c "import librosa; print('librosa OK')"

echo "=== 环境就绪 ==="
