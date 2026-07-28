#!/bin/bash
# install_v2.sh — 使用清华镜像加速
set -e
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/miniconda3/bin:$PATH"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -q
pip install torchcrepe soundfile pyyaml pretty_midi textgrid tqdm numpy librosa -q

python -c "
import torch; import torchcrepe; import librosa
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

echo "=== DONE ==="
