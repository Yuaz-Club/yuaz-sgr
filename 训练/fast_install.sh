#!/bin/bash
# fast_install.sh — 清华镜像快速安装
export PATH="/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate yuaz 2>/dev/null || echo "no conda env, using system pip"

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple -q 2>&1 | tail -5
pip install torchcrepe soundfile pyyaml pretty_midi textgrid tqdm numpy librosa -i https://pypi.tuna.tsinghua.edu.cn/simple -q 2>&1 | tail -3

echo "=== verify ==="
python3 -c "import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')"
python3 -c "import torchcrepe; import librosa; print('all OK')"
echo "=== DONE ==="
