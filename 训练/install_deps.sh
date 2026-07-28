#!/bin/bash
# install_deps.sh
set -e
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/miniconda3/bin:$PATH"

echo "=== installing torch ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -q

echo "=== installing pip deps ==="
pip install torchcrepe soundfile pyyaml pretty_midi textgrid tqdm numpy librosa -q

echo "=== verify ==="
python -c "
import torch; import torchcrepe; import librosa
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "=== DONE ==="
