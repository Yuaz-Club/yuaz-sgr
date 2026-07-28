#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz
python -c "import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')"
python -c "import torchcrepe; import librosa; print('all deps OK')"
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
