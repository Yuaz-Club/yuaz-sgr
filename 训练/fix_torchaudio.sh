#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

echo "=== 检查 torchaudio lib ==="
ls -la /root/miniconda3/envs/yuaz/lib/python3.10/site-packages/torchaudio/lib/

echo "=== PyTorch version ==="
python -c "import torch; print(torch.__version__)"

echo "=== 检查 torchaudio .so ==="
python -c "
import os
libdir = '/root/miniconda3/envs/yuaz/lib/python3.10/site-packages/torchaudio/lib'
for f in os.listdir(libdir):
    print(f, os.path.getsize(os.path.join(libdir, f)))
"

echo "=== 尝试单独 import torchaudio ==="
python -c "
import sys
sys.path.insert(0, '/root/miniconda3/envs/yuaz/lib/python3.10/site-packages')
try:
    import torchaudio
    print('torchaudio imported OK')
except Exception as e:
    print(f'Error: {e}')
"
