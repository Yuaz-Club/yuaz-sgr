#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

echo "=== 彻底清除旧 torchaudio ==="
pip uninstall -y torchaudio 2>/dev/null
rm -rf /root/miniconda3/envs/yuaz/lib/python3.10/site-packages/torchaudio

echo "=== 重新安装 torchaudio ==="
pip install torchaudio --index-url https://download.pytorch.org/whl/cu126 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall 2>&1

echo "=== 验证 torchaudio ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
try:
    import torchaudio
    print(f'torchaudio {torchaudio.__version__} OK')
except Exception as e:
    print(f'Error: {e}')
"

echo "=== 验证 torchcrepe ==="
python -c "
try:
    import torchcrepe
    print('torchcrepe OK')
except Exception as e:
    print(f'Error: {e}')
"
echo "=== DONE ==="
