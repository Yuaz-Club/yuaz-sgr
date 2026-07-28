#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz
python -c "
import torch
ckpt = torch.load('/root/autodl-tmp/base_checkpoint.pt', map_location='cpu')
print(type(ckpt))
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print(keys[:15])
    for k in keys[:3]:
        v = ckpt[k]
        if isinstance(v, dict):
            print(f'  {k}: dict with {len(v)} keys')
        elif hasattr(v, 'shape'):
            print(f'  {k}: shape {v.shape}')
        else:
            print(f'  {k}: {type(v).__name__} = {v}')
"
