#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/autodl-tmp/yuaz_sgr
python -c "
import torch, sys
sys.path.insert(0, '.')
from models.svs_model import SVSModel
import yaml

# 加载 config
config = {}
for name in ['config/base.yaml', 'config/model.yaml', 'config/train.yaml']:
    with open(name) as f:
        config.update(yaml.safe_load(f))

# 创建模型
model = SVSModel(config)
print(f'Total params: {sum(p.numel() for p in model.parameters())}')

# 加载 checkpoint
ckpt = torch.load('/root/autodl-tmp/checkpoints/model_110503.pt', map_location='cpu', weights_only=False)
model_sd = model.state_dict()
loaded = 0
skipped = []
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        loaded += 1
    else:
        skipped.append(k)
print(f'Loaded: {loaded}, Skipped: {len(skipped)}')
for k in skipped:
    if k in model_sd:
        print(f'  SKIP: {k}  ckpt={tuple(ckpt[\"model\"][k].shape)}  model={tuple(model_sd[k].shape)}')
    else:
        print(f'  MISSING from model: {k}  shape={tuple(ckpt[\"model\"][k].shape)}')
for k in model_sd:
    if k not in ckpt['model']:
        print(f'  NEW in model: {k}  shape={tuple(model_sd[k].shape)}')
"
