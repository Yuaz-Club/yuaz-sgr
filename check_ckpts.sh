#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/autodl-tmp/yuaz_sgr
for ckpt_path in /root/autodl-tmp/checkpoints/model_*.pt; do
    ckpt_name=$(basename "$ckpt_path")
    step=$(python -c "import torch; c=torch.load('$ckpt_path', map_location='cpu', weights_only=False); print(c.get('step','?'))" 2>/dev/null)
    env_shape=$(python -c "import torch; c=torch.load('$ckpt_path', map_location='cpu', weights_only=False); print(tuple(c['model']['ddsp_decoder.env_net.2.weight'].shape))" 2>/dev/null)
    echo "$ckpt_name  step=$step  env_net.2=$env_shape"
done
