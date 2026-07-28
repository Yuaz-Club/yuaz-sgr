#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
python3 -c "
import torch
c = torch.load('/root/autodl-tmp/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
print('best_model.pt: step=' + str(c.get('step', '?')))
c2 = torch.load('/root/autodl-tmp/checkpoints/best_val_model.pt', map_location='cpu', weights_only=False)
print('best_val_model.pt: step=' + str(c2.get('step', '?')))
env_w = c2['model']['ddsp_decoder.env_net.2.weight']
print('env_net.2 shape=' + str(tuple(env_w.shape)))
"
