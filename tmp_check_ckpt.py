import torch
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_083000.pt', map_location='cpu', weights_only=False)
for k in sorted(ckpt['model'].keys()):
    if 'cond_fuse' in k or 'weight_net' in k:
        print(k, ckpt['model'][k].shape)
