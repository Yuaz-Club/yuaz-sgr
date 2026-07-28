"""诊断 dur_loss 高的根因"""
import os, sys, json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel

config = {}
for name in ['base', 'model']:
    import yaml
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']

device = 'cuda'
CKPT = sorted([f for f in os.listdir('/root/autodl-tmp/yuaz_sgr/checkpoints') if f.startswith('model_07')])[-1]
CKPT = '/root/autodl-tmp/yuaz_sgr/checkpoints/' + CKPT
print(f"CKPT: {CKPT}")

model = SVSModel(config).to(device)
model.eval()
ckpt = torch.load(CKPT, map_location=device)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    clean_k = k.replace('_orig_mod.', '')
    if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
        model_sd[clean_k] = v
model.load_state_dict(model_sd, strict=False)

from data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json',
    bs=1, device=device,
)

# 取几个样本看 dur_gt vs dur_pred
for i in range(3):
    batch = next(train_iter())
    audio = batch['audio']; ph = batch['phoneme_ids']; dur_gt = batch['durations'].float()
    f0_gt = batch['f0_audio']; nl = batch['note_lengths']
    
    with torch.no_grad():
        z1, f0_e = model.encoder(audio, f0_override=f0_gt)
        mu = model.cond_encoder(ph, dur_gt.long(), f0_e.squeeze(1))
        lag_pred, dur_pred = model.temporal_predictor(mu, f0_e.squeeze(1), dur_gt.long(), nl)
    
    mask = dur_gt[0] > 0
    gt = dur_gt[0][mask]
    pred = dur_pred[0][mask]
    
    mse = F.mse_loss(pred, gt).item()
    print(f"\n样本 {i}: {gt.shape[0]} 音素, MSE={mse:.1f}")
    print(f"  dur_gt:  mean={gt.mean():.1f} std={gt.std():.1f} min={gt.min():.1f} max={gt.max():.1f}")
    print(f"  dur_pred: mean={pred.mean():.1f} std={pred.std():.1f} min={pred.min():.1f} max={pred.max():.1f}")
    
    # 看 ratio
    ratio = pred / (gt + 1e-8)
    print(f"  pred/gt ratio: mean={ratio.mean():.2f} median={ratio.median():.2f}")
    print(f"  gt 示例: {gt[:10].tolist()}")
    print(f"  pred 示例: {pred[:10].tolist()}")

# 检查 temporal_predictor 参数
tp = model.temporal_predictor
for name, p in tp.named_parameters():
    print(f"\ntemporal_predictor.{name}: shape={list(p.shape)} mean={p.mean():.4f} std={p.std():.4f}")
