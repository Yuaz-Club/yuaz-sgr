#!/usr/bin/env python3
"""快速验证: 用训练 checkpoint 的 cond_encoder + DDSP 直接合成"""
import sys, os
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
import torch, yaml, json, numpy as np
import soundfile as sf
from pathlib import Path

# 加载配置+模型
config = {}
for n in ['base', 'model', 'train']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{n}.yaml') as f:
        config.update(yaml.safe_load(f))

device = torch.device('cuda')
from models.svs_model import SVSModel
model = SVSModel(config).to(device)

ckpt = torch.load('/root/autodl-tmp/checkpoints/model_004000.pt', map_location=device, weights_only=False)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
model.load_state_dict(model_sd, strict=False)
model.eval()
print(f"步数: {ckpt.get('step', '?')}")

with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json') as f:
    pm = json.load(f)
sr = config.get('sample_rate', 24000)

# 取 3 段训练数据
npz_files = sorted(Path('/root/autodl-tmp/yuaz_sgr/data/preprocessed/train').glob('*.npz'))[:3]
out_dir = '/root/autodl-tmp/inference_output'
os.makedirs(out_dir, exist_ok=True)

for i, npz_path in enumerate(npz_files):
    data = np.load(str(npz_path), allow_pickle=True)
    audio = torch.FloatTensor(data['audio']).unsqueeze(0).to(device)
    phonemes = data['phonemes'].tolist()
    durations = torch.LongTensor(data['durations']).unsqueeze(0).to(device)
    f0_gt = torch.FloatTensor(data['f0']).unsqueeze(0).to(device)
    ph_ids = torch.LongTensor([pm['p2i'].get(p, 0) for p in phonemes]).unsqueeze(0).to(device)

    with torch.no_grad():
        # cond_encoder → mu → DDSP (跳过 encoder, 用真实的 f0)
        mu = model.cond_encoder(ph_ids, durations, f0_gt)
        wav = model.ddsp_decoder(f0_gt, mu, target_samples=audio.shape[-1])

        f0_np = f0_gt.squeeze().cpu().numpy()
        wav_np = wav.squeeze().cpu().numpy()
        mask = np.repeat((f0_np > 1).astype(np.float32), 256)[:len(wav_np)]
        mask = np.concatenate([mask, np.ones(max(0, len(wav_np)-len(mask)))])
        wav_np = wav_np * mask

    wav_path = os.path.join(out_dir, f'cond_{npz_path.stem}.wav')
    sf.write(wav_path, wav_np, sr)
    ph_str = ' '.join(phonemes[:8])
    print(f"[{i+1}] {npz_path.stem}: {len(phonemes)} ph: {ph_str}{'...' if len(phonemes)>8 else ''}")
    print(f"      mu.std={mu.std().item():.4f}  f0非零={int((f0_np>1).sum())}/{len(f0_np)}")
    print(f"      -> {wav_path}")

print(f"\n完成! 下载听听不同音素是否不同")
