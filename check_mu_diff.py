#!/usr/bin/env python3
"""验证 cond_encoder 是否学会区分不同音素——跳过 DDSP，直接看 mu 的差异"""
import sys, os
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
import torch, yaml, json, numpy as np
from pathlib import Path

config = {}
for n in ['base', 'model', 'train']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{n}.yaml') as f:
        config.update(yaml.safe_load(f))

device = torch.device('cuda')
from models.svs_model import SVSModel
model = SVSModel(config).to(device)

ckpt = torch.load('/root/autodl-tmp/checkpoints/model_009000.pt', map_location=device, weights_only=False)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
model.load_state_dict(model_sd, strict=False)
model.eval()

with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json') as f:
    pm = json.load(f)
sr = config.get('sample_rate', 24000)

# 取 5 段音素序列差异大的
npz_files = sorted(Path('/root/autodl-tmp/yuaz_sgr/data/preprocessed/train').glob('*.npz'))
# 挑 5 段来自不同歌的
selected = [npz_files[i] for i in [0, 100, 200, 500, 1000]]

print("cond_encoder mu 差异分析（9000步）")
print("=" * 70)
print(f"{'idx':>3} | {'file':>14} | {'ph_count':>7} | {'mu_mean':>7} | {'mu_std':>6} | mu 前 5 维")
print("-" * 70)

mus = []
info = []
for i, npz_path in enumerate(selected):
    data = np.load(str(npz_path), allow_pickle=True)
    phonemes = data['phonemes'].tolist()
    durations = torch.LongTensor(data['durations']).unsqueeze(0).to(device)
    f0_gt = torch.FloatTensor(data['f0']).unsqueeze(0).to(device)
    ph_ids = torch.LongTensor([pm['p2i'].get(p, 0) for p in phonemes]).unsqueeze(0).to(device)

    with torch.no_grad():
        mu = model.cond_encoder(ph_ids, durations, f0_gt)
    mus.append(mu)
    info.append(f"{npz_path.stem}")
    ph_short = ' '.join(phonemes[:5])
    print(f"{i:3d} | {npz_path.stem:>14} | {mu.shape[-1]:>7} | {mu.mean().item():>7.4f} | {mu.std().item():>6.4f} | {mu[0,:5].cpu().numpy().round(3)}")
    print(f"    | 音素前5: {ph_short}{'...' if len(phonemes)>5 else ''}")

# 计算不同段之间 mu 的余弦相似度
print("\n段间 mu 余弦相似度矩阵（<1.0 = 不同）")
print("     ", end="")
for j in range(len(selected)):
    print(f"    s{j}", end="")
print()
for i in range(len(selected)):
    print(f"  s{i} ", end="")
    for j in range(len(selected)):
        mi = mus[i].flatten()
        mj = mus[j].flatten()
        sim = torch.nn.functional.cosine_similarity(mi.unsqueeze(0), mj.unsqueeze(0)).item()
        print(f"  {sim:.3f}", end="")
    print()
print("\n✅ 如果余弦相似度 < 0.99，说明不同音素序列产生了不同 mu")
print("✅ 如果所有相似度都接近 1.0，说明 cond_encoder 还没学到音素差异")
