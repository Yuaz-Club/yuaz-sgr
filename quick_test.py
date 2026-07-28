#!/usr/bin/env python3
"""
快速验证: 用训练中的模型重建几段真实音频
如果重建后能听出不同的发音, 说明模型学会了音素区分
"""
import sys, os
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
import torch, yaml, json, numpy as np
import soundfile as sf
from pathlib import Path
from models.svs_model import SVSModel

config = {}
for n in ['base', 'model', 'train']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{n}.yaml') as f:
        config.update(yaml.safe_load(f))

device = torch.device('cuda')

# 找最新 checkpoint
ckpts = sorted(Path('/root/autodl-tmp/checkpoints').glob('model_*.pt'))
if not ckpts:
    print("No checkpoint found!")
    sys.exit(1)

ckpt_path = ckpts[-1]
print(f"使用: {ckpt_path}")

ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
model = SVSModel(config).to(device)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
model.load_state_dict(model_sd, strict=False)
model.eval()
print(f"加载完成. 步数: {ckpt.get('step', '?')}")

# 找 3 段不同音素密度的训练数据
train_dir = Path('/root/autodl-tmp/yuaz_sgr/data/preprocessed/train')
npz_files = sorted(train_dir.glob('*.npz'))
print(f"训练数据: {len(npz_files)} 段")

sr = config.get('sample_rate', 24000)

out_dir = '/root/autodl-tmp/inference_output'
os.makedirs(out_dir, exist_ok=True)

for i, npz_path in enumerate(npz_files[:5]):
    data = np.load(str(npz_path), allow_pickle=True)
    phonemes = data['phonemes'].tolist()
    
    # 用完整的 4 秒音频 (含填充静音, pyworld 能处理)
    audio = torch.FloatTensor(data['audio']).unsqueeze(0).to(device)
    
    with torch.no_grad():
        z1, f0 = model.encoder(audio)
        
        # 2) RVQ 量化 (训练时 DDSP 看到的)
        z_q, _, _ = model.rvq(z1)
        
        # 3) DDSP 解码
        wav_recon = model.ddsp_decoder(f0, z_q, target_samples=audio.shape[-1])
        
        # 4) 只保留 F0>0 的区域
        f0_np = f0.squeeze().cpu().numpy()
        wav = wav_recon.squeeze().cpu().numpy()
        silence_mask = np.zeros_like(f0_np)
        silence_mask[f0_np > 1.0] = 1.0
        hop = 256
        audio_mask = np.repeat(silence_mask, hop)[:len(wav)]
        audio_mask = np.concatenate([audio_mask, np.ones(max(0, len(wav) - len(audio_mask)))])
        wav = wav * audio_mask
    
    wav_path = os.path.join(out_dir, f'recon_{npz_path.stem}.wav')
    sf.write(wav_path, wav, sr)
    
    # 打印音素和统计
    ph_str = ' '.join(phonemes[:10])
    print(f"[{i+1}] {npz_path.stem}: {len(phonemes)} phonemes | f0非零={int((f0_np>1).sum())}/{len(f0_np)} | saved={wav_path}")
    print(f"     音素: {ph_str}{'...' if len(phonemes)>10 else ''}")

print(f"\n完成! 文件在 {out_dir}")
print("下载到本地听听重建质量")
