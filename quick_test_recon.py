#!/usr/bin/env python3
"""验证AE重建 - 自动用最新ckpt"""
import sys, os, glob
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
import torch, yaml, json, numpy as np
import soundfile as sf
from pathlib import Path

config = {}
for n in ['base', 'model', 'train']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{n}.yaml') as f:
        config.update(yaml.safe_load(f))

device = torch.device('cuda')
from models.svs_model import SVSModel
model = SVSModel(config).to(device)

ckpt_files = sorted(glob.glob('/root/autodl-tmp/checkpoints/model_*.pt'))
if not ckpt_files:
    print("错误: 未找到 checkpoint!")
    sys.exit(1)
latest_ckpt = ckpt_files[-1]
print(f"使用最新 checkpoint: {latest_ckpt}")
ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
model.load_state_dict(model_sd, strict=False)
model.eval()
print(f"步数: {ckpt.get('step', '?')}")

sr = config.get('sample_rate', 24000)
npz_files = sorted(Path('/root/autodl-tmp/yuaz_sgr/data/preprocessed/train').glob('*.npz'))[:3]
out_dir = '/root/autodl-tmp/inference_output'
os.makedirs(out_dir, exist_ok=True)

for i, npz_path in enumerate(npz_files):
    data = np.load(str(npz_path), allow_pickle=True)
    audio_np = data['audio'].astype(np.float32)

    audio = torch.FloatTensor(audio_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)

    # 原音频 reference
    ref_path = os.path.join(out_dir, f'ref_{npz_path.stem}.wav')
    sf.write(ref_path, audio_np, sr)

    with torch.no_grad():
        f0_npz = data.get('f0', None)
        f0_pre = torch.FloatTensor(f0_npz.astype(np.float32)).unsqueeze(0).to(device) if f0_npz is not None else None
        z1, f0_enc = model.encoder(audio, f0_override=f0_pre)
        z_q, _, _ = model.rvq(z1)
        wav_recon = model.ddsp_decoder(f0_enc, z_q, target_samples=audio.shape[-1])

    wav = wav_recon.squeeze().cpu().numpy()
    recon_path = os.path.join(out_dir, f'recon_{npz_path.stem}.wav')
    sf.write(recon_path, wav, sr)

    print(f"[{i+1}] {npz_path.stem}")
    print(f"      ref={ref_path} ({len(audio_np)/sr:.2f}s)")
    print(f"      recon={recon_path}")
    print(f"      z1.std={z1.std().item():.4f}  z_q.std={z_q.std().item():.4f}")
    f0_np = f0_enc.squeeze().cpu().numpy()
    print(f"      f0非零={int((f0_np>1).sum())}/{len(f0_np)}")

print(f"\n下载 ref_*.wav 和 recon_*.wav 对比听听重建效果")
print(f"如果重建能还原原音频内容说明AE在工作")
