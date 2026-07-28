#!/usr/bin/env python3
"""
CPU重建测试 - 不停训练, 用最新 checkpoint
Usage: python recon_cpu.py [--ckpt /path/to/model_N.pt]
"""
import sys, os, glob, argparse
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
import torch, yaml, json, numpy as np
import soundfile as sf
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=None,
                    help='指定 checkpoint, 默认用最新')
args = parser.parse_args()

device = torch.device('cpu')
print(f"设备: {device}")

config = {}
for n in ['base', 'model', 'train']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{n}.yaml') as f:
        config.update(yaml.safe_load(f))

from models.svs_model import SVSModel
model = SVSModel(config).to(device)
model.eval()

if args.ckpt and os.path.exists(args.ckpt):
    ckpt_path = args.ckpt
else:
    ckpt_files = sorted(glob.glob('/root/autodl-tmp/checkpoints/model_*.pt'))
    if not ckpt_files:
        print("错误: 未找到 checkpoint!")
        sys.exit(1)
    ckpt_path = ckpt_files[-1]

print(f"Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
model.load_state_dict(model_sd, strict=False)
print(f"步数: {ckpt.get('step', '?')}")

sr = config.get('sample_rate', 24000)
npz_files = sorted(Path('/root/autodl-tmp/yuaz_sgr/data/preprocessed/test').glob('*.npz'))[:5]
out_dir = '/root/autodl-tmp/inference_output'
os.makedirs(out_dir, exist_ok=True)

for i, npz_path in enumerate(npz_files):
    data = np.load(str(npz_path), allow_pickle=True)
    audio_np = data['audio'].astype(np.float32)

    audio = torch.FloatTensor(audio_np).unsqueeze(0).unsqueeze(0).to(device)

    ref_path = os.path.join(out_dir, f'ref_{npz_path.stem}.wav')
    sf.write(ref_path, audio_np, sr)

    with torch.no_grad():
        f0_audio = torch.FloatTensor(data['f0_audio'].astype(np.float32)).unsqueeze(0).to(device)
        z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
        z_q, _, _ = model.rvq(z1)
        wav_recon = model.ddsp_decoder(f0_enc, z_q, target_samples=audio.shape[-1])

    wav = wav_recon.squeeze().cpu().numpy()
    recon_path = os.path.join(out_dir, f'recon_{npz_path.stem}.wav')
    sf.write(recon_path, wav, sr)

    print(f"[{i+1}] {npz_path.stem}")
    print(f"      ref={ref_path} ({len(audio_np)/sr:.2f}s)")
    print(f"      recon={recon_path}")
    f0_np = f0_enc.squeeze().cpu().numpy()
    print(f"      f0非零={int((f0_np>1).sum())}/{len(f0_np)}")

print(f"\ndone. 下载 {out_dir} 下的 wav 听听重建效果")
