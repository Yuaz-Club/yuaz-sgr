"""快速诊断 weight_net gate 值"""
import os, sys, json
import torch
import numpy as np
import soundfile as sf

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

device = 'cuda'; sr = 24000
CKPT = sorted([f for f in os.listdir('/root/autodl-tmp/yuaz_sgr/checkpoints') if f.startswith('model_0')])[-1]
CKPT = '/root/autodl-tmp/yuaz_sgr/checkpoints/' + CKPT
OUT = '/root/autodl-tmp/diag_gate'
os.makedirs(OUT, exist_ok=True)
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

# 检查 weight_net
wn = model.ddsp_decoder.weight_net
for name, param in wn.named_parameters():
    print(f"weight_net.{name}: shape={list(param.shape)} mean={param.mean():.6f} std={param.std():.6f}")

# 取样本
from data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json',
    bs=1, device=device,
)
batch = next(train_iter())
audio = batch['audio']; f0_gt = batch['f0_audio']; T_audio = audio.shape[-1]

sf.write(f'{OUT}/orig.wav', audio.squeeze().cpu().numpy() / (audio.abs().max()+1e-8).item(), sr)

with torch.no_grad():
    z1, f0_e = model.encoder(audio, f0_override=f0_gt)
    print(f"\nf0_e: mean={f0_e.mean():.1f} std={f0_e.std():.1f}")
    print(f"z1: mean={z1.mean():.4f} std={z1.std():.4f}")
    
    z_q, _, _ = model.rvq(z1)
    
    # 标准解码 (看 gate 值)
    wav = model.ddsp_decoder(f0_e, z_q, target_samples=T_audio)
    
    # 手动计算 gate 值
    feat = torch.cat([z_q, f0_e], dim=1).transpose(1, 2)
    h = model.ddsp_decoder.emformer_input_proj(feat)
    h, _ = model.ddsp_decoder.emformer(h)
    h = model.ddsp_decoder.emformer_proj(h).transpose(1, 2)
    gate = model.ddsp_decoder.weight_net(h.transpose(1, 2)).transpose(1, 2)
    # gate 已经过 Sigmoid
    print(f"\ngate (λ_h): mean={gate.mean():.6f} std={gate.std():.6f}")
    print(f"λ_n = 1-gate: mean={1-gate.mean():.6f}")
    
    # 手动解码: 强制 gate=0.7 (更多谐波)
    # 不能简单用 forward, 得手动跑一遍
    S_c = model.ddsp_decoder.env_net(h.transpose(1,2)).transpose(1,2)
    A_c = model.ddsp_decoder.ap_net(h.transpose(1,2)).transpose(1,2)
    print(f"S_c: mean={S_c.mean():.4f} std={S_c.std():.4f}")
    print(f"A_c: mean={A_c.mean():.4f} std={A_c.std():.4f}")

wav_np = wav.squeeze().cpu().numpy()
wav_np = wav_np / (np.abs(wav_np).max()+1e-8)
sf.write(f'{OUT}/ae_default.wav', wav_np, sr)
print(f"\n  ae_default.wav — 标准解码 (gate={gate.mean():.4f})")
