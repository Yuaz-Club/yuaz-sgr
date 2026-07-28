#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

cd /root/autodl-tmp

CKPT=${1:-finetune_checkpoints/model_012000.pt}
SAMPLE=${2:-DIyitian_wav_003}

python -c "
import os, sys, json, yaml
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, 'yuaz_sgr')
from models.svs_model import SVSModel

config_dir = 'yuaz_sgr/config'
data_dir = 'data/preprocessed/test'
ckpt_path = '$CKPT'
sample_name = '$SAMPLE'
device = 'cuda'

config = {}
for name in ['base', 'model']:
    path = os.path.join(config_dir, f'{name}.yaml')
    with open(path) as f:
        config.update(yaml.safe_load(f))
finetune_path = os.path.join(config_dir, 'finetune.yaml')
if os.path.exists(finetune_path):
    with open(finetune_path) as f:
        config.update(yaml.safe_load(f))

pm_path = 'data/preprocessed/phoneme_map.json'
with open(pm_path) as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']

model = SVSModel(config).to(device)
model.eval()

ckpt = torch.load(ckpt_path, map_location=device)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    clean_k = k.replace('_orig_mod.', '')
    if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
        model_sd[clean_k] = v
    elif clean_k in model_sd and v.dim() >= 2 and v.shape[1:] == model_sd[clean_k].shape[1:]:
        n_base = v.shape[0]
        model_sd[clean_k][:n_base] = v
model.load_state_dict(model_sd, strict=False)

data = np.load(os.path.join(data_dir, f'{sample_name}.npz'), allow_pickle=True)
gt_f0 = data['f0'].astype(np.float32)
gt_durations = data['durations'].astype(np.int32)
phoneme_strs = data['phonemes'].tolist()
note_lengths = data['note_lengths'].astype(np.int32)

sr = config.get('sample_rate', 24000)
hop_length = config.get('hop_length', 256)

ph_ids = [pm['p2i'].get(p, 0) for p in phoneme_strs]
total_frames = int(gt_durations.sum())
f0 = gt_f0[:total_frames]
if len(f0) < total_frames:
    f0 = np.pad(f0, (0, total_frames - len(f0)))

ph_t = torch.LongTensor(ph_ids).unsqueeze(0).to(device)
dur_t = torch.LongTensor(gt_durations).unsqueeze(0).to(device)
f0_t = torch.FloatTensor(f0).unsqueeze(0).unsqueeze(0).to(device)
nl_t = torch.LongTensor(note_lengths).unsqueeze(0).to(device)

print(f'=== F0 分析 ===')
print(f'GT F0: range=[{f0.min():.1f}, {f0.max():.1f}] mean={f0.mean():.1f} Hz')
nonzero_f0 = f0[f0 > 50]
print(f'非静音 F0: range=[{nonzero_f0.min():.1f}, {nonzero_f0.max():.1f}] mean={nonzero_f0.mean():.1f} Hz')
print(f'F0>0 帧: {len(nonzero_f0)}/{len(f0)} ({len(nonzero_f0)/len(f0)*100:.1f}%)')

# 完整合成: 用预测F0
mu = model.cond_encoder(ph_t, dur_t, f0_t.squeeze(1))
with torch.no_grad():
    z = model.flow_gen.sample(mu, n_steps=1)
    if isinstance(z, tuple):
        z = z[0]
    z_q, _, _ = model.rvq(z)
    
    # 版本1: 用 GT F0
    f0_gt = f0_t
    if f0_gt.shape[-1] != z_q.shape[-1]:
        f0_gt = torch.nn.functional.interpolate(f0_gt, size=z_q.shape[-1], mode='linear')
    wav_gt_f0 = model.ddsp_decoder(f0_gt, z_q)
    
    # 版本2: 用预测 F0
    f0_pred = model.f0_predictor(mu) * 2000.0
    f0_pred = torch.clamp(f0_pred, min=50.0, max=2000.0)
    if f0_pred.shape[-1] != z_q.shape[-1]:
        f0_pred = torch.nn.functional.interpolate(f0_pred, size=z_q.shape[-1], mode='linear')
    wav_pred_f0 = model.ddsp_decoder(f0_pred, z_q)
    
    # 版本3: AE 重建 (encoder → rvq → decoder)
    audio_orig = data['audio'].astype(np.float32)
    target_samples = len(audio_orig)
    audio_t = torch.FloatTensor(audio_orig).unsqueeze(0).unsqueeze(0).to(device)
    # 用原声提取 f0
    z1, f0_enc = model.encoder(audio_t)
    z_q_ae, _, _ = model.rvq(z1)
    f0_ae = f0_enc
    if f0_ae.shape[-1] != z_q_ae.shape[-1]:
        f0_ae = torch.nn.functional.interpolate(f0_ae, size=z_q_ae.shape[-1], mode='linear')
    wav_ae = model.ddsp_decoder(f0_ae, z_q_ae, target_samples=target_samples)

for name, w in [('GT_F0', wav_gt_f0), ('Pred_F0', wav_pred_f0), ('AE_recon', wav_ae)]:
    w_np = w.squeeze().cpu().numpy()
    rms = np.sqrt(np.mean(w_np ** 2))
    peak = np.abs(w_np).max()
    print(f'{name}: RMS={rms:.4f} peak={peak:.4f}')
    sf.write(f'{sample_name}_{name.lower()}.wav', w_np, sr)

print('\\n=== 诊断完成 ===')
print(f'文件: {sample_name}_gt_f0.wav, {sample_name}_pred_f0.wav, {sample_name}_ae_recon.wav')
" 2>&1
