#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

cd /root/autodl-tmp

CKPT=${1:-finetune_checkpoints/model_012000.pt}
SAMPLE=${2:-DIyitian_wav_000}

python -c "
import os, sys, json, yaml
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, 'yuaz_sgr')
from models.svs_model import SVSModel

# 配置
config_dir = 'yuaz_sgr/config'
data_dir = 'data/preprocessed/test'
ckpt_path = '$CKPT'
sample_name = '$SAMPLE'
device = 'cuda'

# 加载配置
config = {}
for name in ['base', 'model']:
    path = os.path.join(config_dir, f'{name}.yaml')
    with open(path) as f:
        config.update(yaml.safe_load(f))
finetune_path = os.path.join(config_dir, 'finetune.yaml')
if os.path.exists(finetune_path):
    with open(finetune_path) as f:
        config.update(yaml.safe_load(f))

# 音素数
pm_path = 'data/preprocessed/phoneme_map.json'
with open(pm_path) as f:
    pm = json.load(f)
if 'conditioning' not in config:
    config['conditioning'] = {}
config['conditioning']['num_phonemes'] = pm['num_phonemes']
print(f'音素数: {pm[\"num_phonemes\"]}')

# 模型
model = SVSModel(config).to(device)
model.eval()

# 加载 checkpoint
ckpt = torch.load(ckpt_path, map_location=device)
model_sd = model.state_dict()
loaded = 0
skipped = 0
for k, v in ckpt['model'].items():
    clean_k = k.replace('_orig_mod.', '')
    if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
        model_sd[clean_k] = v
        loaded += 1
    elif clean_k in model_sd and v.dim() >= 2 and v.shape[1:] == model_sd[clean_k].shape[1:]:
        n_base = v.shape[0]
        model_sd[clean_k][:n_base] = v
        loaded += 1
    else:
        skipped += 1
model.load_state_dict(model_sd, strict=False)
step = ckpt.get('step', 0)
print(f'Checkpoint step={step}: {loaded} 加载, {skipped} 跳过')

# 加载测试样本
data_path = os.path.join(data_dir, f'{sample_name}.npz')
print(f'加载: {data_path}')
data = np.load(data_path, allow_pickle=True)

audio = data['audio'].astype(np.float32)
gt_f0 = data['f0'].astype(np.float32)
gt_durations = data['durations'].astype(np.int32)
phoneme_strs = data['phonemes'].tolist()
note_lengths = data['note_lengths'].astype(np.int32)
lags = data['lags'].astype(np.float32)

sr = config.get('sample_rate', 24000)
hop_length = config.get('hop_length', 256)

# 音素字符串 → ID
ph_ids = [pm['p2i'].get(p, 0) for p in phoneme_strs]
print(f'音素: {phoneme_strs}')
print(f'音素ID: {ph_ids}')
print(f'时长(帧): {gt_durations.tolist()}')
print(f'F0帧数: {len(gt_f0)}')
print(f'note_lengths: {note_lengths.tolist()}')
total_frames = int(gt_durations.sum())
print(f'总帧数: {total_frames}')

# 裁剪/补齐 F0 到总帧数
f0 = gt_f0[:total_frames]
if len(f0) < total_frames:
    f0 = np.pad(f0, (0, total_frames - len(f0)))

# 裁剪 lags
lags = lags[:total_frames]
if len(lags) < total_frames:
    lags = np.pad(lags, (0, total_frames - len(lags)))

# Tensor
ph_t = torch.LongTensor(ph_ids).unsqueeze(0).to(device)  # (1, P)
dur_t = torch.LongTensor(gt_durations).unsqueeze(0).to(device)  # (1, P)
f0_t = torch.FloatTensor(f0).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
nl_t = torch.LongTensor(note_lengths).unsqueeze(0).to(device)  # (1, N)

# ─── 完整合成（用 GT 音素、GT 时长、GT F0）───
print('\\n=== 合成: GT 音素 + GT 时长 + GT F0 (无预测) ===')
with torch.no_grad():
    wav1, mu, z, f0_out, lag_pred, dur_pred = model.forward_synthesis(
        ph_t, dur_t, f0_t,
        n_steps=1,
        note_lengths=nl_t,
        predict_durations=False,          # 用 GT 时长
        use_predicted_f0=False,           # 用 GT F0
    )
wav1 = wav1.squeeze().cpu().numpy()
sf.write(f'{sample_name}_gt_all.wav', wav1, sr)

# ─── 完整合成（GT 音素 + GT 时长 + 预测 F0）───
print('=== 合成: GT 音素 + GT 时长 + 预测 F0 ===')
with torch.no_grad():
    wav2, mu, z, f0_out, lag_pred, dur_pred = model.forward_synthesis(
        ph_t, dur_t, f0_t,
        n_steps=1,
        note_lengths=nl_t,
        predict_durations=False,          # 用 GT 时长
        use_predicted_f0=True,            # 用 F0Predictor
    )
wav2 = wav2.squeeze().cpu().numpy()
sf.write(f'{sample_name}_gt_ph_dur_pred_f0.wav', wav2, sr)

# ─── 完整合成（GT 音素 + 预测时长 + GT F0）───
print('=== 合成: GT 音素 + 预测时长 + GT F0 ===')
with torch.no_grad():
    wav3, mu, z, f0_out, lag_pred, dur_pred = model.forward_synthesis(
        ph_t, dur_t, f0_t,
        n_steps=1,
        note_lengths=nl_t,
        predict_durations=True,           # 预测时长
        use_predicted_f0=False,           # 用 GT F0
    )
wav3 = wav3.squeeze().cpu().numpy()
sf.write(f'{sample_name}_gt_ph_pred_dur_gt_f0.wav', wav3, sr)

# ─── 完整合成（GT 音素 + 预测时长 + 预测 F0）───
print('=== 合成: GT 音素 + 预测时长 + 预测 F0 ===')
with torch.no_grad():
    wav4, mu, z, f0_out, lag_pred, dur_pred = model.forward_synthesis(
        ph_t, dur_t, f0_t,
        n_steps=1,
        note_lengths=nl_t,
        predict_durations=True,           # 预测时长
        use_predicted_f0=True,            # 预测 F0
    )
wav4 = wav4.squeeze().cpu().numpy()
sf.write(f'{sample_name}_gt_ph_pred_dur_pred_f0.wav', wav4, sr)

# 原声
sf.write(f'{sample_name}_orig.wav', audio, sr)

# 对比
for name, w in [('原声', audio), ('GT_ALL', wav1), ('GT_PH+DUR+PredF0', wav2),
                 ('GT_PH+PredDur+GTF0', wav3), ('GT_PH+PredDur+PredF0', wav4)]:
    rms = np.sqrt(np.mean(w ** 2))
    print(f'  {name:25s}  RMS={rms:.4f}')

print(f'\\n输出文件:')
for f in sorted(os.listdir('.')):
    if f.startswith(sample_name) and f.endswith('.wav'):
        sz = os.path.getsize(f)
        print(f'  {f} ({sz//1024} KB)')
"
