#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

cd /root/autodl-tmp

CKPT=${1:-finetune_checkpoints/model_007000.pt}
N_SAMPLES=${2:-3}
OUTDIR="ae_reconstructions_$(basename $CKPT .pt)"

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
output_dir = '$OUTDIR'
num_samples = $N_SAMPLES
device = 'cuda'

os.makedirs(output_dir, exist_ok=True)

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
        print(f'  部分加载 {clean_k}: {v.shape} → {model_sd[clean_k].shape}')
    else:
        skipped += 1
model.load_state_dict(model_sd, strict=False)
step = ckpt.get('step', 0)
print(f'Checkpoint step={step}: {loaded} 加载, {skipped} 跳过')

# 测试文件
val_files = sorted(Path(data_dir).glob('*.npz'))
print(f'测试集 {len(val_files)} 段, 取前 {num_samples} 段')

sr = config.get('sample_rate', 24000)
target_samples = 4 * sr

for idx, fpath in enumerate(val_files[:num_samples]):
    data = np.load(fpath, allow_pickle=True)
    audio = data['audio'].astype(np.float32)
    f0 = data['f0'].astype(np.float32)
    phonemes = data['phonemes']

    # 对齐到 4 秒
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    audio = audio[:target_samples]

    audio_t = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
    f0_t = torch.FloatTensor(f0).unsqueeze(0).to(device)
    if f0_t.shape[-1] < target_samples // 256 + 1:
        pad_len = target_samples // 256 + 1 - f0_t.shape[-1]
        f0_t = torch.nn.functional.pad(f0_t, (0, pad_len))
    f0_t = f0_t[:, :target_samples // 256 + 1]

    with torch.no_grad():
        z1, f0_out, z_q, fake_wav, rvq_loss = model.forward_ae(audio_t, f0_override=f0_t)

    fake_wav = fake_wav.squeeze().cpu().numpy()

    base_name = fpath.stem
    sf.write(os.path.join(output_dir, f'{base_name}_orig.wav'), audio, sr)
    sf.write(os.path.join(output_dir, f'{base_name}_recon.wav'), fake_wav, sr)

    orig_rms = np.sqrt(np.mean(audio ** 2))
    recon_rms = np.sqrt(np.mean(fake_wav ** 2))
    print(f'  [{idx+1}/{num_samples}] {base_name}: rvq={rvq_loss:.4f}  orig_rms={orig_rms:.4f}  recon_rms={recon_rms:.4f}')

print(f'\n完成! 输出目录: {output_dir}')
for f in sorted(os.listdir(output_dir)):
    sz = os.path.getsize(os.path.join(output_dir, f))
    print(f'  {f} ({sz//1024} KB)')
"
