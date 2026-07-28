"""用 60k 检查点测试 AE 重建和推理"""
import os, sys, json, torch, soundfile as sf, numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel

# 配置
config = {}
for name in ['base', 'model']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        import yaml
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']

device = 'cuda'
sr = 24000

# 加载模型 (60k 检查点)
model = SVSModel(config).to(device)
model.eval()
ckpt = torch.load('/root/autodl-tmp/checkpoints/model_060000.pt', map_location=device)
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)
print(f"加载检查点 step={ckpt['step']}")

# 加载数据
from yuaz_sgr.data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json',
    bs=1, device=device,
)
batch = next(train_iter())
audio = batch['audio']; phonemes = batch['phoneme_ids']; durations = batch['durations']
f0_audio = batch['f0_audio']; note_lengths = batch['note_lengths']
target = audio.shape[-1]

print(f"数据: audio={list(audio.shape)} f0={list(f0_audio.shape)} dur={list(durations.shape)}")

# ── AE 重建 ──
with torch.no_grad(), torch.cuda.amp.autocast():
    z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
    z_q, _, _ = model.rvq(z1)
    ae_wav = model.ddsp_decoder(f0_enc, z_q, target_samples=target)

ae = ae_wav.squeeze().cpu().numpy()
ae = ae / (np.abs(ae).max() + 1e-8)
orig = audio.squeeze().cpu().numpy()
orig = orig / (np.abs(orig).max() + 1e-8)
sf.write('/root/autodl-tmp/ae_60k.wav', ae, sr)
sf.write('/root/autodl-tmp/orig_60k.wav', orig, sr)
print(f"AE 重建保存: {len(ae)/sr:.1f}s")

# ── 推理 (平滑 F0 + 原始音素) ──
f0_raw = f0_audio.squeeze().cpu().numpy()
f0_sm = gaussian_filter1d(f0_raw.astype(np.float64), sigma=6.0)
nl = note_lengths.squeeze().cpu().numpy().astype(int)
f0_step = np.zeros_like(f0_sm); pos = 0
for n in nl:
    if n <= 0: continue
    end = min(pos + n, len(f0_step))
    if end > pos: f0_step[pos:end] = np.mean(f0_sm[pos:end])
    pos = end
f0_t = torch.FloatTensor(f0_step[:len(f0_raw)]).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    wav8, _, _, _, _, _ = model.forward_synthesis(
        phonemes, durations, f0_t, n_steps=8, note_lengths=note_lengths)
    wav1, _, _, _, _, _ = model.forward_synthesis(
        phonemes, durations, f0_t, n_steps=1, note_lengths=note_lengths)

w8 = wav8.squeeze().cpu().numpy(); w8 = w8 / (np.abs(w8).max() + 1e-8)
w1 = wav1.squeeze().cpu().numpy(); w1 = w1 / (np.abs(w1).max() + 1e-8)
sf.write('/root/autodl-tmp/infer8_60k.wav', w8, sr)
sf.write('/root/autodl-tmp/infer1_60k.wav', w1, sr)
print(f"推理保存: 8步={len(w8)/sr:.1f}s 1步={len(w1)/sr:.1f}s")
