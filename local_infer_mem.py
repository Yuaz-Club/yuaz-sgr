#!/usr/bin/env python3
"""本地推理 + 内存占用测速"""
import sys, os, time
sys.path.insert(0, r'D:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr')
import torch, yaml, numpy as np
import soundfile as sf
import tracemalloc
import gc

BASE = r'D:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own'
CKPT = os.path.join(BASE, 'model_40k.pt')
NPZ = os.path.join(BASE, 'test_2001000001.npz')
DESKTOP = os.path.join(os.path.expanduser('~'), 'Desktop')
OUT = os.path.join(DESKTOP, 'recon_local.wav')
CONFIG_DIR = os.path.join(BASE, 'yuaz_sgr', 'config')

config = {}
for name in ['base', 'model', 'train']:
    with open(os.path.join(CONFIG_DIR, f'{name}.yaml')) as f:
        config.update(yaml.safe_load(f))

device = torch.device('cpu')
print(f"设备: {device}")
print(f"{'='*40}")

from models.svs_model import SVSModel
model = SVSModel(config)
model.eval()

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
model.load_state_dict(model_sd, strict=False)
print(f"模型加载完成: {ckpt.get('step', '?')} 步")

# 模型参数内存
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"模型参数内存: {param_bytes/1024/1024:.1f} MB")

# 加载 npz
data = np.load(NPZ, allow_pickle=True)
audio_np = data['audio'].astype(np.float32)
audio = torch.FloatTensor(audio_np).unsqueeze(0).unsqueeze(0)

# ---- 单项推理内存测试 ----
gc.collect()
tracemalloc.start()

for _ in range(3):  # 预热
    with torch.no_grad():
        z1, f0 = model.encoder(audio)
        z_q, _, _ = model.rvq(z1)
        wav = model.ddsp_decoder(f0, z_q, target_samples=audio.shape[-1])

# 单独测一次峰值
gc.collect()
_, peak_before = tracemalloc.get_traced_memory()

with torch.no_grad():
    z1, f0 = model.encoder(audio)
    z_q, _, _ = model.rvq(z1)
    wav = model.ddsp_decoder(f0, z_q, target_samples=audio.shape[-1])

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

wav_np = wav.squeeze().numpy()
sf.write(OUT, wav_np, 24000)

# ---- 多 batch 测试 ----
print(f"\n{'='*35}")
print(f"推理内存报告")
print(f"{'='*35}")
print(f"推理前基线:   {peak_before/1024/1024:.1f} MB")
print(f"推理后:       {current/1024/1024:.1f} MB")
print(f"推理峰值:     {peak/1024/1024:.1f} MB")
print(f"推理增量:     {(peak - peak_before)/1024/1024:.1f} MB")

# 测速
print(f"\n{'='*35}")
print(f"推理速度 (CPU)")
print(f"{'='*35}")
n_runs = 10
for _ in range(3):
    with torch.no_grad():
        z1, f0 = model.encoder(audio)
        z_q, _, _ = model.rvq(z1)
        _ = model.ddsp_decoder(f0, z_q, target_samples=audio.shape[-1])

times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    with torch.no_grad():
        z1, f0 = model.encoder(audio)
        z_q, _, _ = model.rvq(z1)
        wav = model.ddsp_decoder(f0, z_q, target_samples=audio.shape[-1])
    times.append(time.perf_counter() - t0)

audio_len = len(audio_np) / 24000
avg_time = sum(times) / len(times)

print(f"音频时长:      {audio_len:.2f}s")
print(f"平均推理:      {avg_time*1000:.1f}ms")
print(f"RTF:           {avg_time/audio_len:.4f}x")

# ---- 多 batch 推理 ----
print(f"\n{'='*35}")
print(f"批量推理 (CPU)")
print(f"{'='*35}")
for bs in [1, 2, 4]:
    gc.collect()
    batch_audio = audio.repeat(bs, 1, 1)
    
    tracemalloc.start()
    _, peak_b4 = tracemalloc.get_traced_memory()
    
    t0 = time.perf_counter()
    with torch.no_grad():
        z1, f0 = model.encoder(batch_audio)
        z_q, _, _ = model.rvq(z1)
        wav = model.ddsp_decoder(f0, z_q, target_samples=batch_audio.shape[-1])
    t = time.perf_counter() - t0
    
    _, peak_hit = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    
    print(f"  batch_size={bs}:  {t*1000:.0f}ms, RTF={t/(audio_len*bs):.4f}, 峰值内存={peak_hit/1024/1024:.1f}MB")

print(f"\n输出: {OUT}")
print(f"总模型+数据参考: ~{(param_bytes + peak)/1024/1024:.0f} MB")
