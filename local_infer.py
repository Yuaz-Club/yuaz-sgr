#!/usr/bin/env python3
"""本地推理测速 - 使用下载的模型和 npz 文件"""
import sys, os, time
sys.path.insert(0, r'D:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr')
import torch, yaml, numpy as np
import soundfile as sf

# 路径配置
BASE = r'D:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own'
CKPT = os.path.join(BASE, 'model_100k.pt')
NPZ = os.path.join(BASE, 'test_2001000001.npz')
OUT = os.path.join(BASE, 'recon_100k_local.wav')
CONFIG_DIR = os.path.join(BASE, 'yuaz_sgr', 'config')

# 加载配置
config = {}
for name in ['base', 'model', 'train']:
    with open(os.path.join(CONFIG_DIR, f'{name}.yaml')) as f:
        config.update(yaml.safe_load(f))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
print(f"{'='*40}")

from models.svs_model import SVSModel
model = SVSModel(config).to(device)
model.eval()

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
model.load_state_dict(model_sd, strict=False)
print(f"模型加载完成: {ckpt.get('step', '?')} 步")

# 加载 npz 测试数据
data = np.load(NPZ, allow_pickle=True)
audio_np = data['audio'].astype(np.float32)

audio = torch.FloatTensor(audio_np).unsqueeze(0).unsqueeze(0).to(device)

# 推理测速
warmup = 3
n_runs = 10

print(f"预热 {warmup} 次...")
for _ in range(warmup):
    with torch.no_grad():
        z1, f0 = model.encoder(audio)
        z_q, _, _ = model.rvq(z1)
        wav = model.ddsp_decoder(f0, z_q, target_samples=audio.shape[-1])

print(f"测速 {n_runs} 次...")
torch.cuda.synchronize() if device.type == 'cuda' else None
times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    with torch.no_grad():
        z1, f0 = model.encoder(audio)
        z_q, _, _ = model.rvq(z1)
        wav = model.ddsp_decoder(f0, z_q, target_samples=audio.shape[-1])
    torch.cuda.synchronize() if device.type == 'cuda' else None
    times.append(time.perf_counter() - t0)

audio_len = len(audio_np) / 24000
avg_time = sum(times) / len(times)
rtf = avg_time / audio_len

print(f"\n{'='*40}")
print(f"音频时长: {audio_len:.2f}s")
print(f"推理时间: {avg_time*1000:.1f}ms (平均, n={n_runs})")
print(f"RTF:      {rtf:.4f}x")
print(f"最快:     {min(times)*1000:.1f}ms")
print(f"最慢:     {max(times)*1000:.1f}ms")
print(f"{'='*40}")

# 保存输出
wav_np = wav.squeeze().cpu().numpy()
sf.write(OUT, wav_np, 24000)
print(f"已保存: {OUT}")
