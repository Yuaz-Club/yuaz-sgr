"""本地推理测试: 内存占用 + 速度"""
import sys, json, torch, numpy as np, soundfile as sf, time, psutil
import torch.nn.functional as F
sys.path.insert(0, 'D:/360MoveData/Users/洛洛/Desktop/yuaz_sgr own/yuaz_sgr')
from models.svs_model import SVSModel
import yaml

config = {}
for name in ['base', 'model']:
    with open(f'D:/360MoveData/Users/洛洛/Desktop/yuaz_sgr own/yuaz_sgr/config/{name}.yaml') as f:
        config.update(yaml.safe_load(f))
with open('D:/360MoveData/Users/洛洛/Desktop/yuaz_sgr own/yuaz_sgr/data/preprocessed/phoneme_map.json', encoding='utf-8-sig') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']
sr = config.get('sample_rate', 24000)
hop = config.get('hop_length', 256)

# 检查是否有 CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# 记录初始内存
process = psutil.Process()
mem_before = process.memory_info().rss / 1e6
print(f"Memory before model load: {mem_before:.0f} MB")

t_load = time.time()
model = SVSModel(config).to(device)
model.eval()
ckpt = torch.load('D:/360MoveData/Users/洛洛/Desktop/yuaz_sgr own/model_091000.pt', map_location=device, weights_only=False)
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)
t_load = time.time() - t_load

mem_after_model = process.memory_info().rss / 1e6
print(f"Memory after model load: {mem_after_model:.0f} MB (+{mem_after_model-mem_before:.0f} MB)")
print(f"Model load time: {t_load:.2f}s")

if device == 'cuda':
    print(f"VRAM after model: {torch.cuda.memory_allocated()/1e6:.0f} MB")

# 构造测试输入 (从虚拟 VSQX 取前 2 秒 ≈ 188 帧)
# 用思年华的片段来测
# 简短输入: 约 2 秒
total_frames = 188  # ~2s
batch_size = 1
phoneme_t = torch.randint(0, 60, (batch_size, 20), device=device)
dur_t = torch.randint(10, 40, (batch_size, 20), device=device)
# 构造 F0: 交替音高
f0_np = np.zeros(total_frames, dtype=np.float32)
f0_vals = [415.3, 330.0, 262.0, 523.3, 440.0, 349.2, 294.0, 587.3]
seg = total_frames // len(f0_vals)
for i, fv in enumerate(f0_vals):
    f0_np[i*seg:(i+1)*seg] = fv
f0_t = torch.FloatTensor(f0_np).unsqueeze(0).unsqueeze(0).to(device)
nl_t = torch.randint(30, 100, (batch_size, 10), device=device)

# Warmup
print("\nWarmup...")
with torch.no_grad():
    for _ in range(3):
        _ = model.forward_synthesis(phoneme_t[:, :10], dur_t[:, :10], f0_t[:, :, :94], n_steps=1, note_lengths=nl_t[:, :5], predict_durations=False)
        _ = model.forward_synthesis(phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t, predict_durations=False)

# 测试 1: 短片段 (2 秒)
print("\n=== Test 1: ~2 second input ===")
torch.cuda.empty_cache() if device == 'cuda' else None
mem_before_test = process.memory_info().rss / 1e6
vram_before = torch.cuda.memory_allocated()/1e6 if device == 'cuda' else 0

t_start = time.time()
n_runs = 5
for _ in range(n_runs):
    with torch.no_grad():
        w, _, _, _, _, _ = model.forward_synthesis(phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t, predict_durations=False)
t_avg = (time.time() - t_start) / n_runs

mem_peak = process.memory_info().rss / 1e6
vram_peak = torch.cuda.memory_allocated()/1e6 if device == 'cuda' else 0
output_len = w.shape[-1] / sr
print(f"  Output: {output_len:.1f}s")
print(f"  Average time: {t_avg*1000:.1f}ms")
print(f"  RTF: {t_avg/output_len:.4f}x")
print(f"  RAM peak: {mem_peak:.0f} MB (+{mem_peak-mem_before_test:.0f} MB)")
if device == 'cuda':
    print(f"  VRAM peak: {vram_peak:.0f} MB")

# 测试 2: 完整歌曲 (~20 秒)
print("\n=== Test 2: ~20 second input ===")
total_frames_20s = 1880
f0_np2 = np.zeros(total_frames_20s, dtype=np.float32)
for i in range(len(f0_vals)):
    seg_f = total_frames_20s // len(f0_vals)
    f0_np2[i*seg_f:(i+1)*seg_f] = f0_vals[i]
f0_t2 = torch.FloatTensor(f0_np2).unsqueeze(0).unsqueeze(0).to(device)
phoneme_t2 = torch.randint(0, 60, (batch_size, 200), device=device)
dur_t2 = torch.randint(10, 40, (batch_size, 200), device=device)
nl_t2 = torch.randint(30, 100, (batch_size, 80), device=device)

torch.cuda.empty_cache() if device == 'cuda' else None
mem_before_test2 = process.memory_info().rss / 1e6
vram_before2 = torch.cuda.memory_allocated()/1e6 if device == 'cuda' else 0

t_start = time.time()
with torch.no_grad():
    w2, _, _, _, _, _ = model.forward_synthesis(phoneme_t2, dur_t2, f0_t2, n_steps=1, note_lengths=nl_t2, predict_durations=False)
t_20s = time.time() - t_start

mem_peak2 = process.memory_info().rss / 1e6
vram_peak2 = torch.cuda.memory_allocated()/1e6 if device == 'cuda' else 0
output_len2 = w2.shape[-1] / sr
print(f"  Output: {output_len2:.1f}s")
print(f"  Time: {t_20s:.2f}s")
print(f"  RTF: {t_20s/output_len2:.4f}x")
print(f"  RAM peak: {mem_peak2:.0f} MB (+{mem_peak2-mem_before_test2:.0f} MB)")
if device == 'cuda':
    print(f"  VRAM peak: {vram_peak2:.0f} MB")

# 测试 3: 预测时长模式
print("\n=== Test 3: ~20s with predict_durations=True ===")
torch.cuda.empty_cache() if device == 'cuda' else None
t_start = time.time()
with torch.no_grad():
    w3, _, _, _, _, _ = model.forward_synthesis(phoneme_t2, dur_t2, f0_t2, n_steps=1, note_lengths=nl_t2, predict_durations=True)
t_pred = time.time() - t_start
print(f"  Output: {w3.shape[-1]/sr:.1f}s")
print(f"  Time: {t_pred:.2f}s")
print(f"  RTF: {t_pred/(w3.shape[-1]/sr):.4f}x")

# 测试 4: 165 秒完整虚拟 (看大序列)
print("\n=== Test 4: ~165s (full 虚拟) ===")
total_frames_165s = 15474
f0_np3 = np.zeros(total_frames_165s, dtype=np.float32)
f0_np3[:5000] = 440.0
f0_np3[5000:10000] = 349.2
f0_np3[10000:15000] = 523.3
f0_np3[15000:] = 294.0
f0_t3 = torch.FloatTensor(f0_np3).unsqueeze(0).unsqueeze(0).to(device)
phoneme_t3 = torch.randint(0, 60, (batch_size, 665), device=device)
dur_t3 = torch.randint(8, 40, (batch_size, 665), device=device)
nl_t3 = torch.randint(30, 120, (batch_size, 363), device=device)

torch.cuda.empty_cache() if device == 'cuda' else None
mem_before3 = process.memory_info().rss / 1e6
vram_before3 = torch.cuda.memory_allocated()/1e6 if device == 'cuda' else 0

t_start = time.time()
with torch.no_grad():
    w4, _, _, _, _, _ = model.forward_synthesis(phoneme_t3, dur_t3, f0_t3, n_steps=1, note_lengths=nl_t3, predict_durations=False)
t_long = time.time() - t_start

mem_peak3 = process.memory_info().rss / 1e6
vram_peak3 = torch.cuda.memory_allocated()/1e6 if device == 'cuda' else 0
output_len4 = w4.shape[-1] / sr
print(f"  Output: {output_len4:.1f}s")
print(f"  Time: {t_long:.2f}s")
print(f"  RTF: {t_long/output_len4:.4f}x")
print(f"  RAM peak: {mem_peak3:.0f} MB (+{mem_peak3-mem_before3:.0f} MB)")
if device == 'cuda':
    print(f"  VRAM peak: {vram_peak3:.0f} MB")

print("\n=== Summary ===")
print(f"  Model size: 18.3M params")
print(f"  Load time: {t_load:.2f}s")
print(f"  RAM after load: {mem_after_model:.0f} MB")
print(f"  2s 推理: {t_avg*1000:.0f}ms ({output_len:.1f}s output)")
print(f"  20s 推理: {t_20s:.2f}s ({output_len2:.1f}s output, RTF={t_20s/output_len2:.4f})")
print(f"  165s 推理: {t_long:.2f}s ({output_len4:.1f}s output, RTF={t_long/output_len4:.4f})")
