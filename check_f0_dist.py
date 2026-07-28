"""检查训练数据 F0 分布 vs 推理 MIDI F0 分布"""
import numpy as np, os

base = '/root/autodl-tmp/yuaz_sgr/data/preprocessed/train'
files = os.listdir(base)[:200]

f0_vals = []
dur_pairs = []
lag_vals = []
ph_counts = {}

for f in files:
    d = np.load(os.path.join(base, f), allow_pickle=True)
    f0 = d['f0']
    f0_vals.extend(f0[f0 > 0].tolist())
    
    dur = d['durations']
    # 检查 durations 是否成对相等
    dur_pairs.append(list(dur[:min(10, len(dur))]))
    
    lag = d['lags']
    lag_vals.extend(lag[lag > 0].tolist())

f0_vals = np.array(f0_vals)
print(f'=== 训练 F0 分布 ({len(files)} 文件) ===')
print(f'  mean={f0_vals.mean():.1f} Hz, std={f0_vals.std():.1f} Hz')
print(f'  min={f0_vals.min():.1f}, max={f0_vals.max():.1f}')
print(f'  P5={np.percentile(f0_vals, 5):.1f}, P25={np.percentile(f0_vals, 25):.1f}')
print(f'  P50={np.percentile(f0_vals, 50):.1f}, P75={np.percentile(f0_vals, 75):.1f}')
print(f'  P95={np.percentile(f0_vals, 95):.1f}')

# 看 durations 的成对差异
print(f'\n=== durations 检查 ===')
unequal = 0
total_pairs = 0
for dur in dur_pairs:
    for i in range(0, len(dur)-1, 2):
        total_pairs += 1
        if abs(dur[i] - dur[i+1]) > 1:
            unequal += 1
print(f'  成对不相等比例: {unequal}/{total_pairs} ({100*unequal/total_pairs:.1f}%)')
print(f'  示例 durations: {dur_pairs[0]}')
print(f'  示例 durations: {dur_pairs[1]}')

# lag
lag_vals = np.array(lag_vals)
print(f'\n=== lag 分布 ===')
print(f'  非零 lag 的数量: {len(lag_vals)} ({100*len(lag_vals)/(200*375):.1f}%)')
if len(lag_vals) > 0:
    print(f'  非零 lag 范围: {lag_vals.min():.1f} ~ {lag_vals.max():.1f}')

# 推理 MIDI F0
midi_f0 = np.array([440 * 2**((n-69)/12) for n in range(40, 85)])
print(f'\n=== MIDI F0 分布 ===')
print(f'  range: {midi_f0.min():.1f} ~ {midi_f0.max():.1f} Hz')
