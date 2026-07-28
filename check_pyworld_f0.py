"""检查 pyworld F0 的详细特征：颤音频率、幅度等"""
import numpy as np, os
base = '/root/autodl-tmp/yuaz_sgr/data/preprocessed/train'
files = sorted(os.listdir(base))[:50]  # 取前50个

all_vibrato_rates = []
all_vibrato_extents = []
all_per_note_f0 = []  # 每个音符内的 F0 序列
all_note_ranges = []  # 每个音符的 F0 范围

for f in files:
    d = np.load(os.path.join(base, f), allow_pickle=True)
    f0 = d['f0']
    note_lengths = d.get('note_lengths', np.array([]))
    lags = d['lags']
    
    # 找连续有声片段（歌曲段）
    voiced = f0 > 0
    if voiced.sum() < 20:
        continue
    
    # 对每个音符（根据 lag 变化检测边界）
    # lag 在音符间变化，所以 lag diff 标记音符边界
    lag_change = np.diff(np.concatenate([[lags[0]], lags]))
    note_starts = np.where(np.abs(lag_change) > 0.5)[0]
    
    for ns in note_starts[:20]:  # 最多看20个音符
        if ns + 30 >= len(f0):
            continue
        # 取音符内持续有声区域
        note_end = min(ns + 120, len(f0))  # 最多120帧
        segment = f0[ns:note_end]
        v = segment > 0
        if v.sum() < 15:
            continue
        seg_vals = segment[v]
        if len(seg_vals) < 15:
            continue
        
        # F0 范围（相对中心音高的偏差）
        median_f0 = np.median(seg_vals)
        if median_f0 < 100:
            continue
        cents_dev = 1200 * np.log2(seg_vals / median_f0)
        
        # 最大偏差（颤音幅度）
        extent = np.max(np.abs(cents_dev))
        all_vibrato_extents.append(extent)
        all_note_ranges.append(seg_vals.max() - seg_vals.min())
        all_per_note_f0.append(seg_vals - median_f0)
        
        # 帧间变化（微抖动）
        diffs = np.abs(np.diff(seg_vals))
        if len(diffs) > 0:
            all_vibrato_rates.extend(diffs.tolist())

all_extents = np.array(all_vibrato_extents)
all_ranges = np.array(all_note_ranges)
all_rates = np.array(all_vibrato_rates)
all_per_note_f0 = np.concatenate(all_per_note_f0) if all_per_note_f0 else np.array([])

print('=== pyworld F0 详细特征 ===')
print(f'总音符采样: {len(all_extents)}')
print(f'\n--- 颤音幅度 (cents) ---')
print(f'  mean={all_extents.mean():.1f}, median={np.median(all_extents):.1f}')
print(f'  P25={np.percentile(all_extents, 25):.1f}')
print(f'  P75={np.percentile(all_extents, 75):.1f}')
print(f'  P90={np.percentile(all_extents, 90):.1f}')

print(f'\n--- 音符内 F0 范围 (Hz) ---')
print(f'  mean={all_ranges.mean():.2f}, median={np.median(all_ranges):.2f}')
print(f'  P25={np.percentile(all_ranges, 25):.2f}')
print(f'  P75={np.percentile(all_ranges, 75):.2f}')
print(f'  P90={np.percentile(all_ranges, 90):.2f}')

print(f'\n--- 帧间变化 (Hz) ---')
print(f'  mean={all_rates.mean():.2f}, std={all_rates.std():.2f}')
print(f'  median={np.median(all_rates):.2f}')
print(f'  P90={np.percentile(all_rates, 90):.2f}')
print(f'  P95={np.percentile(all_rates, 95):.2f}')
print(f'  P99={np.percentile(all_rates, 99):.2f}')

print(f'\n--- 音符内相对 F0 分布 (Hz) ---')
if len(all_per_note_f0) > 0:
    print(f'  std={all_per_note_f0.std():.2f}')
    print(f'  P1={np.percentile(all_per_note_f0, 1):.2f}, P99={np.percentile(all_per_note_f0, 99):.2f}')
