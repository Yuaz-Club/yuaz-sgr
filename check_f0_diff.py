"""检查训练 F0 的帧间变化量"""
import numpy as np, os
base = '/root/autodl-tmp/yuaz_sgr/data/preprocessed/train'
files = os.listdir(base)[:50]

all_diffs = []
for f in files:
    d = np.load(os.path.join(base, f), allow_pickle=True)
    f0 = d['f0']
    voiced = f0 > 0
    diffs = np.abs(np.diff(f0[voiced]))
    all_diffs.extend(diffs.tolist())

all_diffs = np.array(all_diffs)
print(f'F0 帧间帧间变化 (abs diff) 统计:')
print(f'  mean={all_diffs.mean():.2f} Hz')
print(f'  std={all_diffs.std():.2f} Hz')
print(f'  median={np.median(all_diffs):.2f} Hz')
print(f'  P90={np.percentile(all_diffs, 90):.2f} Hz')
print(f'  P95={np.percentile(all_diffs, 95):.2f} Hz')
print(f'  P99={np.percentile(all_diffs, 99):.2f} Hz')
print(f'  min={all_diffs.min():.2f}, max={all_diffs.max():.2f}')
