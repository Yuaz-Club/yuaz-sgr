import numpy as np, glob, random
files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/train/*.npz"))
random.seed(42)
for idx in random.sample(range(len(files)), 5):
    d = np.load(files[idx])
    f0 = d['f0']
    fn = files[idx].split('/')[-1]
    has_voice = (f0>50).sum()
    if has_voice > 0:
        print(f"{fn}: voiced={has_voice} mean={f0[f0>50].mean():.0f}Hz uniq={len(set(f0[f0>50].astype(int)))}")
    else:
        print(f"{fn}: ALL ZERO")
# 也看整体统计
all_voiced = []
for f in files:
    d = np.load(f)
    f0 = d['f0']
    all_voiced.append((f0>50).mean()*100)
print(f"\n整体: 均值{(np.mean(all_voiced)):.0f}% 零值段={sum(1 for v in all_voiced if v<5)}/{len(all_voiced)}")

