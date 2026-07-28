"""验证新预处理的 CREPE F0"""
import numpy as np, glob
files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/train/*.npz"))
for f in files[:3]:
    d = np.load(f)
    f0 = d['f0']
    fn = f.split('/')[-1]
    print(f"{fn}: f0={f0.shape} mean={f0[f0>50].mean():.0f}Hz uniq={len(set(f0[f0>50].astype(int)))} zeros={(f0==0).mean()*100:.0f}%")
print(f"\n总计训练: {len(files)} 段, 验证: {len(glob.glob('/root/autodl-tmp/yuaz_sgr/data/preprocessed/val/*.npz'))} 段, 测试: {len(glob.glob('/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz'))} 段")
