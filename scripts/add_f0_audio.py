import os, json, sys
import numpy as np
from pathlib import Path

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/yuaz_sgr/data/preprocessed"
npz_files = [str(p) for p in Path(data_dir).rglob("*.npz")]
total = len(npz_files)
print(f"共 {total} 个 npz 文件, 逐帧提取真实 F0...")

added = 0
for i, fpath in enumerate(npz_files):
    d = np.load(fpath, allow_pickle=True)
    if 'f0_audio' in d:
        if (i+1) % 500 == 0:
            print(f"[{i+1}/{total}] ...")
        continue
    audio = d['audio'].astype(np.float64)
    import pyworld as pw
    _f0, _t = pw.dio(audio, 24000, frame_period=256/24000*1000, f0_floor=50, f0_ceil=2000)
    _f0 = pw.stonemask(audio, _f0, _t, 24000)
    target = len(audio) // 256 + 1
    if len(_f0) < target:
        _f0 = np.pad(_f0, (0, target - len(_f0)))
    elif len(_f0) > target:
        _f0 = _f0[:target]
    new = {k: d[k] for k in d.files if k != 'f0_audio'}
    new['f0_audio'] = _f0.astype(np.float32)
    np.savez(fpath, **new)
    added += 1
    if (i+1) % 500 == 0:
        print(f"[{i+1}/{total}] 已添加 {added}")
print(f"done. 新增 {added} 个文件的 f0_audio")
