#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
cd /root/autodl-tmp/yuaz_sgr
python3 << 'PYEOF'
import numpy as np, pathlib, json
ok = err = 0
with open("data/preprocessed/phoneme_map.json") as f:
    pm = json.load(f)
print(f"音素数: {pm['num_phonemes']}")
print(f"音素表: {sorted(pm['p2i'].keys())}")
for f in sorted(pathlib.Path("data/preprocessed/train").glob("*.npz")):
    d = np.load(str(f), allow_pickle=True)
    s = d["durations"].sum()
    f0 = len(d["f0"])
    if abs(s - f0) > 5:
        err += 1
        print(f"MISMATCH: {f.name} dur={s} f0={f0}")
    else:
        ok += 1
print(f"OK={ok}, ERR={err}, Total={ok+err}")
# check file sizes
sizes = [f.stat().st_size for f in pathlib.Path("data/preprocessed/train").glob("*.npz")]
print(f"Avg npz size: {sum(sizes)/len(sizes)/1024:.0f} KB, Total: {sum(sizes)/1024/1024:.0f} MB")
PYEOF
