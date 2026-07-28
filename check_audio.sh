#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
cd /root/autodl-tmp/yuaz_sgr
python3 << 'PYEOF'
import numpy as np, pathlib
f = sorted(pathlib.Path("data/preprocessed/train").glob("*.npz"))
print(f"Total: {len(f)}")
for p in f[:5]:
    d = np.load(str(p), allow_pickle=True)
    a = d["audio"]
    nz = (np.abs(a) > 1e-6).sum()
    has_rest = (d["f0"] == 0).all()
    print(f"{p.name}: len={len(a)} nonzeros={nz} all_rest={has_rest}")
PYEOF
