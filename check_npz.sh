#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
cd /root/autodl-tmp/yuaz_sgr
python3 << 'PYEOF'
import numpy as np
d = np.load("data/preprocessed/test/2001_0002.npz", allow_pickle=True)
for k in d.keys():
    v = d[k]
    if isinstance(v, np.ndarray):
        print("{}: shape={}, dtype={}, val={}".format(k, v.shape, v.dtype, v.ravel()[:10]))
    else:
        print("{}: type={}, val={}".format(k, type(v).__name__, str(v)[:80]))
PYEOF
