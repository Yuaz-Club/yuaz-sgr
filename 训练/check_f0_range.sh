#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz
cd /root/autodl-tmp
for f in data/preprocessed/test/*.npz; do
    name=$(basename $f)
    python3 -c "
import numpy as np
d = np.load('$f')
f0 = d['f0']
audio = d['audio']
dur = len(audio) / 24000
nz = f0[f0 > 50]
if len(nz) > 0:
    print(f'$name: dur={dur:.1f}s  F0=[{nz.min():.0f}, {nz.max():.0f}] mean={nz.mean():.0f}Hz  {len(nz)}/{len(f0)} voiced')
else:
    print(f'$name: dur={dur:.1f}s  F0=all zero')
"
done
