import numpy as np
d = np.load("/root/autodl-tmp/yuaz_sgr/data/preprocessed/train/2007_0000.npz", allow_pickle=True)
print("keys:", list(d.keys()))
for k in d.keys():
    v = d[k]
    if isinstance(v, np.ndarray):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
