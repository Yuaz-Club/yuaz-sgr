import numpy as np
from pathlib import Path

npz = sorted(Path("/root/autodl-tmp/yuaz_sgr/data/preprocessed/train").glob("*.npz"))[0]
d = np.load(str(npz), allow_pickle=True)
print("npz:", npz.name)
print("audio:", d["audio"].shape, d["audio"].dtype)
print("audio stats:", d["audio"].min(), d["audio"].max(), d["audio"].mean(), d["audio"].std())
print("nonzeros:", np.count_nonzero(d["audio"]))
print("f0:", d["f0"].shape, d["f0"].min(), d["f0"].max())
print("durations:", d["durations"])
print("note_lengths:", d["note_lengths"])
