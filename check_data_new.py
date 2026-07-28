import numpy as np
from pathlib import Path
d = np.load(str(sorted(Path("/root/autodl-tmp/yuaz_sgr/data/preprocessed/train").glob("*.npz"))[0]),
            allow_pickle=True)
print("audio:", d["audio"].shape, d["audio"].dtype, d["audio"].min(), d["audio"].max(), d["audio"].mean())
print("f0:", d["f0"].shape, d["f0"].min(), d["f0"].max(), d["f0"].mean())
print("phonemes:", d["phonemes"][:5], "durations:", d["durations"][:5])
print("note_lengths:", d["note_lengths"][:5])
print("lags:", d["lags"][:5], d["lags"].min(), d["lags"].max())
