import soundfile as sf, numpy as np, librosa
from pathlib import Path
wav = sorted(Path("/root/autodl-tmp/yuaz_sgr/opencpop/segments/wavs").glob("*.wav"))[0]
print("wav:", wav)
d, sr = sf.read(str(wav))
print("sf:", d.shape, d.dtype, d.min(), d.max())
d2, sr2 = librosa.load(str(wav), sr=24000, mono=True)
print("librosa:", d2.shape, d2.dtype, d2.min(), d2.max(), d2.mean())
