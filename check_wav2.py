import librosa, numpy as np
from pathlib import Path
p = "/root/autodl-tmp/yuaz_sgr/opencpop"
wav = Path(p) / "wavs" / "2007.wav"
if wav.exists():
    print("wav size:", wav.stat().st_size)
    a, sr = librosa.load(str(wav), sr=24000, mono=True)
    print("librosa:", a.shape, a.dtype, a.min(), a.max(), a.mean(), "nonzero:", np.count_nonzero(a))
else:
    print("NOT FOUND")
    for w in sorted(Path(p).glob("**/*.wav"))[:5]:
        print("  found:", w.relative_to(p))
