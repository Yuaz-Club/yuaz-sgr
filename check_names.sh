#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
cd /root/autodl-tmp/yuaz_sgr
python3 << 'PYEOF'
import pathlib
# Check train.txt names
with open("data/opencpop/segments/train.txt") as f:
    train_names = set(l.strip() for l in f if l.strip())
with open("data/opencpop/segments/test.txt") as f:
    test_names = set(l.strip() for l in f if l.strip())
print(f"Train: {len(train_names)}, Test: {len(test_names)}")
print(f"Train sample: {sorted(train_names)[:5]}")
print(f"Test sample: {sorted(test_names)[:5]}")

# Check WAV files
wavs = list(pathlib.Path("data/opencpop/segments/wavs").glob("*.wav"))
print(f"\nWAVs: {len(wavs)}")
print(f"WAV sample: {[w.stem for w in wavs[:5]]}")
print(f"WAV sample names from train: {[w.stem for w in wavs if w.stem in train_names][:5]}")
print(f"WAV sample names from test: {[w.stem for w in wavs if w.stem in test_names][:5]}")
print(f"ALL WAVs match train: {sum(1 for w in wavs if w.stem in train_names)}")
print(f"ALL WAVs match test: {sum(1 for w in wavs if w.stem in test_names)}")

# Check first line of transcriptions
with open("data/opencpop/segments/transcriptions.txt") as f:
    first_line = f.readline().strip()
print(f"\nFirst transcription line prefix: {first_line[:80]}")
PYEOF
