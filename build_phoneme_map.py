import json, numpy as np
from pathlib import Path

data_dir = Path("/root/autodl-tmp/yuaz_sgr/data/preprocessed")
phonemes = set()
for split in ["train", "val", "test"]:
    for f in sorted((data_dir / split).glob("*.npz")):
        d = np.load(str(f), allow_pickle=True)
        for ph in d["phonemes"]:
            if isinstance(ph, np.ndarray):
                for p in ph:
                    phonemes.add(p)
            else:
                phonemes.add(str(ph))

phonemes = sorted(phonemes)
p2i = {p: i for i, p in enumerate(phonemes)}
output = {"p2i": p2i, "i2p": {str(i): p for p, i in p2i.items()}, "num_phonemes": len(phonemes)}
json.dump(output, open(str(data_dir / "phoneme_map.json"), "w"), ensure_ascii=False, indent=2)
print(f"Built phoneme_map.json: {len(phonemes)} phonemes")
