import sys, os, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json
config = {}
for n in ["base", "model"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    pm = json.load(f)
config["conditioning"]["num_phonemes"] = pm["num_phonemes"]
from models.svs_model import SVSModel
model = SVSModel(config).cuda().eval()
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_005000.pt", weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
dd = model.ddsp_decoder

mel_basis = dd.mel_basis.cpu().numpy()
mel_pinv = dd.mel_pinv.cpu().numpy()
print(f"mel_basis: {mel_basis.shape}")
print(f"mel_pinv: {mel_pinv.shape}")
nonzero_per_bin = (np.abs(mel_pinv) > 1e-7).sum(axis=1)
print(f"\nNonzero mel contributions per linear bin (out of 128):")
for bin_idx in [0, 1, 5, 10, 14, 20, 30, 50, 100, 200, 300, 400, 500, 512]:
    freq = bin_idx * 24000 / 1024
    print(f"  bin {bin_idx:3d} ({freq:6.1f} Hz): {nonzero_per_bin[bin_idx]:3d} nonzeros from mel_pinv")
print(f"\nMean nonzero per bin: {nonzero_per_bin.mean():.1f}")
print(f"Bins with <3 nonzeros: {(nonzero_per_bin < 3).sum()} / 513")

rec = mel_pinv @ mel_basis
diag = np.diag(rec)
non_diag = np.abs(rec - np.diag(np.diag(rec))).mean()
print(f"\nM+ @ M diag: mean={diag.mean():.4f}, std={diag.std():.4f}")
print(f"M+ @ M off-diag mean={non_diag:.6f}")
