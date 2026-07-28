import sys, os, torch, numpy as np
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

# Count all state_dict keys
model_keys = set(model.state_dict().keys())
print(f"Model has {len(model_keys)} keys")

ckpt = torch.load("/root/autodl-tmp/checkpoints/model_020000.pt", weights_only=False)
ckpt_keys = set(ckpt["model"].keys())
print(f"Ckpt has {len(ckpt_keys)} keys")

missing_in_ckpt = model_keys - ckpt_keys
missing_in_model = ckpt_keys - model_keys

print(f"\nMissing in ckpt (model has, ckpt lacks): {len(missing_in_ckpt)}")
if missing_in_ckpt:
    for k in sorted(missing_in_ckpt)[:10]:
        print(f"  - {k}")
    if len(missing_in_ckpt) > 10:
        print(f"  ... and {len(missing_in_ckpt)-10} more")

print(f"\nMissing in model (ckpt has, model lacks): {len(missing_in_model)}")
if missing_in_model:
    for k in sorted(missing_in_model)[:10]:
        print(f"  - {k}")
    if len(missing_in_model) > 10:
        print(f"  ... and {len(missing_in_model)-10} more")

# Key shape check
shape_mismatches = 0
for k in sorted(model_keys & ckpt_keys):
    if model.state_dict()[k].shape != ckpt["model"][k].shape:
        print(f"  SHAPE MISMATCH: {k}: model={list(model.state_dict()[k].shape)} ckpt={list(ckpt['model'][k].shape)}")
        shape_mismatches += 1
        if shape_mismatches > 10:
            print(f"  ... stopping at 10")
            break

if shape_mismatches == 0:
    print("\nAll shared keys have matching shapes - checkpoint is fully compatible!")
else:
    print(f"\n{shape_mismatches} keys have shape mismatches")

# Show total params
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal params: {total_params:,}")
