import torch
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_005000.pt", weights_only=False)
keys = list(ckpt["model"].keys())
print("Total keys:", len(keys))
print("First 5:", keys[:5])
has_mod = any("_orig_mod" in k for k in keys)
print("Has _orig_mod:", has_mod)
if has_mod:
    print("Sample _orig_mod key:", [k for k in keys if "_orig_mod" in k][0])
