import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, soundfile as sf

config = {}
for n in ["base", "model"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    pm = json.load(f)
config["conditioning"]["num_phonemes"] = pm["num_phonemes"]

from models.svs_model import SVSModel
model = SVSModel(config).cuda().eval()

# Use FIXED loading logic
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_020000.pt", weights_only=False)
model_sd = model.state_dict()
loaded, skipped = 0, 0
for k, v in ckpt["model"].items():
    clean_k = k.replace("_orig_mod.", "")
    if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
        model_sd[clean_k] = v
        loaded += 1
    else:
        skipped += 1
model.load_state_dict(model_sd, strict=False)
print(f"FIXED loading: loaded={loaded}, skipped={skipped}")

assert loaded > 400, "Still too few keys loaded!"

import glob
files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
data = np.load(files[0], allow_pickle=True)
a_orig = data["audio"].squeeze()
a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()
rms_in = np.sqrt(np.mean(a_orig[:96000]**2))
print(f"Input RMS={rms_in:.5f}")

with torch.no_grad():
    z1, f0 = model.encoder(a)
    z_q, _, _ = model.rvq(z1)
    fake = model.ddsp_decoder(f0, z_q, target_samples=a.shape[-1])
wav = fake.squeeze().cpu().numpy()
rms_out = np.sqrt(np.mean(wav**2))
print(f"Output RMS={rms_out:.5f}, ratio={rms_out/rms_in:.1f}x, max={np.abs(wav).max():.3f}")
os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)
sf.write("/root/autodl-tmp/inference_output/fixed_load.wav", wav, 24000)
sf.write("/root/autodl-tmp/inference_output/fixed_load_orig.wav", a_orig[:96000], 24000)
print("OK!")
