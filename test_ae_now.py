import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, glob, soundfile as sf

config = {}
for n in ["base", "model"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    pm = json.load(f)
config["conditioning"]["num_phonemes"] = pm["num_phonemes"]

from models.svs_model import SVSModel
model = SVSModel(config).cuda().eval()

import glob
files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)

for npz in files[:2]:
    data = np.load(npz, allow_pickle=True)
    a_orig = data["audio"].squeeze()
    a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()
    name = os.path.basename(npz).replace(".npz", "")
    rms_in = np.sqrt(np.mean(a_orig[:96000]**2))
    print(f"{name}: in_rms={rms_in:.5f}")

    with torch.no_grad():
        z1, f0 = model.encoder(a)
        z_q, _, _ = model.rvq(z1)
        fake = model.ddsp_decoder(f0, z_q, target_samples=a.shape[-1])
    wav = fake.squeeze().cpu().numpy()
    rms_out = np.sqrt(np.mean(wav**2))
    print(f"  out_rms={rms_out:.5f}, ratio={rms_out/rms_in:.1f}x, max={np.abs(wav).max():.3f}")
    sf.write(f"/root/autodl-tmp/inference_output/{name}_ae.wav", wav, 24000)
    sf.write(f"/root/autodl-tmp/inference_output/{name}_orig.wav", a_orig[:96000], 24000)
print("Done!")
