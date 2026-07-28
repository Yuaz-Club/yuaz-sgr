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
m = SVSModel(config).cuda().eval()
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_020000.pt", weights_only=False)
sd = {k: v for k, v in ckpt["model"].items() if "mpd." not in k and "msd." not in k}
m.load_state_dict(sd, strict=False)
print("Loaded ckpt 20000, keys:", len(sd))

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
for npz in files[:2]:
    data = np.load(npz, allow_pickle=True)
    a_orig = data["audio"].squeeze()
    a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()
    name = os.path.basename(npz).replace(".npz", "")
    rms_in = np.sqrt(np.mean(a_orig[:96000]**2))
    print(f"\n{name}: audio RMS={rms_in:.5f}")

    with torch.no_grad():
        z1, f0 = m.encoder(a)
        z_q, _, _ = m.rvq(z1)
        fake = m.ddsp_decoder(f0, z_q, target_samples=a.shape[-1])
    wav = fake.squeeze().cpu().numpy()
    rms_out = np.sqrt(np.mean(wav**2))
    print(f"Output RMS={rms_out:.5f}, ratio={rms_out/rms_in:.2f}, max={np.abs(wav).max():.3f}")
    sf.write(f"/root/autodl-tmp/inference_output/{name}_c2k.wav", wav, 24000)
    sf.write(f"/root/autodl-tmp/inference_output/{name}_c2k_orig.wav", a_orig[:96000], 24000)
print("\nDone!")
