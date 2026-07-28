import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, soundfile as sf, glob

config = {}
for n in ["base", "model"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    pm = json.load(f)
config["conditioning"]["num_phonemes"] = pm["num_phonemes"]

from models.svs_model import SVSModel
m = SVSModel(config).cuda().eval()
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_020000.pt", weights_only=False)
m.load_state_dict(ckpt["model"], strict=False)
print(f"ckpt step={ckpt.get('step','?')}")

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)

import scipy.signal
for npz in files[:3]:
    data = np.load(npz, allow_pickle=True)
    a_orig = data["audio"].squeeze()
    a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()
    name = os.path.basename(npz).replace(".npz", "")
    with torch.no_grad():
        z1, f0 = m.encoder(a)
        z_q, _, _ = m.rvq(z1)
        fake = m.ddsp_decoder(f0, z_q, target_samples=a.shape[-1])
    wav = fake.squeeze().cpu().numpy()
    rms_in = np.sqrt(np.mean(a_orig[:96000]**2))
    rms_out = np.sqrt(np.mean(wav**2))
    f_i, P_i = scipy.signal.welch(a_orig[:96000], 24000, nperseg=2048)
    f_o, P_o = scipy.signal.welch(wav, 24000, nperseg=2048)
    f0v = f0[f0 > 50].mean().item() if (f0 > 50).any() else 0
    hp = sum(P_o[np.abs(f_o - f0v*m) <= 20].sum() for m in [1,2,3,4,5] if f0v*m < 12000)
    tp = P_o.sum()
    print(f"{name}: in_rms={rms_in:.4f}  out_rms={rms_out:.4f}  ratio={rms_out/rms_in:.1f}x  max={np.abs(wav).max():.2f}")
    print(f"  F0={f0v:.0f}Hz  harm/total={hp/tp:.4f}")
    sf.write(f"/root/autodl-tmp/inference_output/{name}_ae.wav", wav, 24000)
    sf.write(f"/root/autodl-tmp/inference_output/{name}_orig.wav", a_orig[:96000], 24000)
print("Done!")
