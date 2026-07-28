import sys, os, torch, numpy as np, soundfile as sf, glob, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, scipy.signal

config = {}
for n in ["base", "model"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    pm = json.load(f)
config["conditioning"]["num_phonemes"] = pm["num_phonemes"]

from models.svs_model import SVSModel

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))

for ckpt_name, ckpt_path in [
    ("00500", "/root/autodl-tmp/checkpoints/model_005000.pt"),
    ("10000", "/root/autodl-tmp/checkpoints/model_010000.pt"),
]:
    print(f"\n=== CKPT {ckpt_name} ===")
    m = SVSModel(config).cuda().eval()
    ckpt = torch.load(ckpt_path, weights_only=False)
    m.load_state_dict(ckpt["model"], strict=False)
    
    for npz in files[:2]:
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
        f_o, P_o = scipy.signal.welch(wav, 24000, nperseg=2048)
        f0v = f0[f0 > 50].mean().item() if (f0 > 50).any() else 0
        hp = sum(P_o[np.abs(f_o - f0v*m) <= 20].sum() for m in [1,2,3,4,5] if f0v*m < 12000)
        print(f"  {name}: rms={rms_out:.4f}(x{rms_out/rms_in:.1f}) max={np.abs(wav).max():.2f} harm={hp/P_o.sum():.3f}")
        
        os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)
        sf.write(f"/root/autodl-tmp/inference_output/ckpt{ckpt_name}_{name}.wav", wav, 24000)
        sf.write(f"/root/autodl-tmp/inference_output/ckpt{ckpt_name}_{name}_orig.wav", a_orig[:96000], 24000)
print("\nDone!")
