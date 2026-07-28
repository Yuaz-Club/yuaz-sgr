import sys, os, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, soundfile as sf, glob

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

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
data = np.load(files[0], allow_pickle=True)
a_orig = data["audio"].squeeze()
a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()

with torch.no_grad():
    z1, f0 = model.encoder(a)
    if f0.dim() == 2: f0 = f0.unsqueeze(1)
    z_q, _, _ = model.rvq(z1)
    B, _, T = z_q.shape
    feat = torch.cat([z_q, f0], dim=1).transpose(1, 2)
    feat = dd.emformer_input_proj(feat)
    emf = dd.emformer(feat); emf = emf[0] if isinstance(emf, tuple) else emf
    S_c = dd.env_net(emf)
    S_lin = dd._decompress_envelope(S_c.transpose(1, 2))
    A_c = dd.ap_net(emf)
    A_lin_raw = A_c.transpose(1, 2)
    A_lin = F.interpolate(A_lin_raw, size=dd.fft_size // 2 + 1, mode='linear').transpose(1, 2)
    w = dd.weight_net(emf)
    lh = torch.clamp(1.0 / (w[:,:,0:1] + 1e-8), min=0)
    ln = torch.clamp(1.0 / (w[:,:,1:2] + 1e-8), min=0)

    fake = dd(f0, z_q, target_samples=a.shape[-1])
    wav = fake.squeeze().cpu().numpy()
    
    import scipy.signal
    f_i, P_i = scipy.signal.welch(a_orig[:96000], 24000, nperseg=2048)
    f_o, P_o = scipy.signal.welch(wav, 24000, nperseg=2048)
    f0_mean = f0[f0 > 50].mean().item() if (f0 > 50).any() else 0
    f0_bin = int(f0_mean * 1024 / 24000)
    
    print(f"=== AE Step 5000 深度诊断 ===")
    print(f"F0: {f0_mean:.0f} Hz (~bin {f0_bin})")
    print(f"S_lin at F0 bin {f0_bin}: mean={S_lin[0, f0_bin, :].mean().item():.5f}")
    print(f"S_lin overall: mean={S_lin.mean().item():.3f}, zeros={((S_lin<1e-6).float().mean()*100).item():.0f}%")
    print(f"lambda_h={lh.mean().item():.2f}, lambda_n={ln.mean().item():.2f}")
    print(f"Input power: {P_i.max():.2e} at {f_i[P_i.argmax()]:.0f}Hz")
    print(f"Output power:{P_o.max():.2e} at {f_o[P_o.argmax()]:.0f}Hz")
    for m in [1,2,3,4,5]:
        fh = f0_mean * m
        if fh < 12000:
            idx = np.abs(f_o - fh).argmin()
            print(f"  H{m} ({fh:.0f}Hz): power={P_o[idx]:.2e}")
    
    # Signal-to-noise estimate: harmonic power / total
    harm_power = 0
    for m in [1,2,3,4,5,6]:
        fh = min(f0_mean * m, 11900)
        band = (np.abs(f_o - fh) <= 20)
        harm_power += P_o[band].sum()
    total_power = P_o.sum()
    print(f"Harmonic/Total power ratio: {harm_power/total_power:.4f}")

os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)
sf.write("/root/autodl-tmp/inference_output/final_diag.wav", wav, 24000)
print("Done")
