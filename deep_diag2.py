import sys, os, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, soundfile as sf
import glob

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

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
data = np.load(files[0], allow_pickle=True)
a_orig = data["audio"].squeeze()
a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()

dd = model.ddsp_decoder
with torch.no_grad():
    z1, f0 = model.encoder(a)
    f0_2d = f0 if f0.dim() == 2 else f0.squeeze(1)
    z_q, _, _ = model.rvq(z1)
    B, _, T = z_q.shape

    # Harmonic oscillator standalone
    hop = dd.hop_length
    n_samples_calc = T * hop
    e_h = dd.harmonic_oscillator(f0, n_samples_calc)
    eh_wav = e_h.squeeze().cpu().numpy()
    print(f"Harmonic oscillator alone: rms={np.sqrt(np.mean(eh_wav**2)):.5f}, max={np.abs(eh_wav).max():.3f}")
    print(f"  first 1000 samples: {eh_wav[:10]}")

    # Full DDSP forward
    fake = dd(f0, z_q, target_samples=a.shape[-1])
    wav = fake.squeeze().cpu().numpy()
    print(f"\nFull output: rms={np.sqrt(np.mean(wav**2)):.5f}, max={np.abs(wav).max():.3f}")
    
    # Spectrum analysis
    import scipy.signal
    f_orig, Pxx_orig = scipy.signal.welch(a_orig[:96000], 24000, nperseg=2048)
    f_out, Pxx_out = scipy.signal.welch(wav, 24000, nperseg=2048)
    print(f"\nInput peak freq: {f_orig[np.argmax(Pxx_orig)]:.1f} Hz, power={Pxx_orig.max():.3e}")
    print(f"Output peak freq: {f_out[np.argmax(Pxx_out)]:.1f} Hz, power={Pxx_out.max():.3e}")
    
    # Check if harmonics are present
    f0_mean = f0_2d[f0_2d > 50].mean().item()
    print(f"\nMean F0: {f0_mean:.1f} Hz")
    for mult in [1, 2, 3, 4, 5]:
        f_harm = f0_mean * mult
        if f_harm < 12000:
            idx = np.argmin(np.abs(f_out - f_harm))
            print(f"  Harmonic {mult}x ({f_harm:.0f} Hz): power={Pxx_out[idx]:.3e}")
    
    # Check S_lin statistics
    if f0.dim() == 2: f0 = f0.unsqueeze(1)
    feat = torch.cat([z_q, f0], dim=1).transpose(1, 2)
    feat = dd.emformer_input_proj(feat)
    emf = dd.emformer(feat)
    emf = emf[0] if isinstance(emf, tuple) else emf
    S_c = dd.env_net(emf)
    S_lin_full = dd._decompress_envelope(S_c.transpose(1, 2))
    print(f"\nS_lin stats: mean={S_lin_full.mean():.5f}, median={S_lin_full.median():.5f}")
    nonzero = (S_lin_full > 1e-5).float().mean().item()
    print(f"S_lin > 1e-5 fraction: {nonzero:.3f}")
    
    # Print S_lin for key frequency bins
    for fbin_desc, fbin_idx in [("50Hz (~2nd mel)", 2), ("200Hz (~12th mel)", 12), ("500Hz (~30th mel)", 30), ("1000Hz (~60th mel)", 60), ("2000Hz (~90th mel)", 90), ("4000Hz (~110th mel)", 110)]:
        if fbin_idx < S_lin_full.shape[1]:
            vals = S_lin_full[0, fbin_idx, :10].cpu().numpy()
            print(f"  S_lin at {fbin_desc} bin {fbin_idx}: {vals}")

os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)
sf.write("/root/autodl-tmp/inference_output/deep_diag.wav", wav, 24000)
sf.write("/root/autodl-tmp/inference_output/deep_diag_orig.wav", a_orig[:96000], 24000)
print("\nDone!")
