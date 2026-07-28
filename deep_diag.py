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
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_005000.pt", weights_only=False)
sd = ckpt["model"]
model.load_state_dict(sd, strict=False)

import glob
files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
data = np.load(files[0], allow_pickle=True)
a_orig = data["audio"].squeeze()
a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()
print(f"Input: shape={a.shape}, rms={np.sqrt(np.mean(a_orig[:96000]**2)):.5f}")

dd = model.ddsp_decoder
with torch.no_grad():
    z1, f0 = model.encoder(a)
    print(f"\nf0: shape={f0.shape}, min={f0.min().item():.1f}, max={f0.max().item():.1f}")
    print(f"f0[0,0,:10]= {f0[0,0,:10].cpu().numpy()}")

    z_q, idx, _ = model.rvq(z1)
    T = z_q.shape[-1]
    B = z_q.shape[0]
    print(f"z_q: {z_q.shape}, T={T}, B={B}")

    # Manual DDSP forward - step by step
    if f0.dim() == 2: f0 = f0.unsqueeze(1)
    feat = torch.cat([z_q, f0], dim=1)
    feat = feat.transpose(1, 2)
    feat = dd.emformer_input_proj(feat)
    emf_raw = dd.emformer(feat)
    emf = emf_raw[0] if isinstance(emf_raw, tuple) else emf_raw
    print(f"\nemformer_out: mean={emf.mean().item():.4f}, std={emf.std().item():.4f}")

    w_raw = dd.weight_net(emf)
    lh = torch.clamp(1.0 / (w_raw[:,:,0:1] + 1e-8), min=0)
    ln = torch.clamp(1.0 / (w_raw[:,:,1:2] + 1e-8), min=0)
    print(f"lambda_h: mean={lh.mean().item():.5f}, lambda_n: mean={ln.mean().item():.5f}")

    S_c = dd.env_net(emf)
    print(f"S_c (compressed env): mean={S_c.mean().item():.4f}, min={S_c.min().item():.4f}, max={S_c.max().item():.4f}")
    
    S_lin = dd._decompress_envelope(S_c.transpose(1, 2))
    print(f"S_lin: mean={S_lin.mean().item():.5f}, min={S_lin.min().item():.5f}, max={S_lin.max().item():.5f}")
    print(f"  S_lin[0, 0, :5] = {S_lin[0, 0, :5].cpu().numpy()}")
    print(f"  S_lin[0, 100, :5] = {S_lin[0, 100, :5].cpu().numpy()}")
    print(f"  S_lin[0, 200, :5] = {S_lin[0, 200, :5].cpu().numpy()}")
    print(f"  S_lin[0, 400, :5] = {S_lin[0, 400, :5].cpu().numpy()}")
    if S_lin.shape[1] > 500:
        print(f"  S_lin[0, 512, :5] = {S_lin[0, 512, :5].cpu().numpy()}")

    A_c = dd.ap_net(emf)
    print(f"\nA_c (compressed ap): mean={A_c.mean().item():.4f}, min={A_c.min().item():.4f}, max={A_c.max().item():.4f}")
    A_c_t = A_c.transpose(1, 2)
    A_lin = F.interpolate(A_c_t, size=dd.fft_size // 2 + 1, mode='linear')
    A_lin = A_lin.transpose(1, 2)
    print(f"A_lin: mean={A_lin.mean().item():.5f}, min={A_lin.min().item():.5f}, max={A_lin.max().item():.5f}")

    n_samples = T * dd.hop_length
    e_h = dd.harmonic_oscillator(f0, n_samples)
    print(f"\nharmonics(e_h): mean={e_h.mean().item():.5f}, std={e_h.std().item():.5f}, max={torch.abs(e_h).max().item():.4f}")
    print(f"  e_h[0, :10] = {e_h[0, :10].cpu().numpy()}")
    
    spec_h = torch.stft(e_h.squeeze(1), dd.fft_size, dd.hop_length,
                        window=dd.hann_window.to(e_h.device),
                        return_complex=True)
    spec_h = spec_h[:, :dd.fft_size // 2 + 1, :T]
    print(f"spec_h: mean_abs={torch.abs(spec_h).mean().item():.5f}")

    nf = min(spec_h.shape[-1], S_lin.shape[-1])
    spec_h_filt = spec_h[:, :, :nf] * S_lin[:, :, :nf]
    x_h = torch.istft(spec_h_filt, dd.fft_size, dd.hop_length,
                      window=dd.hann_window.to(spec_h.device),
                      length=n_samples)
    print(f"x_h (harmonic): rms={torch.sqrt(torch.mean(x_h**2)).item():.5f}")

    e_n = torch.randn(B, n_samples, device=e_h.device)
    spec_n = torch.stft(e_n, dd.fft_size, dd.hop_length,
                        window=dd.hann_window.to(e_n.device),
                        return_complex=True)
    spec_n = spec_n[:, :dd.fft_size // 2 + 1, :T]
    nf_n = min(spec_n.shape[-1], A_lin.shape[-1], S_lin.shape[-1])
    spec_n_filt = spec_n[:, :, :nf_n] * A_lin[:, :, :nf_n] * S_lin[:, :, :nf_n]
    x_n = torch.istft(spec_n_filt, dd.fft_size, dd.hop_length,
                      window=dd.hann_window.to(spec_n.device),
                      length=n_samples)
    print(f"x_n (noise): rms={torch.sqrt(torch.mean(x_n**2)).item():.5f}")

    lh_up = F.interpolate(lh, size=n_samples, mode='linear')
    ln_up = F.interpolate(ln, size=n_samples, mode='linear')
    wav_full = lh_up * x_h.unsqueeze(1) + ln_up * x_n.unsqueeze(1)
    wav_np = wav_full.squeeze().cpu().numpy()
    rms_out = np.sqrt(np.mean(wav_np**2))
    print(f"\nFinal wav: rms={rms_out:.5f}, max={np.abs(wav_np).max():.4f}")

    # Also run the full DDSP forward for comparison
    fake = dd(f0, z_q, target_samples=a.shape[-1])
    wav2 = fake.squeeze().cpu().numpy()
    print(f"Full forward: rms={np.sqrt(np.mean(wav2**2)):.5f}, max={np.abs(wav2).max():.4f}")
    
import torch.nn.functional as F
os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)
sf.write("/root/autodl-tmp/inference_output/deep_diag.wav", wav_np, 24000)
sf.write("/root/autodl-tmp/inference_output/deep_diag_orig.wav", a_orig[:96000], 24000)
print("\nAll diagnostics complete!")
