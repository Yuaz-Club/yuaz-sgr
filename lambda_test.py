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
data = np.load(files[0], allow_pickle=True)
a_orig = data["audio"].squeeze()
a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()

m2 = SVSModel(config).cuda().eval()
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_020000.pt", weights_only=False)
m2.load_state_dict(ckpt["model"], strict=False)

dd = m2.ddsp_decoder
with torch.no_grad():
    z1_t, f0_t = m2.encoder(a)
    z_q_t, _, _ = m2.rvq(z1_t)
    if f0_t.dim() == 2: f0_t = f0_t.unsqueeze(1)
    B, _, T = f0_t.shape
    
    feat = torch.cat([z_q_t, f0_t], dim=1).transpose(1, 2)
    feat = dd.emformer_input_proj(feat)
    h, _ = dd.emformer(feat)
    h = dd.emformer_proj(h).transpose(1, 2)
    
    Sc = dd.env_net(h.transpose(1, 2)).transpose(1, 2)
    Ac = dd.ap_net(h.transpose(1, 2)).transpose(1, 2)
    
    S_lin = dd._decompress_envelope(Sc)
    A_lin = F.interpolate(Ac.transpose(1,2), size=dd.fft_size//2+1, mode='linear').transpose(1,2)
    
    n_samples = T * dd.hop_length
    e_h = dd.harmonic_oscillator(f0_t, n_samples)
    spec_h = torch.stft(e_h.squeeze(1), dd.fft_size, dd.hop_length,
                        window=dd.hann_window, return_complex=True)
    spec_h = spec_h[:, :dd.fft_size // 2 + 1, :T]
    nf = min(spec_h.shape[-1], S_lin.shape[-1])
    spec_h_filt = spec_h[:, :, :nf] * S_lin[:, :, :nf]
    x_h = torch.istft(spec_h_filt, dd.fft_size, dd.hop_length,
                      window=dd.hann_window, length=n_samples)
    
    e_n = torch.randn(B, n_samples, device=z_q_t.device)
    spec_n = torch.stft(e_n, dd.fft_size, dd.hop_length,
                        window=dd.hann_window, return_complex=True)
    spec_n = spec_n[:, :dd.fft_size // 2 + 1, :T]
    nf_n = min(spec_n.shape[-1], A_lin.shape[-1], S_lin.shape[-1])
    spec_n_filt = spec_n[:, :, :nf_n] * A_lin[:, :, :nf_n] * S_lin[:, :, :nf_n]
    x_n = torch.istft(spec_n_filt, dd.fft_size, dd.hop_length,
                      window=dd.hann_window, length=n_samples)

# Test different λ ratios
os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)
for lh_val, ln_val, label in [(1.0, 0.0, "pure_harmonic"),
                                (0.9, 0.1, "h90_n10"),
                                (0.5, 0.5, "h50_n50"),
                                (0.1, 0.1, "low_both"),
                                (0.104, 3.55, "trained")]:
    wav = (lh_val * x_h.unsqueeze(1) + ln_val * x_n.unsqueeze(1)).squeeze().cpu().numpy()
    f_o, P_o = scipy.signal.welch(wav, 24000, nperseg=2048)
    f0v2 = f0_t[f0_t > 50].mean().item()
    hp = sum(P_o[np.abs(f_o - f0v2*m) <= 20].sum() for m in [1,2,3,4,5] if f0v2*m < 12000)
    rms_v = np.sqrt(np.mean(wav**2))
    print(f"{label:15s}: rms={rms_v:.4f} max={np.abs(wav).max():.3f} harm/total={hp/P_o.sum():.3f}")
    sf.write(f"/root/autodl-tmp/inference_output/test_{label}.wav", wav, 24000)

sf.write("/root/autodl-tmp/inference_output/test_orig.wav", a_orig[:96000], 24000)
print("Done!")
