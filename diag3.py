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

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
data = np.load(files[0], allow_pickle=True)
a_orig = data["audio"].squeeze()
a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()

dd = m.ddsp_decoder
with torch.no_grad():
    z1, f0 = m.encoder(a)
    z_q, _, _ = m.rvq(z1)
    T = z_q.shape[-1]
    B = z_q.shape[0]

    # Manually trace DDSP forward
    if f0.dim() == 2: f0 = f0.unsqueeze(1)
    feat = torch.cat([z_q, f0], dim=1)
    feat = feat.transpose(1, 2)
    feat = dd.emformer_input_proj(feat)
    emf = dd.emformer(feat, T=T)
    print(f"emformer_out: {emf.shape}, mean={emf.mean().item():.4f}, std={emf.std().item():.4f}")

    S_c = dd.env_net(emf)
    print(f"S_c: {S_c.shape}, mean={S_c.mean().item():.4f}, max={S_c.max().item():.2f}")
    
    A_c = dd.ap_net(emf)
    print(f"A_c: {A_c.shape}, mean={A_c.mean().item():.4f}")

    w_raw = dd.weight_net(emf)
    print(f"weight_net raw: mean w0={w_raw[:,:,0].mean().item():.4f}, w1={w_raw[:,:,1].mean().item():.4f}")
    
    lh = torch.clamp(1.0 / (w_raw[:,:,0:1] + 1e-8), min=0)
    ln = torch.clamp(1.0 / (w_raw[:,:,1:2] + 1e-8), min=0)
    print(f"lambda_h mean={lh.mean().item():.4f}, lambda_n mean={ln.mean().item():.4f}")

    S_c_t = S_c.transpose(1, 2)
    S_lin = dd._decompress_envelope(S_c_t)
    print(f"S_lin: {S_lin.shape}, mean={S_lin.mean().item():.4f}, max={S_lin.max().item():.4f}")

    # Check harmonics alone
    n_samples = T * dd.hop_length
    print(f"n_samples={n_samples}, T={T}")
    e_h = dd.harmonic_oscillator(f0, n_samples)
    print(f"harmonics: mean={e_h.mean().item():.4f}, std={e_h.std().item():.4f}, max={torch.abs(e_h).max().item():.4f}")
    
    # Forward pass w/ defaults but check intermediate norms of x_h, x_n
    fake = dd(f0, z_q, target_samples=a.shape[-1])
    wav = fake.squeeze().cpu().numpy()
    print(f"Final output RMS={np.sqrt(np.mean(wav**2)):.4f}, max={np.abs(wav).max():.4f}")
