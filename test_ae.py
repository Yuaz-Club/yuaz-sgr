"""AE 重建测试: 加载测试集一段音频, encoder→rvq→ddsp 重建"""
import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, soundfile as sf, glob

config = {}
for n in ["base", "model", "train"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
_sd = torch.load("/root/autodl-tmp/checkpoints/model_040000.pt", weights_only=False, map_location="cpu")["model"]
for k, v in _sd.items():
    if "cond_encoder.phoneme_embed.weight" in k:
        config["conditioning"]["num_phonemes"] = v.shape[0]; break

from models.svs_model import SVSModel

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))[:2]
os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)

for ckpt_file, label in [("model_035000", "35k"), ("model_040000", "40k")]:
    ckpt_path = f"/root/autodl-tmp/checkpoints/{ckpt_file}.pt"
    if not os.path.exists(ckpt_path): continue
    m = SVSModel(config).cuda().eval()
    sd = torch.load(ckpt_path, weights_only=False)["model"]
    m.load_state_dict(sd, strict=False)

    for npz in files:
        data = np.load(npz, allow_pickle=True)
        a_orig = data["audio"].squeeze()
        a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()
        name = os.path.basename(npz).replace(".npz", "")

        with torch.no_grad():
            z1, f0 = m.encoder(a)
            z_q, _, _ = m.rvq(z1)
            fake = m.ddsp_decoder(f0, z_q, target_samples=a.shape[-1])

            dd = m.ddsp_decoder
            if f0.dim() == 2: f0_3 = f0.unsqueeze(1)
            else: f0_3 = f0
            feat = torch.cat([z_q, f0_3], dim=1).transpose(1, 2)
            feat = dd.emformer_input_proj(feat)
            h, _ = dd.emformer(feat)
            h = dd.emformer_proj(h).transpose(1, 2)
            w = dd.weight_net(h.transpose(1, 2)).transpose(1, 2)
            lh = torch.clamp(1.0 / (w[:, 0:1, :] + 1e-8), min=0).mean().item()
            ln_ = torch.clamp(1.0 / (w[:, 1:2, :] + 1e-8), min=0).mean().item()
            S_c = dd.env_net(h.transpose(1, 2)).transpose(1, 2)
            S_lin = dd._decompress_envelope(S_c)
            nonzero = (S_lin > 0).float().mean().item() * 100
            S_lin_low = S_lin[0, :50, :].mean().item()
            S_lin_high = S_lin[0, 400:, :].mean().item()

        wav_rec = fake.squeeze().cpu().numpy()
        wav_orig = a.squeeze().cpu().numpy()
        sf.write(f"/root/autodl-tmp/inference_output/{label}_ae_{name}.wav", wav_rec, 24000)
        if ckpt_file == "model_040000":
            sf.write(f"/root/autodl-tmp/inference_output/{label}_ae_{name}_orig.wav", wav_orig, 24000)

        import scipy.signal
        f_o, P_o = scipy.signal.welch(wav_rec, 24000, nperseg=2048)
        f0v = f0_3[f0_3 > 50].mean().item() if (f0_3 > 50).any() else 0
        hp = sum(P_o[np.abs(f_o - f0v * m) <= 20].sum() for m in range(1, 6) if f0v * m < 12000)
        harm = hp / (P_o.sum() + 1e-8)
        tilt = (P_o[np.abs(f_o - 2000) <= 200].sum()) / (P_o[np.abs(f_o - 8000) <= 500].sum() + 1e-8)

        print(f"[{label}] {name}: lh={lh:.3f} ln={ln_:.3f} harm={harm:.3f} tilt={tilt:.2f} S%={nonzero:.0f} Lo/Hi={S_lin_low/S_lin_high:.2f}")
print("Done!")
