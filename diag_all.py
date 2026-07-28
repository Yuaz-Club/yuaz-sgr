"""快速验证 AE→GAN 全阶段重建效果"""
import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, soundfile as sf, glob, importlib

config = {}
for n in ["base", "model", "train"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))

# 从最新 checkpoint 获取音素数
latest = "/root/autodl-tmp/checkpoints/model_030000.pt"
_sd = torch.load(latest, weights_only=False, map_location="cpu")["model"]
for k, v in _sd.items():
    if "cond_encoder.phoneme_embed.weight" in k:
        config["conditioning"]["num_phonemes"] = v.shape[0]
        break

from models.svs_model import SVSModel

ckpts = [
    ("model_005000.pt", "AE"),
    ("model_010000.pt", "GAN"),
    ("model_015000.pt", "GAN"),
    ("model_020000.pt", "GAN"),
    ("model_025000.pt", "GAN"),
    ("model_030000.pt", "GAN"),
]

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))[:2]
os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)

results = {}

for ckpt_file, stage in ckpts:
    ckpt_path = f"/root/autodl-tmp/checkpoints/{ckpt_file}"
    if not os.path.exists(ckpt_path):
        continue

    m = SVSModel(config).cuda().eval()
    sd = torch.load(ckpt_path, weights_only=False)["model"]
    m.load_state_dict(sd, strict=False)

    seg_metrics = []
    for npz in files:
        data = np.load(npz, allow_pickle=True)
        a_orig = data["audio"].squeeze()
        a = torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()
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
            Sc = dd.env_net(h.transpose(1, 2)).transpose(1, 2)
            S_lin = dd._decompress_envelope(Sc)
            nonzero = (S_lin > 0).float().mean().item() * 100

        wav_rec = fake.squeeze().cpu().numpy()
        wav_orig = a.squeeze().cpu().numpy()

        import scipy.signal
        f_o, P_o = scipy.signal.welch(wav_rec, 24000, nperseg=2048)
        f0v = f0_3[f0_3 > 50].mean().item() if (f0_3 > 50).any() else 0
        hp = sum(P_o[np.abs(f_o - f0v * m) <= 20].sum() for m in range(1, 6) if f0v * m < 12000)
        harm = hp / (P_o.sum() + 1e-8)

        seg_metrics.append({
            "lh": lh, "ln": ln_, "harm": harm, "nonzero": nonzero,
        })
        name = os.path.basename(npz).replace(".npz", "")
        sf.write(f"/root/autodl-tmp/inference_output/{ckpt_file.replace('.pt','')}_{name}.wav", wav_rec, 24000)

    avg = {k: np.mean([s[k] for s in seg_metrics]) for k in seg_metrics[0]}
    results[ckpt_file] = avg
    print(f"[{stage}] {ckpt_file}: lh={avg['lh']:.3f} ln={avg['ln']:.3f} harm={avg['harm']:.3f} S%={avg['nonzero']:.0f}")

print("\n=== 阶段趋势 ===")
print(f"{'CKPT':<18} {'lh':>6} {'ln':>6} {'harm':>6} {'S%':>5}")
for ckpt_file, stage in ckpts:
    if ckpt_file in results:
        r = results[ckpt_file]
        print(f"{ckpt_file:<18} {r['lh']:>6.3f} {r['ln']:>6.3f} {r['harm']:>6.3f} {r['nonzero']:>5.0f}")
print("Done!")
