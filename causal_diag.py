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
print(f"Input rms={np.sqrt(np.mean(a_orig[:96000]**2)):.5f}")

print("\n=== 随机权重 DDSP ===")
m1 = SVSModel(config).cuda().eval()
with torch.no_grad():
    z1_r, f0_r = m1.encoder(a)
    z_q_r, _, _ = m1.rvq(z1_r)
    fake_r = m1.ddsp_decoder(f0_r, z_q_r, target_samples=a.shape[-1])
wr = fake_r.squeeze().cpu().numpy()
f_o, P_o = scipy.signal.welch(wr, 24000, nperseg=2048)
f0v = f0_r[f0_r > 50].mean().item() if (f0_r > 50).any() else 0
hp = sum(P_o[np.abs(f_o - f0v*m) <= 20].sum() for m in [1,2,3,4,5] if f0v*m < 12000)
print(f"rms={np.sqrt(np.mean(wr**2)):.5f}  max={np.abs(wr).max():.2f}  harm/total={hp/P_o.sum():.4f}")

print("\n=== 20k 训练 DDSP ===")
m2 = SVSModel(config).cuda().eval()
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_020000.pt", weights_only=False)
m2.load_state_dict(ckpt["model"], strict=False)
with torch.no_grad():
    z1_t, f0_t = m2.encoder(a)
    z_q_t, _, _ = m2.rvq(z1_t)
    fake_t = m2.ddsp_decoder(f0_t, z_q_t, target_samples=a.shape[-1])
wt = fake_t.squeeze().cpu().numpy()
f_o2, P_o2 = scipy.signal.welch(wt, 24000, nperseg=2048)
f0v2 = f0_t[f0_t > 50].mean().item() if (f0_t > 50).any() else 0
hp2 = sum(P_o2[np.abs(f_o2 - f0v2*m) <= 20].sum() for m in [1,2,3,4,5] if f0v2*m < 12000)
print(f"rms={np.sqrt(np.mean(wt**2)):.5f}  max={np.abs(wt).max():.2f}  harm/total={hp2/P_o2.sum():.4f}")

# Weight diff analysis
dd1, dd2 = m1.ddsp_decoder, m2.ddsp_decoder
print("\n=== DDSP权重要素 ===")
for name, p1, p2 in [("env_net.L0.w", dd1.env_net[0].weight, dd2.env_net[0].weight),
                      ("ap_net.L0.w", dd1.ap_net[0].weight, dd2.ap_net[0].weight),
                      ("weight_net.L0.w", dd1.weight_net[0].weight, dd2.weight_net[0].weight),
                      ("weight_net.L2.bias", dd1.weight_net[2].bias, dd2.weight_net[2].bias)]:
    d = (p1 - p2).abs().mean().item()
    print(f"  {name:25s} |Δ|={d:.6f}")

# Internal values for trained model
print("\n=== 训练DDSP中间值 ===")
dd = dd2
with torch.no_grad():
    if f0_t.dim() == 2: f0_t = f0_t.unsqueeze(1)
    feat = torch.cat([z_q_t, f0_t], dim=1).transpose(1, 2)
    feat = dd.emformer_input_proj(feat)
    h, _ = dd.emformer(feat)
    h = dd.emformer_proj(h).transpose(1, 2)
    Sc = dd.env_net(h.transpose(1, 2)).transpose(1, 2)
    Ac = dd.ap_net(h.transpose(1, 2)).transpose(1, 2)
    w = dd.weight_net(h.transpose(1, 2)).transpose(1, 2)
    lh = torch.clamp(1.0 / (w[:, 0:1, :] + 1e-8), min=0)
    ln = torch.clamp(1.0 / (w[:, 1:2, :] + 1e-8), min=0)
    S_lin = dd._decompress_envelope(Sc)
    A_lin = F.interpolate(Ac.transpose(1,2), size=dd.fft_size//2+1, mode='linear').transpose(1,2)

print(f"S_lin: mean={S_lin.mean():.3f} zeros={(S_lin<1e-6).float().mean()*100:.0f}%")
print(f"A_lin: mean={A_lin.mean():.4f}")
print(f"lambda_h={lh.mean():.4f} lambda_n={ln.mean():.4f}")
print(f"S_lin[0,:8,0]={S_lin[0,:8,0].cpu().numpy()}")

os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)
sf.write("/root/autodl-tmp/inference_output/diag_random.wav", wr, 24000)
sf.write("/root/autodl-tmp/inference_output/diag_trained.wav", wt, 24000)
sf.write("/root/autodl-tmp/inference_output/diag_orig.wav", a_orig[:96000], 24000)
print("Done!")
