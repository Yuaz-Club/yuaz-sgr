"""诊断推理咬字问题：对比 AE 和 Flow 路径"""
import os, sys, json
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf

sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel

# 加载配置
config = {}
for name in ['base', 'model']:
    import yaml
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']

device = 'cuda'; sr = 24000
CKPT = sorted([f for f in os.listdir('/root/autodl-tmp/yuaz_sgr/checkpoints') if f.startswith('model_0')])[-1]
CKPT = '/root/autodl-tmp/yuaz_sgr/checkpoints/' + CKPT
OUT = '/root/autodl-tmp/diag_infer'
os.makedirs(OUT, exist_ok=True)
print(f"Checkpoint: {CKPT}")

# 加载模型
model = SVSModel(config).to(device)
model.eval()
ckpt = torch.load(CKPT, map_location=device)
model_sd = model.state_dict()
skp = 0
for k, v in ckpt['model'].items():
    clean_k = k.replace('_orig_mod.', '')
    if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
        model_sd[clean_k] = v
    else:
        skp += 1
model.load_state_dict(model_sd, strict=False)
print(f"  跳过 {skp} 参数, step={ckpt['step']}")

# 取一个训练样本
from data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json',
    bs=1, device=device,
)
batch = next(train_iter())
audio = batch['audio']; ph = batch['phoneme_ids']; dur = batch['durations']
f0_gt = batch['f0_audio']; nl = batch['note_lengths']
T_audio = audio.shape[-1]

sf.write(f'{OUT}/orig.wav', audio.squeeze().cpu().numpy() / (audio.abs().max()+1e-8).item(), sr)
print(f"audio={list(audio.shape)} dur={list(dur.shape)}")

# ========== 1. AE 路径 (encoder z → decoder) ==========
print("\n--- AE路径 ---")
with torch.no_grad():
    z1, f0_e = model.encoder(audio, f0_override=f0_gt)
    z_q_ae, _, _ = model.rvq(z1)
    ae_wav = model.ddsp_decoder(f0_e, z_q_ae, target_samples=T_audio)
ae_wav = ae_wav.squeeze().cpu().numpy(); ae_wav = ae_wav / (np.abs(ae_wav).max()+1e-8)
sf.write(f'{OUT}/ae_recon.wav', ae_wav, sr)
print(f"z1: mean={z1.mean():.4f} std={z1.std():.4f} shape={list(z1.shape)}")

# ========== 2. Flow 路径 (cond_encoder → flow → decoder) ==========
print("\n--- Flow路径 ---")
f0_sq = f0_e.squeeze(1)  # encoder F0 (训练分布)
with torch.no_grad():
    mu = model.cond_encoder(ph, dur, f0_sq)
    print(f"mu: mean={mu.mean():.4f} std={mu.std():.4f} shape={list(mu.shape)}")

    # 8步ODE
    z_flow_8, f0_ref_8 = model.flow_gen.sample(mu, n_steps=8)
    z_q_f8, _, _ = model.rvq(z_flow_8)
    f0_f8 = f0_e[:, :, :z_q_f8.shape[-1]]
    flow8_wav = model.ddsp_decoder(f0_f8, z_q_f8, target_samples=T_audio)
    
    # 1步ODE
    z_flow_1, f0_ref_1 = model.flow_gen.sample(mu, n_steps=1)
    z_q_f1, _, _ = model.rvq(z_flow_1)
    f0_f1 = f0_e[:, :, :z_q_f1.shape[-1]]
    flow1_wav = model.ddsp_decoder(f0_f1, z_q_f1, target_samples=T_audio)

flow8 = flow8_wav.squeeze().cpu().numpy(); flow8 = flow8 / (np.abs(flow8).max()+1e-8)
flow1 = flow1_wav.squeeze().cpu().numpy(); flow1 = flow1 / (np.abs(flow1).max()+1e-8)
sf.write(f'{OUT}/flow_8step.wav', flow8, sr)
sf.write(f'{OUT}/flow_1step.wav', flow1, sr)

# z 对比
Tmin = min(z1.shape[-1], z_flow_8.shape[-1])
cos_8 = F.cosine_similarity(z1[:,:,:Tmin].reshape(-1), z_flow_8[:,:,:Tmin].reshape(-1), dim=0).item()
cos_1 = F.cosine_similarity(z1[:,:,:Tmin].reshape(-1), z_flow_1[:,:,:Tmin].reshape(-1), dim=0).item()
mse_8 = F.mse_loss(z1[:,:,:Tmin], z_flow_8[:,:,:Tmin]).item()
mse_1 = F.mse_loss(z1[:,:,:Tmin], z_flow_1[:,:,:Tmin]).item()
print(f"z_flow vs z1: 8step Cosine={cos_8:.4f} MSE={mse_8:.4f}")
print(f"z_flow vs z1: 1step Cosine={cos_1:.4f} MSE={mse_1:.4f}")
print(f"z1: mean={z1[:,:,:Tmin].mean():.4f} std={z1[:,:,:Tmin].std():.4f}")
print(f"z_flow_8: mean={z_flow_8[:,:,:Tmin].mean():.4f} std={z_flow_8[:,:,:Tmin].std():.4f}")
print(f"z_flow_1: mean={z_flow_1[:,:,:Tmin].mean():.4f} std={z_flow_1[:,:,:Tmin].std():.4f}")

# F0_refined 诊断
if f0_ref_8 is not None:
    f0r = f0_ref_8.abs()
    print(f"\nF0_refined(8step): mean={f0r.mean():.1f} std={f0r.std():.1f} min={f0r.min():.1f} max={f0r.max():.1f}")
    print(f"Encoder F0:        mean={f0_e.abs().mean():.1f} std={f0_e.abs().std():.1f}")

# ========== 3. MIDI F0 推理（模拟真实推理） ==========
print("\n--- MIDI F0 推理 ---")
# 用 encoder F0 构造步进 F0 (模拟 MIDI)
from scipy.ndimage import gaussian_filter1d
f0r = f0_gt.squeeze().cpu().numpy()
f0s = gaussian_filter1d(f0r.astype(np.float64), sigma=6.0)
nl_c = nl.squeeze().cpu().numpy().astype(int)
f0_step = np.zeros_like(f0s); pos = 0
for n in nl_c:
    if n <= 0: continue
    e = min(pos + n, len(f0_step))
    if e > pos: f0_step[pos:e] = np.mean(f0s[pos:e])
    pos = e
f0_t = torch.FloatTensor(f0_step[:len(f0r)]).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    wav, mu_i, z_i, f0_ref_i, lag, dur_p = model.forward_synthesis(
        ph, dur, f0_t, n_steps=8, note_lengths=nl
    )
wav_midi = wav.squeeze().cpu().numpy(); wav_midi = wav_midi / (np.abs(wav_midi).max()+1e-8)
sf.write(f'{OUT}/forward_synth_8.wav', wav_midi, sr)

# 检查 F0_refined
f0r_i = f0_ref_i.abs()
print(f"F0_refined(forward_synth): mean={f0r_i.mean():.1f} std={f0r_i.std():.1f}")
print(f"Input F0(MIDI):   mean={f0_t.abs().mean():.1f} std={f0_t.abs().std():.1f}")

# 检查 μ 差异 (训练 F0 的 μ vs MIDI F0 的 μ)
with torch.no_grad():
    mu_train = model.cond_encoder(ph, dur, f0_sq)
    mu_midi = model.cond_encoder(ph, dur, f0_t.squeeze(1))
Tmu = min(mu_train.shape[-1], mu_midi.shape[-1])
cos_mu = F.cosine_similarity(mu_train[:,:,:Tmu].reshape(-1), mu_midi[:,:,:Tmu].reshape(-1), dim=0).item()
print(f"\nmu(train_F0) vs mu(MIDI_F0): Cosine={cos_mu:.4f}")
print(f"mu_train: mean={mu_train.mean():.4f} std={mu_train.std():.4f}")
print(f"mu_midi:  mean={mu_midi.mean():.4f} std={mu_midi.std():.4f}")

# z 质量对比 (full forward_synthesis)
Tmin2 = min(z1.shape[-1], z_i.shape[-1])
cos_fs = F.cosine_similarity(z1[:,:,:Tmin2].reshape(-1), z_i[:,:,:Tmin2].reshape(-1), dim=0).item()
mse_fs = F.mse_loss(z1[:,:,:Tmin2], z_i[:,:,:Tmin2]).item()
print(f"\nforward_synth z vs encoder z1: Cosine={cos_fs:.4f} MSE={mse_fs:.4f}")
print(f"z_i: mean={z_i[:,:,:Tmin2].mean():.4f} std={z_i[:,:,:Tmin2].std():.4f}")

print(f"\nDone! {OUT}/")
print(f"  ae_recon.wav      — AE重建 (encoder z → decoder)")
print(f"  flow_1step.wav     — Flow 1步 → decoder (encoder F0)")
print(f"  flow_8step.wav     — Flow 8步 → decoder (encoder F0)")
print(f"  forward_synth_8.wav — 完整推理 (MIDI F0, 8步)")
