"""单样本过拟合 v2 — 严格对标论文 §4.2 训练流程

关键修复 (vs overfit_final.py):
  1. 两阶段训练: 阶段1(AE-only, 1500步) → 阶段2(全联合, 1500步)
     对标论文: AE pretraining → joint optimization
  2. Flow-through-decoder 路径: CFM 预测 → RVQ → decoder → STFT loss
     确保 flow 输出能被 decoder 正确解码 (论文 §4.2 joint)
  3. F0_refined 监督: flow 第129通道 MSE 对齐 encoder F0
  4. env_net 随机初始化: 各 mel 通道不同初始值, 产生非平坦频谱包络
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel
from losses.stft_loss import MultiResolutionSTFTLoss

# config
config = {}
for name in ['base', 'model']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        import yaml
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']

device = 'cuda'; sr = 24000
AE_STEPS = 1500   # 论文: AE 预训练
JT_STEPS = 2500   # 论文: 全联合 (加长使 flow z 更接近 encoder z)
TOTAL = AE_STEPS + JT_STEPS
OUT = '/root/autodl-tmp/overfit_v2'

# ── 数据 ──
from yuaz_sgr.data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json',
    bs=1, device=device,
)
batch = next(train_iter())
audio = batch['audio']; ph = batch['phoneme_ids']; dur = batch['durations']
f0_gt = batch['f0_audio']; nl = batch['note_lengths']
T_audio = audio.shape[-1]

os.makedirs(OUT, exist_ok=True)
sf.write(f'{OUT}/orig.wav', audio.squeeze().cpu().numpy() / (audio.abs().max()+1e-8).item(), sr)
print(f"audio={list(audio.shape)} dur={list(dur.shape)}")

# ── 模型 (env_net 随机偏置初始化: 各通道不同, 产生非平坦频谱包络) ──
model = SVSModel(config).to(device)
# 覆盖 env_net 偏置: 随机初始化 (mean=-2.8 控制振幅, std=0.5 产生通道差异)
nn = torch.nn
nn.init.normal_(model.ddsp_decoder.env_net[-2].bias, mean=-2.0, std=0.5)
model.freeze_f0_encoder()
model.train()
print(f"params: {sum(p.numel() for p in model.generator_params())/1e6:.1f}M")

# 论文: Adam, lr=1e-5, β₁=0.8, β₂=0.999
opt = torch.optim.Adam(model.generator_params(), lr=1e-4, betas=(0.8, 0.999))
scl = torch.cuda.amp.GradScaler()
stft_fn = MultiResolutionSTFTLoss()

# ── 阶段 1: AE 预训练 (对标论文 0→20k AE-only) ──
print(f"\n{'='*50}")
print(f"阶段 1: AE 预训练 ({AE_STEPS} 步)")
print(f"{'='*50}")
t0 = time.time()
for s in range(AE_STEPS):
    with torch.cuda.amp.autocast():
        z1, f0_e = model.encoder(audio, f0_override=f0_gt)
        z_q, _, rvq = model.rvq(z1)
        fw = model.ddsp_decoder(f0_e, z_q, target_samples=T_audio)
        stft = stft_fn(audio, fw)
        loss = stft + 0.5 * rvq

    opt.zero_grad()
    scl.scale(loss).backward()
    scl.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.generator_params(), 5.0)
    scl.step(opt)
    scl.update()

    if s % 300 == 0 or s == AE_STEPS - 1:
        print(f"AE step {s:04d} loss={loss.item():.3f} stft={stft.item():.2f} rvq={rvq.item():.4f} [{time.time()-t0:.0f}s]")

# ── 阶段 2: 全联合训练 (对标论文 200k→250k joint) ──
print(f"\n{'='*50}")
print(f"阶段 2: 全联合训练 ({JT_STEPS} 步)")
print(f"{'='*50}")
t0 = time.time()
for s in range(JT_STEPS):
    with torch.cuda.amp.autocast():
        # —— AE 路径 (encoder → RVQ → decoder) ——
        z1, f0_e = model.encoder(audio, f0_override=f0_gt)
        z_q, _, rvq = model.rvq(z1)
        fw_ae = model.ddsp_decoder(f0_e, z_q, target_samples=T_audio)
        stft_ae = stft_fn(audio, fw_ae)
        loss = stft_ae + 0.5 * rvq

        # —— 条件编码器 ——
        mu = model.cond_encoder(ph, dur, f0_e.squeeze(1))
        T = min(z1.shape[-1], mu.shape[-1])

        # 论文 §3.2 Eq(19): 特征对齐 L_feature
        feature_loss = 0.1 * F.mse_loss(mu[:, :, :T], z1[:, :, :T].detach())
        loss = loss + feature_loss

        # —— CFM: 训练 flow generator (论文 §3.2 Eq(18)) ——
        z0_cfm = torch.randn_like(z1)
        t_cfm = torch.rand(1, device=device)
        z_t_cfm = (1 - t_cfm) * z0_cfm[:, :, :T] + t_cfm * z1[:, :, :T].detach()
        gen_out = model.flow_gen(z_t_cfm, t_cfm.unsqueeze(0), mu[:, :, :T])

        # 速度场损失 (前 128 通道)
        v_pred = gen_out[:, :128, :]
        v_target = z1[:, :, :T].detach() - z0_cfm[:, :, :T]
        weight_cfm = 1.0 / (1.0 - t_cfm + 1e-6)
        cfm_loss = (weight_cfm * ((v_pred - v_target) ** 2)).mean()
        loss = loss + cfm_loss

        # ★ 关键修复 1: Flow-through-Decoder 路径 ★
        # 论文 §4.2: "Full joint optimization of encoder, decoder, flow predictor"
        # CFM 预测 → RVQ → decoder → STFT loss
        # 确保 flow 输出能被 decoder 正确解码
        z_pred = z0_cfm[:, :, :T] + v_pred  # z₁ ≈ z₀ + v
        z_q_pred, _, rvq_pred = model.rvq(z_pred)
        f0_cfm = f0_e[:, :, :T]
        fw_flow = model.ddsp_decoder(f0_cfm, z_q_pred, target_samples=T_audio)
        stft_flow = stft_fn(audio, fw_flow)
        loss = loss + 0.7 * stft_flow  # 提高权重, 强制 decoder 适应 flow z 分布

        # ★ 关键修复 2: F0_refined 监督 ★
        # 论文 §4.3.2: flow 第 129 通道 = 精炼 F0
        # 归一化到 440Hz 范围 (论文 A4=440Hz)
        f0_refined = gen_out[:, 128:129, :] / 440.0
        f0_enc_cfm = f0_e[:, :, :T] / 440.0
        if f0_enc_cfm.shape[-1] != f0_refined.shape[-1]:
            f0_enc_cfm = F.interpolate(f0_enc_cfm, size=f0_refined.shape[-1], mode='linear')
        f0_ref_loss = 0.01 * F.mse_loss(f0_refined, f0_enc_cfm.detach())
        loss = loss + f0_ref_loss

    opt.zero_grad()
    scl.scale(loss).backward()
    scl.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.generator_params(), 5.0)
    scl.step(opt)
    scl.update()

    if s % 300 == 0 or s == JT_STEPS - 1:
        print(f"JT step {s:04d} loss={loss.item():.3f} stft_ae={stft_ae.item():.2f} "
              f"stft_flow={stft_flow.item():.2f} cfm={cfm_loss.item():.3f} "
              f"f0_ref={f0_ref_loss.item():.4f} [{time.time()-t0:.0f}s]")

# ── 评估 ──
model.eval()
print(f"\n{'='*50}")
print("评估")
print(f"{'='*50}")

# AE 重建
with torch.no_grad(), torch.cuda.amp.autocast():
    z1, f0_e = model.encoder(audio, f0_override=f0_gt)
    z_q, _, _ = model.rvq(z1)
    ae = model.ddsp_decoder(f0_e, z_q, target_samples=T_audio)
ae = ae.squeeze().cpu().numpy(); ae = ae / (np.abs(ae).max() + 1e-8)
sf.write(f'{OUT}/ae.wav', ae, sr)
print(f"AE: {len(ae)/sr:.1f}s")

# 平滑 F0 (推理用)
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

# 推理: 8 步 + 1 步
with torch.no_grad(), torch.cuda.amp.autocast():
    w8, _, _, _, _, _ = model.forward_synthesis(ph, dur, f0_t, n_steps=8, note_lengths=nl)
    w1, _, _, _, _, _ = model.forward_synthesis(ph, dur, f0_t, n_steps=1, note_lengths=nl)

for name, w in [('infer_8', w8), ('infer_1', w1)]:
    w = w.squeeze().cpu().numpy(); w = w / (np.abs(w).max() + 1e-8)
    sf.write(f'{OUT}/{name}.wav', w, sr)
    print(f"{name}: {len(w)/sr:.1f}s")

# 额外: 直接使用 encoder 的 z1 通过 flow 路径解码 (诊断用)
with torch.no_grad(), torch.cuda.amp.autocast():
    z1_direct, f0_e_d = model.encoder(audio, f0_override=f0_gt)
    z_q_direct, _, _ = model.rvq(z1_direct)
    fw_direct = model.ddsp_decoder(f0_e_d, z_q_direct, target_samples=T_audio)
w_direct = fw_direct.squeeze().cpu().numpy(); w_direct = w_direct / (np.abs(w_direct).max() + 1e-8)
sf.write(f'{OUT}/ae_direct.wav', w_direct, sr)
print(f"AE_direct: {len(w_direct)/sr:.1f}s")

# 诊断: flow 预测的 z 与 encoder z1 的差异
with torch.no_grad(), torch.cuda.amp.autocast():
    mu_diag = model.cond_encoder(ph, dur, f0_t.squeeze(1))
    z_flow, f0_ref = model.flow_gen.sample(mu_diag, n_steps=8)
    T_diag = min(z1_direct.shape[-1], z_flow.shape[-1])
    mse_z = F.mse_loss(z1_direct[:, :, :T_diag], z_flow[:, :, :T_diag]).item()
    cos_z = F.cosine_similarity(
        z1_direct[:, :, :T_diag].reshape(-1),
        z_flow[:, :, :T_diag].reshape(-1), dim=0
    ).item()
    f0r = f0_ref.abs()
    print(f"\nFlow z vs Encoder z1: MSE={mse_z:.4f} Cosine={cos_z:.4f}")
    print(f"F0_refined: mean={f0r.mean().item():.1f} std={f0r.std().item():.1f} "
          f"min={f0r.min().item():.1f} max={f0r.max().item():.1f}")
    print(f"F0 input:    mean={f0_t.abs().mean().item():.1f} std={f0_t.abs().std().item():.1f}")

print(f"\n完成: {OUT}/orig.wav ae.wav infer_8.wav infer_1.wav ae_direct.wav")
