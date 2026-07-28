"""单样本过拟合 v3 — 严格对标当前 trainer.py 训练流程

变更 (vs overfit_v2.py):
  1. F0Predictor 替代旧 flow matching 第129通道 f0_refined 作为推理 F0 源
  2. DDSP decoder 推理时不传 cond=mu (修复咬字模糊)
  3. F0Predictor 训练: 输入阶梯 F0 → μ → 预测平滑 CREPE F0 (L1 loss)
  4. 两阶段训练: AE(1500步) → Joint(2500步)
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
AE_STEPS = 1500
JT_STEPS = 2500
TOTAL = AE_STEPS + JT_STEPS
OUT = '/root/autodl-tmp/overfit_v3'

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

# ── 模型 ──
model = SVSModel(config).to(device)
nn = torch.nn
nn.init.normal_(model.ddsp_decoder.env_net[-2].bias, mean=-2.0, std=0.5)
model.freeze_f0_encoder()
model.train()
print(f"params: {sum(p.numel() for p in model.generator_params())/1e6:.1f}M")

opt = torch.optim.Adam(model.generator_params(), lr=1e-4, betas=(0.8, 0.999))
scl = torch.cuda.amp.GradScaler()
stft_fn = MultiResolutionSTFTLoss()

# ── 阶段 1: AE 预训练 ──
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

# ── 阶段 2: 全联合训练 ──
print(f"\n{'='*50}")
print(f"阶段 2: 全联合训练 ({JT_STEPS} 步)")
print(f"{'='*50}")
t0 = time.time()
for s in range(JT_STEPS):
    with torch.cuda.amp.autocast():
        # —— AE 路径 ——
        z1, f0_e = model.encoder(audio, f0_override=f0_gt)
        z_q, _, rvq = model.rvq(z1)
        fw_ae = model.ddsp_decoder(f0_e, z_q, target_samples=T_audio)
        stft_ae = stft_fn(audio, fw_ae)
        loss = stft_ae + 0.5 * rvq

        # —— 条件编码器 ——
        f0_sq = f0_e.squeeze(1)
        mu = model.cond_encoder(ph, dur, f0_sq)
        T = min(z1.shape[-1], mu.shape[-1])

        # 特征对齐
        feature_loss = 0.1 * F.mse_loss(mu[:, :, :T], z1[:, :, :T].detach())
        loss = loss + feature_loss

        # —— CFM ——
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

        # Flow-through-decoder 路径
        z_pred = z0_cfm[:, :, :T] + v_pred
        z_q_pred, _, _ = model.rvq(z_pred)
        f0_cfm = f0_e[:, :, :T]
        fw_flow = model.ddsp_decoder(f0_cfm, z_q_pred, target_samples=T_audio)
        stft_flow = stft_fn(audio, fw_flow)
        loss = loss + 0.7 * stft_flow

        # F0_refined 监督 (flow 第129通道)
        f0_refined = gen_out[:, 128:129, :] / 440.0
        f0_enc_cfm = f0_e[:, :, :T] / 440.0
        if f0_enc_cfm.shape[-1] != f0_refined.shape[-1]:
            f0_enc_cfm = F.interpolate(f0_enc_cfm, size=f0_refined.shape[-1], mode='linear')
        f0_ref_loss = 0.01 * F.mse_loss(f0_refined, f0_enc_cfm.detach())
        loss = loss + f0_ref_loss

        # ★ F0Predictor 训练 ★
        # 输入: 阶梯 F0 (每音符均值) → μ → F0Predictor → 预测平滑 CREPE F0
        note_lens = nl
        f0_step = torch.zeros_like(f0_sq)
        B = f0_sq.shape[0]
        for b in range(B):
            pos = 0
            for n in range(note_lens.shape[1]):
                n_len = int(note_lens[b, n].item())
                if n_len <= 0:
                    continue
                end = min(pos + n_len, f0_sq.shape[-1])
                if end > pos:
                    f0_step[b, pos:end] = f0_sq[b, pos:end].mean()
                pos = end
        mu_f0 = model.cond_encoder(ph, dur, f0_step)
        T_cfm = min(mu_f0.shape[-1], z1.shape[-1])
        f0_pred = model.f0_predictor(mu_f0[:, :, :T_cfm].detach())
        f0_enc_norm = f0_e[:, :, :T_cfm] / 2000.0  # 归一化到 [0,~1]
        if f0_pred.shape[-1] != f0_enc_norm.shape[-1]:
            f0_enc_norm = F.interpolate(f0_enc_norm, size=f0_pred.shape[-1], mode='linear')
        f0_pred_loss = F.l1_loss(f0_pred, f0_enc_norm.detach())
        loss = loss + 0.1 * f0_pred_loss

    opt.zero_grad()
    scl.scale(loss).backward()
    scl.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.generator_params(), 5.0)
    scl.step(opt)
    scl.update()

    if s % 300 == 0 or s == JT_STEPS - 1:
        print(f"JT step {s:04d} loss={loss.item():.3f} stft_ae={stft_ae.item():.2f} "
              f"stft_flow={stft_flow.item():.2f} cfm={cfm_loss.item():.3f} "
              f"f0_ref={f0_ref_loss.item():.4f} f0_pred={f0_pred_loss.item():.4f} [{time.time()-t0:.0f}s]")

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

# 阶梯 F0 (推理用)
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

# 推理: 原版 F0 (use_predicted_f0=False)
print("\n推理测试...")
with torch.no_grad(), torch.cuda.amp.autocast():
    w_orig, _, _, _, _, _ = model.forward_synthesis(
        ph, dur, f0_t, n_steps=8, note_lengths=nl, use_predicted_f0=False)
    w_pred, _, _, f0_pred_hz, _, _ = model.forward_synthesis(
        ph, dur, f0_t, n_steps=8, note_lengths=nl, use_predicted_f0=True)

for name, w in [('infer_orig_f0', w_orig), ('infer_pred_f0', w_pred)]:
    w = w.squeeze().cpu().numpy(); w = w / (np.abs(w).max() + 1e-8)
    sf.write(f'{OUT}/{name}.wav', w, sr)
    print(f"{name}: {len(w)/sr:.1f}s")

# 诊断: F0Predictor 输出 vs 输入
f0_pred_np = f0_pred_hz.squeeze().cpu().numpy()
f0_in_np = f0_t.squeeze().cpu().numpy()
f0_enc_np = f0_gt.squeeze().cpu().numpy()
print(f"\nF0 诊断:")
print(f"  输入阶梯 F0: mean={f0_in_np.mean():.1f} std={f0_in_np.std():.1f}")
print(f"  预测 F0:     mean={f0_pred_np.mean():.1f} std={f0_pred_np.std():.1f}")
print(f"  编码器 F0:   mean={f0_enc_np.mean():.1f} std={f0_enc_np.std():.1f}")

# 直接使用 encoder z1 解码 (诊断)
with torch.no_grad(), torch.cuda.amp.autocast():
    z1_direct, f0_e_d = model.encoder(audio, f0_override=f0_gt)
    z_q_direct, _, _ = model.rvq(z1_direct)
    fw_direct = model.ddsp_decoder(f0_e_d, z_q_direct, target_samples=T_audio)
w_direct = fw_direct.squeeze().cpu().numpy(); w_direct = w_direct / (np.abs(w_direct).max() + 1e-8)
sf.write(f'{OUT}/ae_direct.wav', w_direct, sr)
print(f"AE_direct: {len(w_direct)/sr:.1f}s")

print(f"\n完成: {OUT}/")
print(f"  orig.wav ae.wav ae_direct.wav infer_orig_f0.wav infer_pred_f0.wav")
