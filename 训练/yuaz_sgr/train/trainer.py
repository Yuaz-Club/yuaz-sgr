# YUAZ SGR
"""
train/trainer.py
论文 §4.2: 三阶段训练器

三阶段:
  阶段 1 (0~20k):     AE 预训练 — L_STFT + L_RVQ
  阶段 2 (20k~200k):  引入判别器 — + L_GAN + L_FM
  阶段 3 (200k~250k): 全联合 — + L_feature + L_lag + L_dur, CFM 独立 backward

论文 L_total 严格 7 项 (CFM 不在 L_total 内, 独立 backward)
"""
import os, sys, time, json, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from losses.stft_loss import MultiResolutionSTFTLoss
from losses.gan_loss import GANLoss


def _extract_note_lags(lags, note_lengths, T_enc):
    """将帧级 lags 提取为音符级: 每音符取第一帧的 lag 值"""
    B = lags.shape[0]
    device = lags.device
    lag_gt = torch.zeros(B, note_lengths.shape[1], device=device)
    T_lags = lags.shape[-1]
    T_bound = min(T_enc, T_lags)
    for b in range(B):
        pos = 0
        for n_idx in range(note_lengths.shape[1]):
            n_len = int(note_lengths[b, n_idx].item())
            if n_len <= 0:
                continue
            if pos >= T_bound:
                break
            src_idx = min(pos, T_bound - 1)
            lag_gt[b, n_idx] = lags[b, src_idx]
            pos += n_len
    return lag_gt


def compute_total_loss(model, batch, stage, stft_loss_fn, gan_loss_fn, w):
    """
    返回: total, losses, fake_wav, fake_wav_flow, d_fake_loss, cfm_loss
    CFM loss 不加入 total, 独立 backward (论文 §4.2 L_total 不含 CFM)
    """
    audio = batch['audio']
    B = audio.shape[0]
    target_samples = audio.shape[-1]

    f0_audio = batch.get('f0_audio', None)
    z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
    T_enc = z1.shape[-1]
    z_q, _, rvq_loss = model.rvq(z1)

    # [已注释] 提前计算 mu，供 DDSP decoder 的 weight_net 区分辅音/元音
    # 原因: overfit_v2.py 验证不加 cond 咬字更清晰 — cond 会干扰 weight_net 的谐波/噪声门控
    # mu_for_decoder = None
    # if stage == 'joint':
    #     f0_sq = f0_enc.squeeze(1)
    #     mu_for_decoder = model.cond_encoder(batch['phoneme_ids'], batch['durations'], f0_sq)

    fake_wav = model.ddsp_decoder(f0_enc, z_q, target_samples=target_samples)
    # [已注释] cond=mu_for_decoder.detach() if mu_for_decoder is not None else None)

    total = w['rvq'] * rvq_loss
    losses = {'rvq': rvq_loss.item()}
    d_loss_out = None
    cfm_loss_out = None

    # AE路径: 所有阶段都算 STFT loss (修复 Joint 阶段解码器退化)
    stft_loss = stft_loss_fn(audio, fake_wav)
    total = total + w['stft'] * stft_loss
    losses['stft'] = stft_loss.item()
    losses['total'] = total.item()

    if stage in ('gan', 'joint'):
        g_loss, fm_loss = gan_loss_fn.compute_g_loss(audio, fake_wav)
        total = total + w['gan'] * g_loss + w['fm'] * fm_loss
        losses.update({'gan': g_loss.item(), 'fm': fm_loss.item()})
        losses['total'] = total.item()

    if stage == 'joint':
        f0_sq = f0_enc.squeeze(1)
        mu = model.cond_encoder(batch['phoneme_ids'], batch['durations'], f0_sq)

    if stage == 'joint':
        T_feat = min(mu.shape[-1], z1.shape[-1])
        feature_loss = w['feature'] * F.mse_loss(
            mu[:, :, :T_feat], z1[:, :, :T_feat].detach()
        )
        total = total + feature_loss
        losses['feature'] = feature_loss.item()

        lag_pred, dur_pred = model.temporal_predictor(
            mu.detach(), f0_sq, batch['durations'], batch['note_lengths'],
            phoneme_ids=batch['phoneme_ids'],
            phoneme_emb=model.cond_encoder.phoneme_embed(batch['phoneme_ids']).detach(),
        )

        lag_gt = _extract_note_lags(batch['lags'], batch['note_lengths'], T_enc)
        dur_gt = batch['durations'].float().to(lag_gt.device)

        lag_loss, dur_loss = model.temporal_predictor.compute_loss(
            lag_pred, dur_pred,
            lag_gt, dur_gt,
            batch['note_lengths'],
        )
        total = total + w['lag'] * lag_loss + w['dur'] * dur_loss
        losses.update({'lag': lag_loss.item(), 'dur': dur_loss.item()})

        # CFM: 训练 flow generator
        # 论文 §4.2: Full joint optimization of encoder, decoder, flow predictor, discriminators
        # 论文 L_total 不含 CFM, CFM 独立 backward
        z0 = torch.randn_like(z1)
        T_cfm = min(z1.shape[-1], mu.shape[-1])
        t_cfm = torch.rand(B, device=z1.device)
        z_t = (1 - t_cfm[:, None, None]) * z0[:, :, :T_cfm] + t_cfm[:, None, None] * z1[:, :, :T_cfm].detach()
        gen_out = model.flow_gen(z_t, t_cfm, mu[:, :, :T_cfm])
        v_pred = gen_out[:, :128, :]
        v_target = z1[:, :, :T_cfm].detach() - z0[:, :, :T_cfm]
        # 论文 Eq(18): L_CFM = E[1/(1-t) · ‖(z₁-z₀) - v_pred‖²]
        weight_cfm = 1.0 / (1.0 - t_cfm + 1e-6)
        cfm_loss_out = (weight_cfm[:, None, None] * ((v_pred - v_target) ** 2)).mean()
        losses['cfm'] = cfm_loss_out.item()

        # F0_refined 监督: flow 第129通道 MSE 对齐 encoder F0
        # 论文 §4.3.2: "one additional channel represents the fundamental frequency F0"
        # 归一化到 440Hz 范围 (A4), 与 overfit_v2.py 一致
        f0_refined = gen_out[:, 128:129, :] / 440.0
        f0_enc_cfm = f0_enc[:, :, :T_cfm] / 440.0
        if f0_enc_cfm.shape[-1] != f0_refined.shape[-1]:
            f0_enc_cfm = F.interpolate(f0_enc_cfm, size=f0_refined.shape[-1], mode='linear')
        f0_ref_loss = 0.01 * F.mse_loss(f0_refined, f0_enc_cfm.detach())
        total = total + f0_ref_loss
        losses['f0_refined'] = f0_ref_loss.item()

        # 关键修复: Flow-through-Decoder 路径
        # 论文 joint 阶段 flow 输出须经 decoder 参与 STFT loss,
        # 给 flow 网络来自解码器的梯度, 确保预测的 z 能被 decoder 正确解码
        z_pred = z0[:, :, :T_cfm] + v_pred  # z1 ≈ z0 + v_pred
        z_q_pred, _, _ = model.rvq(z_pred)
        f0_cfm = f0_enc[:, :, :T_cfm]
        fake_wav_flow = model.ddsp_decoder(f0_cfm, z_q_pred, target_samples=target_samples)
        stft_flow_loss = stft_loss_fn(audio, fake_wav_flow)
        # 用较高权重, 强制 decoder 适应 flow z 分布 (否则推理咬字偏浅)
        total = total + w['stft'] * 0.7 * stft_flow_loss
        losses['stft_flow'] = stft_flow_loss.item()

        # F0 预测器: 精炼 F0 任务 (L1 loss, 权重 0.1)
        # 独立于 flow matching, 不共享网络容量
        # 关键修复: μ 必须用阶梯 F0 计算 (模拟推理条件), 目标保持 CREPE 平滑 F0
        #  训练: μ(阶梯 F0) → F0Predictor → 预测平滑 CREPE F0
        #  推理: μ(MIDI 阶梯 F0) → F0Predictor → 精炼 F0
        # 之前用平滑 F0 算 μ, F0Predictor 只学到"从 μ 读出 F0"的身份映射
        note_lens = batch['note_lengths']
        f0_step = torch.zeros_like(f0_sq)
        for b in range(B):
            pos = 0
            for n in range(note_lens.shape[1]):
                nl = int(note_lens[b, n].item())
                if nl <= 0:
                    continue
                end = min(pos + nl, f0_sq.shape[-1])
                if end > pos:
                    f0_step[b, pos:end] = f0_sq[b, pos:end].mean()
                pos = end
        mu_f0 = model.cond_encoder(batch['phoneme_ids'], batch['durations'], f0_step)
        f0_pred = model.f0_predictor(mu_f0[:, :, :T_cfm].detach())
        f0_enc_cfm = f0_enc[:, :, :T_cfm] / 2000.0      # 归一化到 [0, ~1]
        if f0_pred.shape[-1] != f0_enc_cfm.shape[-1]:
            f0_enc_cfm = F.interpolate(f0_enc_cfm, size=f0_pred.shape[-1], mode='linear')
        f0_pred_loss = F.l1_loss(f0_pred, f0_enc_cfm.detach())
        total = total + 0.1 * f0_pred_loss
        losses['f0_pred'] = f0_pred_loss.item()

        losses['total'] = total.item()
        return total, losses, fake_wav, fake_wav_flow, None, cfm_loss_out

    losses['total'] = total.item()
    return total, losses, fake_wav, fake_wav, d_loss_out, None


# === 已合并到 compute_total_loss joint 阶段, 此函数不再使用 ===



class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # 优化器 (论文: Adam, lr=1e-5, β₁=0.8, β₂=0.999)
        lr = config['optimizer']['lr']
        self.optim_g = torch.optim.Adam(
            model.generator_params(),
            lr=lr,
            betas=config['optimizer']['betas'],
        )
        self.optim_d = torch.optim.Adam(
            model.discriminator_params(),
            lr=lr,
            betas=config['optimizer']['betas'],
        )

        self.scaler = GradScaler(enabled=(config['precision'] == 'fp16'))

        self.stft_loss_fn = MultiResolutionSTFTLoss()
        self.gan_loss_fn = GANLoss(model.mpd, model.msd)

        self.ae_steps = config['stages']['ae']['end']
        self.gan_steps = config['stages']['gan']['end']
        self.loss_weights = config['loss_weights']

        self.log_interval = config.get('log_interval', 100)
        self.save_interval = config.get('save_interval', 5000)

    def _get_stage(self, step):
        if step <= self.ae_steps:
            return 'ae'
        elif step <= self.gan_steps:
            return 'gan'
        else:
            return 'joint'

    def train_step(self, batch, step):
        stage = self._get_stage(step)
        use_amp = self.config['precision'] == 'fp16'

        with autocast(enabled=use_amp):
            total_loss, losses, fake_wav, fake_wav_flow, d_loss, cfm_loss = compute_total_loss(
                self.model, batch, stage, self.stft_loss_fn, self.gan_loss_fn, self.loss_weights
            )

        self.optim_g.zero_grad()
        need_retain = (stage == 'joint')
        self.scaler.scale(total_loss).backward(retain_graph=need_retain)
        if cfm_loss is not None:
            self.scaler.scale(cfm_loss).backward()

        # 梯度裁剪: 防止大 batch 发散
        self.scaler.unscale_(self.optim_g)
        torch.nn.utils.clip_grad_norm_(self.model.generator_params(), max_norm=5.0)

        if stage != 'ae':
            self.optim_d.zero_grad()
            disc_wav = fake_wav_flow if stage == 'joint' else fake_wav
            d_loss = self.gan_loss_fn.compute_d_loss(batch['audio'], disc_wav)
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.optim_d)
            torch.nn.utils.clip_grad_norm_(self.model.discriminator_params(), max_norm=5.0)
            losses['d_loss'] = d_loss.item()

        self.scaler.step(self.optim_g)
        if stage != 'ae':
            self.scaler.step(self.optim_d)
        self.scaler.update()

        return losses, stage

    def validate(self, val_loader_or_iter, step):
        stage = self._get_stage(step)
        all_losses = {}
        n_batches = 0

        self.model.eval()
        with torch.no_grad():
            it = val_loader_or_iter() if callable(val_loader_or_iter) else val_loader_or_iter
            for batch in it:
                _, losses, _, _, _, _ = compute_total_loss(
                    self.model, batch, stage, self.stft_loss_fn, self.gan_loss_fn, self.loss_weights
                )
                for k, v in losses.items():
                    all_losses[k] = all_losses.get(k, 0) + v
                n_batches += 1
                if n_batches >= 10:
                    break

        self.model.train()
        torch.cuda.empty_cache()
        return {k: v / n_batches for k, v in all_losses.items()}

    def save_checkpoint(self, step, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f'model_{step:06d}.pt')
        raw_sd = self.model.state_dict()
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
        torch.save({
            'step': step,
            'model': clean_sd,
            'optim_g': self.optim_g.state_dict(),
            'optim_d': self.optim_d.state_dict(),
            'scaler': self.scaler.state_dict(),
        }, path)
        return path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        model_sd = self.model.state_dict()
        loaded = 0
        skipped = 0
        for k, v in ckpt['model'].items():
            clean_k = k.replace('_orig_mod.', '')
            if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
                model_sd[clean_k] = v
                loaded += 1
            else:
                skipped += 1
        self.model.load_state_dict(model_sd, strict=False)
        if skipped > 0:
            print(f"  [恢复] 跳过 {skipped} 个不匹配参数 (新层随机初始化)")
            arch_changed = True
        else:
            arch_changed = False
        if arch_changed:
            skip_optim_d = True
            skip_optim_g = True
        else:
            skip_optim_d = False
            skip_optim_g = False
        if 'optim_g' in ckpt and not skip_optim_g:
            try:
                self.optim_g.load_state_dict(ckpt['optim_g'])
            except Exception:
                print("  [恢复] 生成器优化器跳过 (架构变更)")
        if 'optim_d' in ckpt and not skip_optim_d:
            try:
                self.optim_d.load_state_dict(ckpt['optim_d'])
            except Exception:
                print("  [恢复] 判别器优化器跳过 (架构变更)")
        if 'scaler' in ckpt:
            try:
                self.scaler.load_state_dict(ckpt['scaler'])
            except Exception:
                print("  [恢复] Scaler 跳过")
        return ckpt['step']