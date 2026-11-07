# YUAZ SGR
"""train/trainer"""
import os, sys, time, json, warnings, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from losses.stft_loss import MultiResolutionSTFTLoss
from losses.gan_loss import GANLoss
from models.conditioning import build_note_positions

def _frame_energy(audio, hop_length, target_frames):
    """帧级归一化能量 (B, target_frames) — 响度条件输入 (修复 #5) 用非重叠帧 RMS 估能量, 逐样本做 z-score + sigmoid 归一化到 (0,1)"""
    B, _, N = audio.shape
    n_frames = max(1, N // hop_length)
    x = audio[:, 0, :n_frames * hop_length].reshape(B, n_frames, hop_length)
    rms = x.float().pow(2).mean(dim=-1).clamp(min=1e-10).sqrt()
    e = torch.log(rms + 1e-6)
    e = (e - e.mean(dim=1, keepdim=True)) / (e.std(dim=1, keepdim=True) + 1e-5)
    e = torch.sigmoid(e)
    if n_frames != target_frames:
        e = F.interpolate(e.unsqueeze(1), size=target_frames, mode='linear').squeeze(1)
    return e

def _frame_activity(audio, hop_length, target_frames, db_below_peak=35.0):
    """帧级"有声活动度"目标 (1=有声, 0=静音/停顿) — 供静音门控监督 (修复 F) 以帧 RMS 相对本段峰值低 35 dB 为界: 低于它视为静音"""
    B, _, N = audio.shape
    n_frames = max(1, N // hop_length)
    x = audio[:, 0, :n_frames * hop_length].reshape(B, n_frames, hop_length)
    rms = x.float().pow(2).mean(dim=-1).clamp(min=1e-10).sqrt()
    rms_db = 20.0 * torch.log10(rms)
    peak = rms_db.max(dim=1, keepdim=True).values
    act = (rms_db > (peak - db_below_peak)).to(audio.dtype)
    if n_frames != target_frames:
        act = F.interpolate(act.unsqueeze(1), size=target_frames, mode='nearest').squeeze(1)
    return act

def _build_step_f0(f0_sq, note_lengths):
    """把平滑 F0 量化为音符级阶梯 F0 — 模拟推理时的 MIDI F0 条件"""
    B, T = f0_sq.shape
    n_notes = note_lengths.shape[1]
    dev = f0_sq.device
    ends = torch.cumsum(note_lengths.to(dev), dim=1)            # (B,n_notes)
    total = ends[:, -1:]
    pos = torch.arange(T, device=dev).view(1, T).expand(B, T)
    valid = (pos < total).to(f0_sq.dtype)
    note_idx = torch.searchsorted(ends.contiguous(), pos.contiguous(), right=True)
    note_idx = note_idx.clamp(max=n_notes - 1)
    sums = torch.zeros(B, n_notes, device=dev, dtype=f0_sq.dtype).scatter_add_(
        1, note_idx, f0_sq * valid)
    cnts = torch.zeros(B, n_notes, device=dev, dtype=f0_sq.dtype).scatter_add_(
        1, note_idx, valid)
    means = sums / cnts.clamp(min=1.0)
    return means.gather(1, note_idx) * valid

def _voicing_target(audio, f0, hop_length, sample_rate, n_frames,
                    mode='binary', ap_min=0.0):
    """修复 #8: 帧级清浊目标 (1=清/非周期, 0=浊) — 自相关周期度, 自监督无需人工标注 在每帧以该帧 F0 对应的时延 (含 ±2 采样抖动搜索, 抗颤音) 计算归一化自相关: 浊音(元音) 有强周期 → 相关高 → 目标≈0 清辅音(zh/ch/sh/s/f"""
    x = audio[:, 0].float()                            # (B,N)
    B, N = x.shape
    dev = x.device
    win = max(2 * hop_length, 512)
    half = win // 2
    frames = F.pad(x, (half, half)).unfold(1, win, hop_length)   # (B,T,win)
    T = min(frames.shape[1], n_frames)
    frames = frames[:, :T]
    frames = frames - frames.mean(dim=-1, keepdim=True)
    e = frames.pow(2).sum(dim=-1) + 1e-8               # (B,T)
    f = f0[:, :T].float().clamp(min=40.0)
    lag0 = (sample_rate / f).round().long()            # (B,T)
    ar = torch.arange(win, device=dev, dtype=torch.long).view(1, 1, win)
    best = torch.zeros(B, T, device=dev, dtype=torch.float32)
    for d in (-2, -1, 0, 1, 2):
        lag = (lag0 + d).clamp(min=1, max=win - 1)     # (B,T)
        idx = ar + lag.unsqueeze(-1)                   # (B,T,win)
        valid = (idx < win).float()
        corr = (frames * torch.gather(frames, 2, idx.clamp(max=win - 1)) * valid).sum(-1) / e
        best = torch.maximum(best, corr)
    p = best.clamp(0.0, 1.0)
    if mode == 'soft':
        # 连续非周期度 A = 1-p，下限 ap_min
        out = (1.0 - p).clamp(min=ap_min, max=1.0).unsqueeze(1)   # (B,1,T)
    else:
        out = ((0.30 - p) / 0.15).clamp(0.0, 1.0).unsqueeze(1)   # (B,1,T)
    if T < n_frames:
        out = F.pad(out, (0, n_frames - T), value=1.0)
    return out

def _hf_spec_loss(pred, tgt, sr=24000, n_fft=2048, hop=512, cutoff=4000.0):
    """高频(>4kHz)对数幅度谱 L1 log1p 对"完全没有高频"惩罚最重, 因此会把能量往高频推; 只改训练, 推理零成本"""
    win = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)
    p = torch.stft(pred.squeeze(1).float(), n_fft, hop, window=win,
                   return_complex=True, center=True).abs()
    t = torch.stft(tgt.squeeze(1).float(), n_fft, hop, window=win,
                   return_complex=True, center=True).abs()
    freqs = torch.linspace(0, sr / 2, p.shape[1], device=pred.device)
    m = freqs > cutoff
    return F.l1_loss(torch.log1p(p[:, m]), torch.log1p(t[:, m]))

def _frame_rms_db(audio, hop_length, target_frames):
    """帧级 RMS (dB) — (B, target_frames)"""
    B, _, N = audio.shape
    n_frames = max(1, N // hop_length)
    x = audio[:, 0, :n_frames * hop_length].float().reshape(B, n_frames, hop_length)
    rms = x.pow(2).mean(dim=-1).clamp(min=1e-10).sqrt()
    db = 20.0 * torch.log10(rms)
    if n_frames != target_frames:
        db = F.interpolate(db.unsqueeze(1), size=target_frames, mode='linear').squeeze(1)
    return db

def _unvoiced_active_mask(f0_sq, audio, hop_length, target_frames, db_below_peak=30.0):
    """清辅音帧 mask: F0=0 且该帧有能量 (排除静音/停顿) 训练数据里清辅音帧的 CREPE F0 被置信度门控置 0, 而静音帧同样为 0; 用帧能量把"有声的清辅音"与"静音"分开"""
    db = _frame_rms_db(audio, hop_length, target_frames)
    peak = db.max(dim=1, keepdim=True).values
    active = (db > (peak - db_below_peak)).float()
    T = min(f0_sq.shape[-1], target_frames)
    unvoiced = (f0_sq[:, :T] <= 1e-3).float()
    mask = torch.zeros_like(db)
    mask[:, :T] = unvoiced * active[:, :T]
    return mask

def _extract_note_lags(lags, note_lengths, T_enc):
    """将帧级 lags 提取为音符级: 每音符取第一帧的 lag 值"""
    dev = lags.device
    notes = note_lengths.to(dev)
    B, n_notes = notes.shape
    T_bound = min(T_enc, lags.shape[-1])
    ends = torch.cumsum(notes, dim=1)
    starts = ends - notes
    src = starts.clamp(min=0, max=T_bound - 1)
    lag_gt = torch.gather(lags[:, :T_bound].float(), 1, src)
    keep = ((notes > 0) & (starts < T_bound)).to(lag_gt.dtype)
    return lag_gt * keep

def _safe_prob(x):
    """把概率类输出限到 (1e-4, 1-1e-4)，NaN/Inf 也一并处理（BCE 要求输入在 [0,1]）"""
    p = torch.nan_to_num(x.float(), nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
    return p.clamp(1e-4, 1.0 - 1e-4)

def compute_total_loss(model, batch, stage, stft_loss_fn, gan_loss_fn, w, use_gan=True):
    """返回: total, losses, fake_wav, fake_wav_syn, d_fake_loss 潜变量由 IRR 迭代精炼得到 (K 步), 无对抗式速度场训练 use_gan=False 时跳过对抗/特征匹配项 (判别器降频, 纯训练组件)"""
    audio = batch['audio']
    B = audio.shape[0]
    target_samples = audio.shape[-1]

    f0_audio = batch.get('f0_audio', None)
    z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
    T_enc = z1.shape[-1]
    z_q, _, rvq_loss = model.rvq(z1)

    # 能量条件（可选）
    use_energy = getattr(model, 'use_energy', False)
    energy = _frame_energy(audio, model.encoder.hop_length, T_enc) if use_energy else None

    # 静音门控监督目标
    silence_w = float(w.get('silence', 0.0))
    act_target = _frame_activity(audio, model.encoder.hop_length, T_enc) if silence_w > 0 else None

    # 阶梯 F0 + 解码器门控条件
    f0_sq = f0_enc.squeeze(1)
    use_decoder_cond = getattr(model, 'use_decoder_cond', False)
    mu_step = None
    if stage == 'joint' or use_decoder_cond:
        f0_step = _build_step_f0(f0_sq, batch['note_lengths'])
        mu_step = model.cond_encoder(batch['phoneme_ids'], batch['durations'],
                                     f0_step, energy=energy)
    mu_dec = mu_step.detach() if (use_decoder_cond and mu_step is not None) else None
    if mu_dec is not None and mu_dec.shape[-1] != z_q.shape[-1]:
        mu_dec = F.interpolate(mu_dec, size=z_q.shape[-1], mode='linear')

    # 抗扰动解码器：给解码器 latent 加相对噪声
    robust_sigma = float(w.get('robust_latent_noise', 0.0))
    z_q_dec = z_q
    if robust_sigma > 0:
        z_q_dec = z_q + torch.randn_like(z_q) * (robust_sigma * z_q.detach().std())

    fake_wav, dec_aux = model.ddsp_decoder(f0_enc, z_q_dec, target_samples=target_samples,
                                           cond=mu_dec, return_aux=True)

    total = w['rvq'] * rvq_loss
    losses = {'rvq': rvq_loss.item()}
    d_loss_out = None

    mel_tgt = None
    if w.get('mel', 0.0) > 0:
        with autocast(enabled=False):
            mel_tgt = model.encoder._compute_logmel(audio.float()).detach()

    # AE 路径
    with autocast(enabled=False):
        stft_loss = stft_loss_fn(audio.float(), fake_wav.float())
    total = total + w['stft'] * stft_loss
    losses['stft'] = stft_loss.item()

    if mel_tgt is not None:
        with autocast(enabled=False):
            mel_pred_ae = model.encoder._compute_logmel(fake_wav.float())
            mel_loss = F.l1_loss(mel_pred_ae, mel_tgt)
            # mel_tdiff: 频谱随时间变化量匹配
            td_pred = mel_pred_ae[:, :, 1:] - mel_pred_ae[:, :, :-1]
            td_tgt = mel_tgt[:, :, 1:] - mel_tgt[:, :, :-1]
            mel_td_loss = F.l1_loss(td_pred, td_tgt)
        total = total + w['mel'] * mel_loss
        losses['mel'] = mel_loss.item()
        w_td = float(w.get('mel_tdiff', 0.0))
        if w_td > 0:
            total = total + w_td * mel_td_loss
            losses['mel_td'] = mel_td_loss.item()

    # 清浊监督
    uv_target = _voicing_target(audio, f0_sq, model.encoder.hop_length,
                                model.encoder.sample_rate, T_enc,
                                mode=str(w.get('ap_target', 'binary')),
                                ap_min=float(w.get('ap_min', 0.0)))
    with autocast(enabled=False):
        a_pred = _safe_prob(dec_aux['aperiodicity'])
        g_pred = _safe_prob(dec_aux['gate'])
        u_tgt = torch.nan_to_num(uv_target.float(), nan=0.0).clamp(0.0, 1.0)
        # ap_voiced_scale: 浊音帧加权
        ap_vs = float(w.get('ap_voiced_scale', 1.0))
        if ap_vs != 1.0:
            wt = 1.0 + (ap_vs - 1.0) * (1.0 - u_tgt)
            ap_loss = (F.binary_cross_entropy(a_pred, u_tgt, weight=wt)
                       + F.binary_cross_entropy(g_pred, 1.0 - u_tgt, weight=wt))
        else:
            ap_loss = (F.binary_cross_entropy(a_pred, u_tgt)
                       + F.binary_cross_entropy(g_pred, 1.0 - u_tgt))
    total = total + w.get('ap', 1.0) * ap_loss
    losses['ap'] = ap_loss.item()
    losses['total'] = total.item()

    hf_w = float(w.get('hf_spec', 0.0))
    hf_ae_w = float(w.get('hf_spec_ae', 0.0))
    if hf_ae_w > 0:
        with autocast(enabled=False):
            hf_ae = _hf_spec_loss(fake_wav, audio)
        total = total + hf_ae_w * hf_ae
        losses['hf_spec_ae'] = hf_ae.item()
        losses['total'] = total.item()

    cons_w = float(w.get('cons_energy', 0.0))
    cons_scale = float(w.get('cons_energy_scale', 2.0))
    if cons_w > 0:
        uv_mask = _unvoiced_active_mask(f0_sq, audio, model.encoder.hop_length, T_enc)
        with autocast(enabled=False):
            db_p = _frame_rms_db(fake_wav.float(), model.encoder.hop_length, T_enc)
            db_t = _frame_rms_db(audio.float(), model.encoder.hop_length, T_enc)
            wt = 1.0 + (cons_scale - 1.0) * uv_mask
            cons_loss = (wt * (db_p - db_t).abs() / 10.0).sum() / wt.sum().clamp(min=1.0)
        total = total + cons_w * cons_loss
        losses['cons'] = cons_loss.item()
        losses['total'] = total.item()

    # 静音门控监督（fp32）
    if act_target is not None and dec_aux.get('act') is not None:
        with autocast(enabled=False):
            act_a = _safe_prob(dec_aux['act'].squeeze(1))
            tgt_a = act_target[:, :act_a.shape[-1]].float().clamp(0.0, 1.0)
            sil_loss = F.binary_cross_entropy(act_a, tgt_a)
        total = total + silence_w * sil_loss
        losses['silence'] = sil_loss.item()
        losses['total'] = total.item()

    if use_gan and stage in ('gan', 'joint'):
        g_loss, fm_loss = gan_loss_fn.compute_g_loss(audio, fake_wav)
        total = total + w['gan'] * g_loss + w['fm'] * fm_loss
        losses.update({'gan': g_loss.item(), 'fm': fm_loss.item()})
        losses['total'] = total.item()

    if stage == 'joint':
        # μ（平滑 F0）；aug_f0_stepify 概率改用阶梯 F0
        aug_step = float(w.get('aug_f0_stepify', 0.0))
        f0_mu = f0_step if (aug_step > 0 and float(torch.rand(()).item()) < aug_step) else f0_sq
        mu = model.cond_encoder(batch['phoneme_ids'], batch['durations'],
                                f0_mu, energy=energy)

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

        # 生成 z: 迭代残差精炼 (IRR), z₀=μ, 固定 K 步
        T_z = min(z1.shape[-1], mu.shape[-1])
        z_pred = model.refiner(mu[:, :, :T_z], steps=int(w.get('irr_steps', 4)))
        # 潜变量直督：回归 encoder 的 z1
        z_l2_w = float(w.get('z_l2', 0.0))
        if z_l2_w > 0:
            z_l2 = F.mse_loss(z_pred, z1[:, :, :T_z].detach())
            if torch.isfinite(z_l2):
                total = total + z_l2_w * z_l2
            losses['z_l2'] = z_l2.item()
        z_q_pred, _, _ = model.rvq(z_pred)
        if robust_sigma > 0:
            z_q_pred = z_q_pred + torch.randn_like(z_q_pred) * (robust_sigma * z_q_pred.detach().std())
        f0_dec = f0_enc[:, :, :T_z]
        cond_dec = None if mu_dec is None else mu_dec[:, :, :T_z]
        fake_wav_syn, dec_aux_syn = model.ddsp_decoder(f0_dec, z_q_pred,
                                                         target_samples=target_samples,
                                                         cond=cond_dec, return_aux=True)
        with autocast(enabled=False):
            stft_syn_loss = stft_loss_fn(audio.float(), fake_wav_syn.float())
        if torch.isfinite(stft_syn_loss):
            total = total + w['stft'] * w.get('stft_syn_scale', w.get('stft_flow_scale', 1.0)) * stft_syn_loss
        losses['stft_syn'] = stft_syn_loss.item()

        if mel_tgt is not None:
            with autocast(enabled=False):
                mel_pred_fl = model.encoder._compute_logmel(fake_wav_syn.float())
                mel_syn_loss = F.l1_loss(mel_pred_fl, mel_tgt)
                td_pred_f = mel_pred_fl[:, :, 1:] - mel_pred_fl[:, :, :-1]
                td_tgt_f = mel_tgt[:, :, 1:] - mel_tgt[:, :, :-1]
                mel_td_syn_loss = F.l1_loss(td_pred_f, td_tgt_f)
            total = total + w['mel'] * mel_syn_loss
            losses['mel_syn'] = mel_syn_loss.item()
            if w_td > 0:
                total = total + w_td * mel_td_syn_loss
                losses['mel_td_syn'] = mel_td_syn_loss.item()

        if hf_w > 0:
            with autocast(enabled=False):
                hf_fl = _hf_spec_loss(fake_wav_syn, audio)
            if torch.isfinite(hf_fl):
                total = total + hf_w * hf_fl
            losses['hf_spec_syn'] = hf_fl.item()
        # 清浊监督（生成路径）
        uv_syn = F.interpolate(uv_target, size=f0_dec.shape[-1], mode='nearest')
        with autocast(enabled=False):
            a_syn = _safe_prob(dec_aux_syn['aperiodicity'])
            g_syn = _safe_prob(dec_aux_syn['gate'])
            u_syn = uv_syn.float().clamp(0.0, 1.0)
            ap_syn = (F.binary_cross_entropy(a_syn, u_syn)
                       + F.binary_cross_entropy(g_syn, 1.0 - u_syn))
        if torch.isfinite(ap_syn):
            total = total + w.get('ap', 1.0) * w.get('ap_syn_scale', w.get('ap_syn_scale', 1.0)) * ap_syn
        losses['ap_syn'] = ap_syn.item()

        # 静音门控监督（生成路径）
        if act_target is not None and dec_aux_syn.get('act') is not None:
            with autocast(enabled=False):
                act_f = _safe_prob(dec_aux_syn['act'].squeeze(1))
                tgt_f = act_target[:, :act_f.shape[-1]].float().clamp(0.0, 1.0)
                sil_syn = F.binary_cross_entropy(act_f, tgt_f)
            total = total + silence_w * sil_syn
            losses['silence'] = sil_syn.item()

        # 清辅音能量监督（生成路径）
        if cons_w > 0:
            cons_uv = _unvoiced_active_mask(f0_sq, audio, model.encoder.hop_length, T_enc)
            with autocast(enabled=False):
                db_pf = _frame_rms_db(fake_wav_syn.float(), model.encoder.hop_length, T_enc)
                wt_f = 1.0 + (cons_scale - 1.0) * cons_uv
                cons_syn = (wt_f * (db_pf - db_t).abs() / 10.0).sum() / wt_f.sum().clamp(min=1.0)
            if torch.isfinite(cons_syn):
                total = total + cons_w * cons_syn
            losses['cons_syn'] = cons_syn.item()

        # F0 预测器：μ(阶梯 F0) → 平滑 F0
        f0_enc_norm = f0_enc[:, :, :T_z] / 2000.0   # 归一化到 [0, ~1]
        if getattr(model, 'f0_version', 'v1') == 'v2':
            # v2: 额外输入阶梯 F0 + 位置
            note_pos, frame_pos = build_note_positions(
                batch['note_lengths'], T_z, z1.device)
            f0_pred = model.f0_predictor(
                mu_step[:, :, :T_z].detach(),
                f0_step=f0_step[:, :T_z].unsqueeze(1),
                note_pos=note_pos, frame_pos=frame_pos)
        else:
            f0_pred = model.f0_predictor(mu_step[:, :, :T_z].detach())
        if f0_pred.shape[-1] != f0_enc_norm.shape[-1]:
            f0_enc_norm = F.interpolate(f0_enc_norm, size=f0_pred.shape[-1], mode='linear')
        f0_tgt = f0_enc_norm.detach()
        if str(w.get('f0_loss', 'l1')).lower() == 'cents':
            # 对数域（音分）监督：低音区和高音区权重一致
            pred_hz = (f0_pred.float() * 2000.0).clamp(min=1.0)
            tgt_hz = (f0_tgt.float() * 2000.0).clamp(min=1.0)
            mask = (tgt_hz > 20.0).float()
            cents = (1200.0 * torch.log2(pred_hz / tgt_hz)).abs() * mask
            f0_pred_loss = cents.sum() / mask.sum().clamp(min=1.0) / 100.0
        else:
            f0_pred_loss = F.l1_loss(f0_pred, f0_tgt)
        total = total + w.get('f0_pred', 0.1) * f0_pred_loss
        losses['f0_pred'] = f0_pred_loss.item()

        # 帧间平滑
        f0_smooth = (f0_pred[:, :, 1:] - f0_pred[:, :, :-1]).abs().mean()
        total = total + w.get('f0_smooth', 0.05) * f0_smooth
        losses['f0_smooth'] = f0_smooth.item()

        losses['total'] = total.item()
        return total, losses, fake_wav, fake_wav_syn, None

    losses['total'] = total.item()
    return total, losses, fake_wav, fake_wav, d_loss_out

class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Adam (β=0.8/0.999)
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

        self.base_lr = lr
        self.current_lr = lr
        sched = (config.get('optimizer', {}) or {}).get('schedule', {}) or {}
        self.warmup_steps = int(sched.get('warmup_steps', 0))
        self.min_lr_ratio = float(sched.get('min_lr_ratio', 1.0))
        self.total_steps = int(config.get('total_steps', 0))

        gen_ids = {id(p) for p in model.generator_params()}
        self._ema = {n: p.detach().clone()
                     for n, p in model.named_parameters() if id(p) in gen_ids}
        self.ema_decay = float(config.get('ema_decay', 0.999))

    def lr_scale(self, step):
        if self.total_steps <= 0 or (self.warmup_steps <= 0 and self.min_lr_ratio >= 1.0):
            return 1.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return max(1e-8, (step + 1) / self.warmup_steps)
        prog = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos

    @torch.no_grad()
    def _ema_step(self):
        d = self.ema_decay
        for n, p in self.model.named_parameters():
            e = self._ema.get(n)
            if e is not None:
                e.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def _get_stage(self, step):
        if step <= self.ae_steps:
            return 'ae'
        elif step <= self.gan_steps:
            return 'gan'
        else:
            return 'joint'

    def train_step(self, batch, step):
        stage = self._get_stage(step)
        prec = str(self.config.get('precision', 'fp16')).lower()
        use_amp = prec in ('fp16', 'bf16')
        amp_dtype = torch.bfloat16 if prec == 'bf16' else torch.float16
        # 判别器每 gan_every 步更新
        gan_every = max(1, int(self.config.get('gan_every', 1)))
        use_gan = (step % gan_every == 0)

        self.current_lr = self.base_lr * self.lr_scale(step)
        for g in self.optim_g.param_groups:
            g['lr'] = self.current_lr
        for g in self.optim_d.param_groups:
            g['lr'] = self.current_lr

        with autocast(enabled=use_amp, dtype=amp_dtype):
            total_loss, losses, fake_wav, fake_wav_syn, d_loss = compute_total_loss(
                self.model, batch, stage, self.stft_loss_fn, self.gan_loss_fn, self.loss_weights,
                use_gan=use_gan
            )

        self.optim_g.zero_grad()
        self.optim_d.zero_grad()
        gen_loss = total_loss
        wav_ok = bool(torch.isfinite(fake_wav).all())
        if stage != 'ae':
            wav_ok = wav_ok and bool(torch.isfinite(fake_wav_syn).all())
        if (not torch.isfinite(gen_loss)) or (not wav_ok):
            self.optim_g.zero_grad()
            self.optim_d.zero_grad()
            self._nan_skips = getattr(self, '_nan_skips', 0) + 1
            if self._nan_skips % 20 == 1:
                print("nan-skip #%d @step %d stage %s" %
                      (self._nan_skips, step, stage), file=sys.stderr)
                nonfinite = sum(1 for p in self.model.parameters() if not torch.isfinite(p).all())
                if nonfinite:
                    print("!! WEIGHTS NONFINITE: %d/%d tensors — 回滚检查点" %
                          (nonfinite, sum(1 for _ in self.model.parameters())), file=sys.stderr)
            losses['total'] = float('nan')
            return losses, stage

        self.scaler.scale(gen_loss).backward()

        # 所有 backward 必须早于 unscale_
        d_ok = False
        if stage != 'ae' and use_gan:
            disc_wav = fake_wav_syn if stage == 'joint' else fake_wav
            d_loss = self.gan_loss_fn.compute_d_loss(batch['audio'], disc_wav)
            d_ok = bool(torch.isfinite(d_loss))
            if d_ok:
                self.scaler.scale(d_loss).backward()
            losses['d_loss'] = d_loss.item()

        self.scaler.unscale_(self.optim_g)
        torch.nn.utils.clip_grad_norm_(self.model.generator_params(), max_norm=5.0)
        if d_ok:
            self.scaler.unscale_(self.optim_d)
            torch.nn.utils.clip_grad_norm_(self.model.discriminator_params(), max_norm=5.0)

        self.scaler.step(self.optim_g)
        if d_ok:
            self.scaler.step(self.optim_d)
        self.scaler.update()
        self._ema_step()

        return losses, stage

    def validate(self, val_loader_or_iter, step):
        stage = self._get_stage(step)
        all_losses = {}
        n_batches = 0

        self.model.eval()
        with torch.no_grad():
            it = val_loader_or_iter() if callable(val_loader_or_iter) else val_loader_or_iter
            for batch in it:
                _, losses, _, _, _ = compute_total_loss(
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
        # 写入能力标志位
        flags = {
            'decoder_cond': bool(getattr(self.model, 'use_decoder_cond', False)),
            'use_energy': bool(getattr(self.model, 'use_energy', False)),
            'f0_predictor': str(getattr(self.model, 'f0_version', 'v1')),
            'encoder_feature': str(getattr(getattr(self.model, 'encoder', None), 'feature', 'mfcc')),
            'dec_mels': int(getattr(getattr(self.model, 'ddsp_decoder', None), 'n_mels', 128)),
            'dec_res_rank': int(getattr(getattr(self.model, 'ddsp_decoder', None), 'spectral_residual_rank', 0)),
            'generator': 'irr',
            'irr_steps': int(getattr(self.model, 'irr_steps', 0)),
        }
        torch.save({
            'step': step,
            'model': clean_sd,
            'ema': {k.replace('_orig_mod.', ''): v for k, v in self._ema.items()},
            'flags': flags,
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
            elif clean_k in model_sd and v.dim() >= 2 and v.shape[1:] == model_sd[clean_k].shape[1:]:
                # 部分加载：复制前 N 行
                n_base = v.shape[0]
                model_sd[clean_k][:n_base] = v
                loaded += 1
                print(f"  [恢复] 部分加载 {clean_k}: {v.shape} → {model_sd[clean_k].shape} (新增 {model_sd[clean_k].shape[0] - n_base} 行)")
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
        if 'ema' in ckpt:
            for k, v in ckpt['ema'].items():
                if k in self._ema and self._ema[k].shape == v.shape:
                    self._ema[k].copy_(v.to(self.device))
        return ckpt['step']