# YUAZ SGR
"""完整 SVS 模型"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .rvq import ResidualVectorQuantizer
from .ddsp_decoder import DDSPDecoder
from .iterative_refiner import IterativeRefiner
from .conditioning import (ConditioningEncoder, TemporalPredictor, F0Predictor,
                           F0PredictorV2, build_note_positions)
from .discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator

class SVSModel(nn.Module):
    def __init__(self, config):
        """config: dict with keys encoder, rvq, ddsp, generator, conditioning, temporal, discriminator"""
        super().__init__()

        # Encoder (DDSP z-encoder + F0 encoder)
        enc_cfg = config['encoder']
        self.encoder = Encoder(
            hidden_dim=enc_cfg['hidden_dim'],
            latent_dim=enc_cfg['latent_dim'],
            sample_rate=enc_cfg.get('sample_rate', 24000),
            n_mfcc=enc_cfg['n_mfcc'],
            n_fft=enc_cfg['n_fft'],
            hop_length=enc_cfg['hop_length'],
            feature=enc_cfg.get('feature', 'mfcc'),
            n_mels=enc_cfg.get('n_mels', 80),
        )

        # RVQ
        rvq_cfg = config['rvq']
        self.rvq = ResidualVectorQuantizer(
            num_quantizers=rvq_cfg['num_quantizers'],
            codebook_size=rvq_cfg['codebook_size'],
            codebook_dim=rvq_cfg['codebook_dim'],
            commitment_weight=rvq_cfg['commitment_weight'],
        )

        # DDSP Decoder
        ddsp_cfg = config['ddsp']
        enc_cfg = config['encoder']
        self.ddsp_decoder = DDSPDecoder(
            n_harmonics=ddsp_cfg['n_harmonics'],
            sample_rate=ddsp_cfg.get('sample_rate', 24000),
            fft_size=ddsp_cfg['fft_size'],
            hop_length=ddsp_cfg['hop_length'],
            hidden_dim=ddsp_cfg['hidden_dim'],
            encoder_hop_length=enc_cfg['hop_length'],
            emformer_num_layers=ddsp_cfg['emformer_num_layers'],
            emformer_num_heads=ddsp_cfg['emformer_num_heads'],
            emformer_ffn_dim=ddsp_cfg['emformer_ffn_dim'],
            emformer_segment_length=ddsp_cfg['emformer_segment_length'],
            emformer_left_context=ddsp_cfg['emformer_left_context'],
            emformer_right_context=ddsp_cfg['emformer_right_context'],
            voicing_harmonic=ddsp_cfg.get('voicing_harmonic', 0.15),
            voicing_aperiodic=ddsp_cfg.get('voicing_aperiodic', 0.60),
            n_mels=ddsp_cfg.get('n_mels', 128),
            spectral_residual_rank=ddsp_cfg.get('spectral_residual_rank', 0),
            harm_rolloff=ddsp_cfg.get('harm_rolloff', 'soft'),
            harm_amp=ddsp_cfg.get('harm_amp', 'fixed'),
            harm_synth=ddsp_cfg.get('harm_synth', 'stft'),
            mel_pinv_clamp=ddsp_cfg.get('mel_pinv_clamp', True),
            silence_gate=ddsp_cfg.get('silence_gate', False),
            env_unbounded=ddsp_cfg.get('env_unbounded', False),
            env_cond=ddsp_cfg.get('env_cond', False),
            noise_scale=ddsp_cfg.get('noise_scale', 1.0),
            env_softplus=ddsp_cfg.get('env_softplus', False),
            harm_amp_pred=ddsp_cfg.get('harm_amp_pred', False),
        )

        # 生成器：迭代残差精炼 (IRR)
        gen_cfg = config.get('generator', {})
        self.irr_steps = int(gen_cfg.get('irr_steps', 4))
        self.refiner = IterativeRefiner(
            cond_dim=int(gen_cfg.get('input_dim', 128)),
            hidden=int(gen_cfg.get('irr_hidden', 384)),
            n_layers=int(gen_cfg.get('irr_layers', 3)),
            max_steps=int(gen_cfg.get('irr_max_steps', 8)),
            multi_scale=bool(gen_cfg.get('irr_multiscale', False)),
        )

        # Conditioning Encoder
        cond_cfg = config['conditioning']
        self.cond_encoder = ConditioningEncoder(
            num_phonemes=cond_cfg['num_phonemes'],
            phoneme_dim=cond_cfg['phoneme_dim'],
            cond_dim=cond_cfg['cond_dim'],
            use_energy=bool(cond_cfg.get('use_energy', False)),
        )

        # Temporal Predictor
        temp_cfg = config['temporal']
        self.temporal_predictor = TemporalPredictor(
            cond_dim=temp_cfg['cond_dim'],
            num_phonemes=cond_cfg['num_phonemes'],
            phoneme_dim=cond_cfg.get('phoneme_dim', 256),
        )

        # 独立 F0 预测器
        # v1: 3×Conv1d；v2: 膨胀卷积 + 音符/位置条件
        self.f0_version = str(cond_cfg.get('f0_predictor', 'v1')).lower()
        if self.f0_version == 'v2':
            self.f0_predictor = F0PredictorV2(
                cond_dim=cond_cfg.get('cond_dim', 128),
                hidden_dim=cond_cfg.get('f0_hidden_dim', 192),
            )
        else:
            self.f0_predictor = F0Predictor(
                cond_dim=cond_cfg.get('cond_dim', 128),
                hidden_dim=128,
            )
        # μ 送入解码器 weight_net
        self.use_decoder_cond = bool(config.get('decoder_cond', False))
        self.use_energy = self.cond_encoder.use_energy

        # 判别器（推理端可关闭）
        self.build_discriminator = bool(config.get('build_discriminator', True))
        if self.build_discriminator:
            disc_cfg = config['discriminator']
            self.mpd = MultiPeriodDiscriminator(disc_cfg['mpd_periods'])
            self.msd = MultiScaleDiscriminator()

    def generator_params(self):
        """返回生成器参数 (Encoder + DDSP + RVQ + Refiner + Cond + Temporal + F0Pred)"""
        params = list(self.encoder.parameters()) + \
                 list(self.rvq.parameters()) + \
                 list(self.ddsp_decoder.parameters()) + \
                   list(self.cond_encoder.parameters()) + \
                 list(self.temporal_predictor.parameters()) + \
                 list(self.f0_predictor.parameters())
        params += list(self.refiner.parameters())
        return params

    def discriminator_params(self):
        """返回判别器参数"""
        if not self.build_discriminator:
            return []
        return list(self.mpd.parameters()) + list(self.msd.parameters())

    def freeze_f0_encoder(self):
        """冻结 CREPE F0 encoder (修复 #42) 修复 #42: CREPE 已在 _extract_f0 中用 torch"""
        pass

    def forward_ae(self, audio, f0_override=None):
        """自编码器前向: 波形 → Encoder → RVQ → DDSP → 波形"""
        z1, f0 = self.encoder(audio, f0_override=f0_override)
        z_q, indices, rvq_loss = self.rvq(z1)
        fake_wav = self.ddsp_decoder(f0, z_q)
        return z1, f0, z_q, fake_wav, rvq_loss

    @staticmethod
    def _smooth_f0(f0, k=5):
        """对 F0 做时间平滑（仅在浊音帧上），供条件 μ 使用；不改音高。"""
        x = f0 if f0.dim() == 3 else f0.unsqueeze(1)
        voiced = (x > 0).to(x.dtype)
        num = F.avg_pool1d((x * voiced).float(), k, 1, k // 2)
        den = F.avg_pool1d(voiced.float(), k, 1, k // 2)
        out = (num / den.clamp(min=1e-3)).to(x.dtype) * voiced
        return out

    def forward_synthesis(self, phoneme_ids, durations, f0, n_steps=1,
                           note_lengths=None, predict_durations=False,
                           use_predicted_f0=True, energy=None,
                           rvq_residual_alpha=0.0,
                           mu_refine=False, mu_f0=None,
                           max_chunk_frames=None, chunk_overlap_frames=96,
                           voicing=None, voicing_harmonic=0.15,
                           voicing_aperiodic=0.60):
        """推理前向: μ → IRR 精炼 → RVQ → DDSP 波形"""
        if predict_durations and note_lengths is not None:
            mu_init = self.cond_encoder(phoneme_ids, durations, self._smooth_f0(f0), energy=energy)
            if f0.dim() == 3:
                f0_1d = f0.squeeze(1)
            else:
                f0_1d = f0
            dur_pred = self.temporal_predictor.predict_durations(mu_init, f0_1d, note_lengths, phoneme_ids, durations=durations)
            durations = dur_pred.long()
            durations = torch.clamp(durations, min=1)

        # μ 用平滑 F0（训练分布），音高仍由 f0 决定
        mu = self.cond_encoder(phoneme_ids, durations, self._smooth_f0(f0), energy=energy)
        if note_lengths is None:
            note_lengths = durations
        lag_pred, temp_dur = self.temporal_predictor(
            mu, f0, durations, note_lengths,
            phoneme_ids=phoneme_ids,
            phoneme_emb=self.cond_encoder.phoneme_embed(phoneme_ids)
        )

        # F0 选择: 原版 f0 vs F0Predictor 精炼 F0
        f0_in = f0 if f0.dim() == 3 else f0.unsqueeze(1)
        if use_predicted_f0:
            if self.f0_version == 'v2':
                # v2: 额外输入阶梯 F0 + 音符内/全局位置
                note_pos, frame_pos = build_note_positions(note_lengths, mu.shape[-1], mu.device)
                f0_pred = self.f0_predictor(mu, f0_step=f0_in,
                                            note_pos=note_pos, frame_pos=frame_pos)
            else:
                f0_pred = self.f0_predictor(mu)  # (B, 1, T_mu), 输出归一化 [0,1]
            f0_hz = f0_pred * 2000.0             # 反归一化回 Hz
        else:
            # 直接使用输入 f0
            f0_hz = f0_in

        # 保留 0（静音/清音），谐波由解码器 voiced 门控关闭
        f0_clamp = torch.clamp(f0_hz, min=0.0, max=2000.0)

        # 可选 μ 精修：用平滑 F0 重建 μ；不改 f0_clamp
        mu_cond = mu
        mu_src = mu_f0 if mu_f0 is not None else (f0_clamp if mu_refine else None)
        if mu_src is not None:
            ms = mu_src if mu_src.dim() == 3 else mu_src.unsqueeze(1)
            if ms.shape[-1] != mu.shape[-1]:
                ms = F.interpolate(ms.float(), size=mu.shape[-1], mode='linear').to(mu.dtype)
            mu_cond = self.cond_encoder(phoneme_ids, durations, ms, energy=energy)

        # 生成 z: 迭代残差精炼 (K 步)
        z = self.refiner(mu_cond, steps=self.irr_steps)
        z_q, _, _ = self.rvq(z)
        # 软量化：回补连续残差
        if rvq_residual_alpha > 0:
            z_q = z_q + rvq_residual_alpha * (z - z_q)

        if f0_clamp.shape[-1] != z_q.shape[-1]:
            f0_clamp = F.interpolate(f0_clamp, size=z_q.shape[-1], mode='linear')

        cond = mu if self.use_decoder_cond else None
        if cond is not None and cond.shape[-1] != z_q.shape[-1]:
            cond = F.interpolate(cond, size=z_q.shape[-1], mode='linear')

        waveform = self._decode(f0_clamp, z_q, cond=cond,
                                max_chunk_frames=max_chunk_frames,
                                chunk_overlap_frames=chunk_overlap_frames,
                                voicing=voicing,
                                voicing_harmonic=voicing_harmonic,
                                voicing_aperiodic=voicing_aperiodic)
        return waveform, mu_cond, z, f0_clamp, lag_pred, temp_dur

    def _decode(self, f0, z, cond=None, max_chunk_frames=None,
                chunk_overlap_frames=96, voicing=None,
                voicing_harmonic=0.15, voicing_aperiodic=0.60):
        """DDSP 解码; T 帧超过 max_chunk_frames 时分块 + 交叉淡化, 控制长曲内存峰值 分块仅在长音频时启用 (短音频走原路径, 输出完全一致)"""
        T = f0.shape[-1]

        def _dec(f0i, zi, condi, voi):
            return self.ddsp_decoder(f0i, zi, cond=condi, voicing=voi,
                                     voicing_harmonic=voicing_harmonic,
                                     voicing_aperiodic=voicing_aperiodic)

        if voicing is not None:
            # 归一化为 (B, 1, T)
            if voicing.dim() == 1:
                voicing = voicing.view(1, 1, -1)
            elif voicing.dim() == 2:
                voicing = voicing.unsqueeze(1)
            if voicing.shape[-1] != T:
                voicing = F.interpolate(voicing.float(), size=T, mode='nearest')

        if max_chunk_frames is None:
            return _dec(f0, z, cond, voicing)

        hop = self.ddsp_decoder.encoder_hop_length
        fft = self.ddsp_decoder.fft_size
        min_frames = fft // hop + 2          # 保证单块样本数 > fft_size
        max_chunk_frames = max(int(max_chunk_frames), min_frames)
        if T <= max_chunk_frames:
            return _dec(f0, z, cond, voicing)

        overlap = int(min(chunk_overlap_frames, max(1, max_chunk_frames // 4)))
        step = max(1, max_chunk_frames - overlap)
        n_total = T * hop
        out = torch.zeros(1, 1, n_total, device=f0.device, dtype=f0.dtype)
        wsum = torch.zeros(1, 1, n_total, device=f0.device, dtype=f0.dtype)
        fade = max(1, overlap * hop)

        start = 0
        while start < T:
            end = min(start + max_chunk_frames, T)
            if end < T and (T - end) <= overlap:
                end = T                       # 吸收过短尾块
            wav = _dec(
                f0[:, :, start:end], z[:, :, start:end],
                None if cond is None else cond[:, :, start:end],
                None if voicing is None else voicing[:, :, start:end],
            )
            s0, s1 = start * hop, end * hop
            w = torch.ones_like(wav)
            n_f = min(fade, wav.shape[-1] // 2) if wav.shape[-1] > 1 else 1
            if start > 0:
                w[..., :n_f] = torch.linspace(0.0, 1.0, n_f, device=wav.device, dtype=wav.dtype)
            if end < T:
                w[..., -n_f:] = torch.linspace(1.0, 0.0, n_f, device=wav.device, dtype=wav.dtype)
            out[..., s0:s1] += wav * w
            wsum[..., s0:s1] += w
            if end >= T:
                break
            start += step

        return out / wsum.clamp(min=1e-6)