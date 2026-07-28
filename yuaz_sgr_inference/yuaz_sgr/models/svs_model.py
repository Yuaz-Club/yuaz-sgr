# YUAZ SGR
"""
models/svs_model.py
完整 SVS 模型 — 组合所有子模块

论文架构:
  Encoder (DDSP z-encoder + F0 encoder)
  → RVQ
  → DDSP Decoder (直接输出波形)

  Conditioning Encoder (音素 + F0 → μ)
  → Flow Matching Generator (z_t, t, μ → v_t, F0_refined)
  → DDSP Decoder (推理路径)

  TemporalPredictor (lag + duration)
  MPD + MSD (判别器)
"""
import torch
import torch.nn as nn
from .encoder import Encoder
from .rvq import ResidualVectorQuantizer
from .ddsp_decoder import DDSPDecoder
from .flow_matching import FlowMatchingGenerator
from .conditioning import ConditioningEncoder, TemporalPredictor, F0Predictor
from .discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator


class SVSModel(nn.Module):
    def __init__(self, config):
        """
        config: dict with keys encoder, rvq, ddsp, flow_matching, conditioning,
                temporal, discriminator
        """
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
        )

        # Flow Matching Generator
        fm_cfg = config['flow_matching']
        self.flow_gen = FlowMatchingGenerator(
            input_dim=fm_cfg['input_dim'],
            hidden_dim=fm_cfg['hidden_dim'],
            num_blocks=fm_cfg['num_blocks'],
            time_dim=fm_cfg['time_dim'],
            output_dim=fm_cfg['output_dim'],
        )

        # Conditioning Encoder
        cond_cfg = config['conditioning']
        self.cond_encoder = ConditioningEncoder(
            num_phonemes=cond_cfg['num_phonemes'],
            phoneme_dim=cond_cfg['phoneme_dim'],
            cond_dim=cond_cfg['cond_dim'],
        )

        # Temporal Predictor
        temp_cfg = config['temporal']
        self.temporal_predictor = TemporalPredictor(
            cond_dim=temp_cfg['cond_dim'],
            num_phonemes=cond_cfg['num_phonemes'],
            phoneme_dim=cond_cfg.get('phoneme_dim', 256),
        )

        # 独立 F0 预测器 (修复: 替代 flow matching 第 129 通道)
        self.f0_predictor = F0Predictor(
            cond_dim=cond_cfg.get('cond_dim', 128),
            hidden_dim=128,
        )

        # Discriminators
        disc_cfg = config['discriminator']
        self.mpd = MultiPeriodDiscriminator(disc_cfg['mpd_periods'])
        self.msd = MultiScaleDiscriminator()

    def generator_params(self):
        """返回生成器参数 (Encoder + DDSP + RVQ + Flow Gen + Cond + Temporal + F0Pred)"""
        return list(self.encoder.parameters()) + \
               list(self.rvq.parameters()) + \
               list(self.ddsp_decoder.parameters()) + \
               list(self.flow_gen.parameters()) + \
               list(self.cond_encoder.parameters()) + \
               list(self.temporal_predictor.parameters()) + \
               list(self.f0_predictor.parameters())

    def discriminator_params(self):
        """返回判别器参数"""
        return list(self.mpd.parameters()) + list(self.msd.parameters())

    def freeze_f0_encoder(self):
        """
        冻结 CREPE F0 encoder (修复 #42)
        修复 #42: CREPE 已在 _extract_f0 中用 torch.no_grad() 包裹,
        参数不会被更新。此处仅做显式标注, 移除崩溃的 self.encoder.f0_encoder 引用。
        """
        # CREPE 参数在 _extract_f0 中已被 torch.no_grad() 保护
        pass

    def forward_ae(self, audio, f0_override=None):
        """自编码器前向: 波形 → Encoder → RVQ → DDSP → 波形"""
        z1, f0 = self.encoder(audio, f0_override=f0_override)
        z_q, indices, rvq_loss = self.rvq(z1)
        fake_wav = self.ddsp_decoder(f0, z_q)
        return z1, f0, z_q, fake_wav, rvq_loss

    def forward_synthesis(self, phoneme_ids, durations, f0, n_steps=1,
                           note_lengths=None, predict_durations=False,
                           use_predicted_f0=True):
        """推理前向: 音素+F0 → μ → Flow Matching → RVQ(量化) → DDSP → 波形

        F0Predictor 从 μ 独立预测 F0, 替代旧 flow matching 第129通道。

        use_predicted_f0=True:  DDSP 解码器使用 F0Predictor 输出的精炼 F0
        use_predicted_f0=False: DDSP 解码器使用输入 f0 (原版行为, 用于对比测试)
        """
        if predict_durations and note_lengths is not None:
            mu_init = self.cond_encoder(phoneme_ids, durations, f0)
            if f0.dim() == 3:
                f0_1d = f0.squeeze(1)
            else:
                f0_1d = f0
            dur_pred = self.temporal_predictor.predict_durations(mu_init, f0_1d, note_lengths, phoneme_ids, durations=durations)
            durations = dur_pred.long()
            durations = torch.clamp(durations, min=1)

        mu = self.cond_encoder(phoneme_ids, durations, f0)
        if note_lengths is None:
            note_lengths = durations
        lag_pred, temp_dur = self.temporal_predictor(
            mu, f0, durations, note_lengths,
            phoneme_ids=phoneme_ids,
            phoneme_emb=self.cond_encoder.phoneme_embed(phoneme_ids)
        )

        # Flow matching → z (只有潜变量, 无 F0 通道)
        z = self.flow_gen.sample(mu, n_steps=n_steps)
        if isinstance(z, tuple):
            z = z[0]  # 兼容旧版返回 (z, f0_refined)
        z_q, _, _ = self.rvq(z)

        # F0 选择: 原版 f0 vs F0Predictor 精炼 F0
        if use_predicted_f0:
            f0_pred = self.f0_predictor(mu)  # (B, 1, T_mu), 输出归一化 [0,1]
            f0_hz = f0_pred * 2000.0         # 反归一化回 Hz
        else:
            # 原版: 直接使用输入的 f0 (可能是 MIDI 转换的阶梯 F0)
            if f0.dim() == 3:
                f0_hz = f0
            else:
                f0_hz = f0.unsqueeze(1)
        f0_clamp = torch.clamp(f0_hz, min=50.0, max=2000.0)

        if f0_clamp.shape[-1] != z_q.shape[-1]:
            f0_clamp = torch.nn.functional.interpolate(f0_clamp, size=z_q.shape[-1], mode='linear')
        waveform = self.ddsp_decoder(f0_clamp, z_q)
        return waveform, mu, z, f0_clamp, lag_pred, temp_dur