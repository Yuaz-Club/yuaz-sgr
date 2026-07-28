# YUAZ SGR
"""
models/ddsp_decoder.py
论文 §3.1.3: DDSP Decoder = 声码器, 直接输出 24kHz 波形

论文 Eq (6):  xh = F⁻¹((1-A)·S · F(eh))  谐波路径用 (1-A) 衰减
论文 Eq (8):  xn = F⁻¹(A·S · F(en))     噪声路径用 A 加权
论文: "STFT with window length of 2048 samples and 75% overlap"
  → fft_size=2048, hop_length=512, freq_bins=1025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa


class DDSPDecoder(nn.Module):
    def __init__(self, n_harmonics=60, sample_rate=24000,
                 fft_size=2048, hop_length=512, hidden_dim=128,
                 encoder_hop_length=256,
                 emformer_num_layers=4, emformer_num_heads=8,
                 emformer_ffn_dim=512, emformer_segment_length=32,
                 emformer_left_context=8, emformer_right_context=4):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.encoder_hop_length = encoder_hop_length

        self.emformer_input_proj = nn.Linear(129, hidden_dim)
        self.emformer = _Emformer(
            input_dim=hidden_dim,
            num_heads=emformer_num_heads,
            ffn_dim=emformer_ffn_dim,
            num_layers=emformer_num_layers,
            segment_length=emformer_segment_length,
            left_context_length=emformer_left_context,
            right_context_length=emformer_right_context,
        )
        self.emformer_proj = nn.Linear(hidden_dim, hidden_dim)

        self.env_net = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
        )
        # env_net 偏置: 随机初始化各 mel 通道不同值, 产生非平坦频谱包络
        # mean=-2.0 控制输出振幅 (比 -3.0 强 10x, 咬字更清晰), std=0.5 给各通道差异性
        nn.init.normal_(self.env_net[-2].bias, mean=-2.0, std=0.5)
        self.ap_net = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 16), nn.Sigmoid()
        )
        # 论文 §3.1.3: A ∈ [0,1] 是非周期性系数
        # sigmoid(-3)≈0.05: 初始化偏谐波, 防止训练中 A 饱和到 1.0 杀死 (1-A)·S 谐波路径
        nn.init.constant_(self.ap_net[-2].bias, -3.0)
        # 论文 §3.1.3 Eq(9): ^x = λ_h·x_h + λ_n·x_n
        # A(非周期性)已在频域做互补滤波, λ 控制整体谐波/噪声混合比例
        # 使用 sigmoid 互补门控: λ_h + λ_n = 1, 防止训练中谐波路径崩溃
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        # 融合 conditioning 特征 (mu) 到 weight_net: 让模型知道当前是辅音还是元音
        # cond_dim=128 来自 ConditioningEncoder, 拼接入 h(hidden_dim) 后投影回 hidden_dim
        self.cond_fuse = nn.Linear(hidden_dim + 128, hidden_dim)

        # 论文 §3.1.3: mel_basis 用 fft_size=2048 → 1025 bins → 128 mels
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=128, fmin=40)
        self.register_buffer('mel_basis', torch.FloatTensor(mel_basis))
        # M⁺ = max(pinv(M), 0) ∈ ℝ^{1025×128}
        mel_pinv = torch.linalg.pinv(torch.FloatTensor(mel_basis))
        mel_pinv = torch.clamp(mel_pinv, min=0)
        self.register_buffer('mel_pinv', mel_pinv)
        self.register_buffer('hann_window', torch.hann_window(fft_size))

    def harmonic_oscillator(self, f0, n_samples):
        B, _, T = f0.shape
        device = f0.device
        f0_up = F.interpolate(f0, size=n_samples, mode='linear')
        phase = 2 * np.pi * torch.cumsum(f0_up / self.sample_rate, dim=-1)
        phase = phase.unsqueeze(2)
        k = torch.arange(1, self.n_harmonics + 1, device=device).float()
        k = k.view(1, 1, -1, 1)
        harmonics = (torch.sin(k * phase) / k).sum(dim=2)
        return harmonics

    def _decompress_envelope(self, S_c):
        """论文 §3.1.3 Eq (11): Ŝ = max(M⁺ · 10^(S_c), 0) 避免负重建伪影
        修复 NaN: clamp S_c 到 [-4, 4] 防止 fp16 下 10^S_c 溢出 (10^4=10000 < 65504)"""
        S_c = torch.clamp(S_c, min=-4.0, max=4.0)
        S_exp = torch.pow(10.0, S_c)  # 从 log₁₀ 域指数化
        S_full = torch.matmul(self.mel_pinv, S_exp)
        S_full = torch.clamp(S_full, min=0)
        return S_full

    def forward(self, f0, z, target_samples=None, cond=None):
        """
        f0: (B, 1, T) — 编码器帧级 F0 (帧率 = 93.75Hz, hop=256)
        z:  (B, 128, T) — 编码器产生的时间对齐潜变量
        cond: (B, 128, T) — conditioning μ, 用于 weight_net 区分辅音/元音
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        B, _, T = f0.shape

        # Emformer 处理
        feat = torch.cat([z, f0], dim=1)
        feat = feat.transpose(1, 2)
        feat = self.emformer_input_proj(feat)
        h, _ = self.emformer(feat)
        h = self.emformer_proj(h)
        h = h.transpose(1, 2)  # (B, hidden_dim, T)

        # 预测参数 (T 帧 @ 编码器帧率)
        S_c = self.env_net(h.transpose(1, 2)).transpose(1, 2)  # (B, 128, T)
        A_c = self.ap_net(h.transpose(1, 2)).transpose(1, 2)  # (B, 16, T)
        # 论文 §3.1.3 Eq(9): sigmoid 互补门控, λ_h + λ_n = 1
        # weight_net 输入: 融合 h 和 conditioning μ (若提供), 让模型区分辅音/元音
        if cond is not None:
            # 时间对齐: cond 和 h 可能有不同的 T
            if cond.shape[-1] != h.shape[-1]:
                cond = F.interpolate(cond, size=h.shape[-1], mode='linear')
            h_w = torch.cat([h, cond], dim=1)  # (B, hidden_dim+128, T)
            h_w = self.cond_fuse(h_w.transpose(1, 2)).transpose(1, 2)  # (B, hidden_dim, T)
        else:
            h_w = h
        gate = self.weight_net(h_w.transpose(1, 2)).transpose(1, 2)       # (B,1,T) ∈ (0,1)
        lambda_h = gate
        lambda_n = 1.0 - gate

        # 解压频谱 → 全频段 (B, 1025, T)
        S_lin = self._decompress_envelope(S_c)

        # 非周期性插值: 16 → 1025 (论文: "linearly interpolated along the frequency axis")
        A_lin = A_c.transpose(1, 2)        # (B, T, 16)
        A_lin = F.interpolate(A_lin, size=self.fft_size // 2 + 1, mode='linear')
        A_lin = A_lin.transpose(1, 2)      # (B, 1025, T)

        # 产生谐波激励 (用编码器 hop 计算样本数)
        n_samples = T * self.encoder_hop_length
        e_h = self.harmonic_oscillator(f0, n_samples)

        # —— Decoder STFT: fft=2048, hop=512 (论文 §3.1.3) ——
        spec_h = torch.stft(e_h.squeeze(1), self.fft_size, self.hop_length,
                            window=self.hann_window.to(z.device),
                            return_complex=True)
        spec_h = spec_h[:, :self.fft_size // 2 + 1, :]
        n_spec_frames = spec_h.shape[-1]

        # 时间插值: 编码器帧 (T) → 解码器 STFT 帧 (n_spec_frames)
        S_spec = F.interpolate(S_lin, size=n_spec_frames, mode='linear')
        A_spec = F.interpolate(A_lin, size=n_spec_frames, mode='linear')

        # 论文 Eq (6): xh = F⁻¹((1-A)·S · F(eh))
        spec_h_filt = spec_h * (1.0 - A_spec) * S_spec
        x_h = torch.istft(spec_h_filt, self.fft_size, self.hop_length,
                          window=self.hann_window.to(z.device),
                          length=n_samples)

        # 论文 Eq (7-8): e_n ~ N(0,1), xn = F⁻¹(A·S · F(en))
        e_n = torch.randn(B, n_samples, device=z.device)
        spec_n = torch.stft(e_n, self.fft_size, self.hop_length,
                            window=self.hann_window.to(z.device),
                            return_complex=True)
        spec_n = spec_n[:, :self.fft_size // 2 + 1, :n_spec_frames]
        spec_n_filt = spec_n * A_spec[:, :, :n_spec_frames] * S_spec[:, :, :n_spec_frames]
        x_n = torch.istft(spec_n_filt, self.fft_size, self.hop_length,
                          window=self.hann_window.to(z.device),
                          length=n_samples)

        # 论文 Eq (9): 加权合成
        lambda_h_up = F.interpolate(lambda_h, size=n_samples, mode='linear')
        lambda_n_up = F.interpolate(lambda_n, size=n_samples, mode='linear')
        wav = lambda_h_up * x_h.unsqueeze(1) + lambda_n_up * x_n.unsqueeze(1)

        if target_samples is not None and wav.shape[-1] != target_samples:
            wav = wav[..., :target_samples]
        return wav


class _EmformerLayer(nn.Module):
    """论文 [18]: 完整 Emformer, 含 memory bank + summary vectors"""
    def __init__(self, input_dim, num_heads, ffn_dim, segment_length,
                 left_context_length, right_context_length, memory_size=8):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.segment_length = segment_length
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.memory_size = memory_size

        self.self_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(ffn_dim, input_dim),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.summary_proj = nn.Linear(input_dim, memory_size * input_dim)

    def _apply_segment_attention(self, x):
        B, T, D = x.shape
        L = self.segment_length
        L_left = self.left_context_length
        L_right = self.right_context_length
        num_segments = (T + L - 1) // L
        memory = torch.zeros(B, 0, D, device=x.device)
        outputs = []
        for seg_idx in range(num_segments):
            seg_start = seg_idx * L
            seg_end = min(seg_start + L, T)
            left_start = max(0, seg_start - L_left)
            left_end = seg_start
            right_start = seg_end
            right_end = min(T, seg_end + L_right)

            left_ctx = x[:, left_start:left_end, :] if left_end > left_start else torch.zeros(B, 0, D, device=x.device)
            seg = x[:, seg_start:seg_end, :]
            right_ctx = x[:, right_start:right_end, :] if right_end > right_start else torch.zeros(B, 0, D, device=x.device)

            cat = torch.cat([memory, left_ctx, seg, right_ctx], dim=1)
            attn_out, _ = self.self_attn(cat, cat, cat)
            cat_out = self.norm1(cat + attn_out)

            mem_len = memory.shape[1]
            left_len = left_ctx.shape[1]
            seg_out = cat_out[:, mem_len + left_len:mem_len + left_len + seg.shape[1], :]

            ffn_out = self.ffn(seg_out)
            seg_out = self.norm2(seg_out + ffn_out)
            outputs.append(seg_out)

            summary = self.summary_proj(seg_out.mean(dim=1, keepdim=True))
            summary = summary.view(B, self.memory_size, D)
            memory = self.norm3(summary)

        return torch.cat(outputs, dim=1)

    def forward(self, x):
        return self._apply_segment_attention(x)


class _Emformer(nn.Module):
    def __init__(self, input_dim, num_heads, ffn_dim, num_layers,
                 segment_length, left_context_length, right_context_length):
        super().__init__()
        self.layers = nn.ModuleList([
            _EmformerLayer(input_dim, num_heads, ffn_dim,
                           segment_length, left_context_length, right_context_length)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x, None
