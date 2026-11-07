# YUAZ SGR
"""models/ddsp_decoder"""
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
                 emformer_left_context=8, emformer_right_context=4,
                 voicing_harmonic=0.15, voicing_aperiodic=0.60,
                 n_mels=128, spectral_residual_rank=0,
                 harm_rolloff='soft', harm_amp='fixed', harm_synth='stft',
                 mel_pinv_clamp=True, silence_gate=False, env_unbounded=False,
                 env_cond=False, noise_scale=1.0, env_softplus=False,
                 harm_amp_pred=False):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.encoder_hop_length = encoder_hop_length
        self.n_mels = n_mels
        self.spectral_residual_rank = spectral_residual_rank
        self.harm_rolloff = harm_rolloff
        self.harm_amp = harm_amp
        self.harm_synth = harm_synth

        self.mel_pinv_clamp = mel_pinv_clamp
        self.silence_gate = silence_gate
        # env_cond: 包络网络是否接收 μ
        self.env_cond = env_cond
        # noise_scale: 噪声通路增益
        self.noise_scale = noise_scale
        # env_softplus: 包络负值走 softplus，替代 clamp 硬零
        self.env_softplus = env_softplus
        # harm_amp_pred: 直接预测各谐波幅度，不经 mel 伪逆
        self.harm_amp_pred = harm_amp_pred

        self.voicing_harmonic = voicing_harmonic
        self.voicing_aperiodic = voicing_aperiodic

        if harm_amp_pred:
            self.harm_head = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, self.n_harmonics),
            )
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

        if env_unbounded:
            self.env_net = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, n_mels),
            )
            nn.init.normal_(self.env_net[-1].bias, mean=-2.0, std=0.5)
        else:
            self.env_net = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, n_mels), nn.LeakyReLU(0.2),
            )
            nn.init.normal_(self.env_net[-2].bias, mean=-2.0, std=0.5)
        self.ap_net = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 16), nn.Sigmoid()
        )
        # A∈[0,1] 非周期性系数；sigmoid(-3)≈0.05 初始化偏谐波
        nn.init.constant_(self.ap_net[-2].bias, -3.0)
        # Eq(9) λ_h + λ_n = 1
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        # 融合 μ 到 weight_net（拼接入 h 后投影回 hidden_dim）
        self.cond_fuse = nn.Linear(hidden_dim + 128, hidden_dim)

        if silence_gate:
            # 帧级活动度门控
            self.act_net = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 1), nn.Sigmoid())

        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_mels, fmin=40)
        self.register_buffer('mel_basis', torch.FloatTensor(mel_basis))
        # M⁺ = max(pinv(M), 0)
        mel_pinv = torch.linalg.pinv(torch.FloatTensor(mel_basis))
        if mel_pinv_clamp:
            mel_pinv = torch.clamp(mel_pinv, min=0)
        self.register_buffer('mel_pinv', mel_pinv)
        self.register_buffer('hann_window', torch.hann_window(fft_size))
        if spectral_residual_rank > 0:
            self.res_coef = nn.Linear(n_mels, spectral_residual_rank)
            self.res_basis = nn.Parameter(
                torch.randn(fft_size // 2 + 1, spectral_residual_rank) * 0.01)
        if harm_amp == 'learn':
            self.harm_gain = nn.Parameter(
                torch.arange(1, n_harmonics + 1).float().reciprocal())
        elif harm_amp == 'power':
            self.harm_p = nn.Parameter(torch.tensor(1.0))

    def harmonic_oscillator(self, f0, n_samples, voiced=None):
        """论文 Eq(4)(5): 谐波激励 e_h(t) = Σ_k (1/k)·sin(φ_k(t)) 修复 #1 (抗混叠): 逐谐波累加并做 Nyquist 掩蔽"""
        B, _, T = f0.shape
        device = f0.device
        f0_up = F.interpolate(f0, size=n_samples, mode='linear')
        f0_up = torch.clamp(f0_up, min=0.0)
        phase = 2 * np.pi * torch.cumsum(f0_up / self.sample_rate, dim=-1)  # (B,1,N)

        nyquist = self.sample_rate / 2.0
        taper = 0.10 * nyquist
        harmonics = torch.zeros_like(phase)  # (B,1,N) — O(N) 而非 O(K·N)
        for k in range(1, self.n_harmonics + 1):
            fk = k * f0_up                     # (B,1,N) 该谐波频率
            if not bool((fk < nyquist).any()):
                break
            valid = self._nyquist_valid(fk, nyquist, taper)
            harmonics = harmonics + torch.sin(k * phase) * self._harm_amp(k) * valid

        if voiced is not None:
            harmonics = harmonics * voiced
        return harmonics

    def _nyquist_valid(self, fk, nyquist, taper):
        """(B) Nyquist 掩蔽: 'soft' 顶部 10% 软滚降 (现有) | 'hard' 硬截止。"""
        if self.harm_rolloff == 'hard':
            return (fk < nyquist).to(fk.dtype)
        return ((nyquist - fk) / taper).clamp(0.0, 1.0)

    def _harm_amp(self, k):
        """(C) 第 k 次谐波振幅权重 (k 从 1 计): fixed=1/k | learn 可学习 | power=1/k^p。"""
        if self.harm_amp == 'learn':
            return self.harm_gain[k - 1]
        if self.harm_amp == 'power':
            return k ** (-self.harm_p)
        return 1.0 / k

    def _harmonic_from_amps(self, f0, n_samples, amps, voiced=None):
        """逐谐波幅度合成: e_h = Σ_k amp_k(t)·sin(k·φ(t)), Nyquist 掩蔽"""
        B, _, T = f0.shape
        nyquist = self.sample_rate / 2.0
        taper = 0.10 * nyquist
        f0_up = F.interpolate(f0, size=n_samples, mode='linear')
        f0_up = torch.clamp(f0_up, min=0.0)
        phase = 2 * np.pi * torch.cumsum(f0_up / self.sample_rate, dim=-1)
        out = torch.zeros_like(phase)
        for k in range(1, self.n_harmonics + 1):
            fk = k * f0_up
            if not bool((fk < nyquist).any()):
                break
            a = F.interpolate(amps[:, k - 1:k, :], size=n_samples, mode='linear')
            out = out + torch.sin(k * phase) * a / k * self._nyquist_valid(fk, nyquist, taper)
        if voiced is not None:
            out = out * voiced
        return out.squeeze(1)

    def _harmonic_td(self, f0, n_samples, S_lin, A_lin, voiced_up):
        """(D) 时域谐波合成: 逐谐波 amplitude = (1-A)·Ŝ 在该谐波频率处采样"""
        B, _, T = f0.shape
        nbins = S_lin.shape[1]
        df = self.sample_rate / self.fft_size
        nyquist = self.sample_rate / 2.0
        taper = 0.10 * nyquist
        f0_up = F.interpolate(f0, size=n_samples, mode='linear')
        f0_up = torch.clamp(f0_up, min=0.0)
        phase = 2 * np.pi * torch.cumsum(f0_up / self.sample_rate, dim=-1)
        f0_f = f0.squeeze(1)                       # (B,T)
        e_h = torch.zeros_like(phase)
        for k in range(1, self.n_harmonics + 1):
            fk = k * f0_f
            if not bool((fk < nyquist).any()):
                break
            idx = (fk / df).clamp(0.0, nbins - 1.0)
            i0 = idx.floor().long()
            i1 = (i0 + 1).clamp(max=nbins - 1)
            w1 = (idx - i0.float()).clamp(0.0, 1.0).unsqueeze(1)
            S0 = S_lin.gather(1, i0.unsqueeze(1))
            S1 = S_lin.gather(1, i1.unsqueeze(1))
            A0 = A_lin.gather(1, i0.unsqueeze(1))
            A1 = A_lin.gather(1, i1.unsqueeze(1))
            S_at = S0 * (1 - w1) + S1 * w1
            A_at = A0 * (1 - w1) + A1 * w1
            valid = self._nyquist_valid(fk, nyquist, taper).unsqueeze(1)
            amp = (1.0 - A_at) * S_at * self._harm_amp(k) * valid
            amp_up = F.interpolate(amp, size=n_samples, mode='linear')
            e_h = e_h + amp_up * torch.sin(k * phase)
        return (e_h * voiced_up).squeeze(1)     # (B,N) 与 istft 输出对齐

    def _decompress_envelope(self, S_c):
        """论文 §3"""
        S_c = torch.clamp(S_c, min=-4.0, max=4.0)
        S_exp = torch.pow(10.0, S_c)  # 从 log₁₀ 域指数化
        S_base = torch.matmul(self.mel_pinv, S_exp)
        if self.spectral_residual_rank > 0:
            log_base = torch.log10(S_base.float().clamp(min=1e-6))
            coef = self.res_coef(S_c.transpose(1, 2)).transpose(1, 2).float()
            log_base = log_base + torch.matmul(self.res_basis.float(), coef)
            S_full = torch.pow(10.0, log_base.clamp(min=-8.0, max=4.0)).to(S_base.dtype)
        else:
            S_full = S_base
        if getattr(self, 'env_softplus', False):
            S_full = F.softplus(S_full)
        else:
            S_full = torch.clamp(S_full, min=0)
        return S_full

    def forward(self, f0, z, target_samples=None, cond=None, voicing=None,
                voicing_harmonic=None, voicing_aperiodic=None, return_aux=False):
        """f0: (B, 1, T) — 编码器帧级 F0 (帧率 = 93"""
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

        # conditioning 融合（μ）
        if cond is not None:
            # 时间对齐
            if cond.shape[-1] != h.shape[-1]:
                cond = F.interpolate(cond, size=h.shape[-1], mode='linear')
            h_w = torch.cat([h, cond], dim=1)  # (B, hidden_dim+128, T)
            h_w = self.cond_fuse(h_w.transpose(1, 2)).transpose(1, 2)  # (B, hidden_dim, T)
        else:
            h_w = h
        # 预测参数（编码器帧率）；env_cond 时包络也接收 μ
        env_in = h_w if self.env_cond else h
        S_c = self.env_net(env_in.transpose(1, 2)).transpose(1, 2)  # (B, n_mels, T)
        A_c = self.ap_net(h.transpose(1, 2)).transpose(1, 2)  # (B, 16, T)
        # Eq(9) 互补门控
        gate = self.weight_net(h_w.transpose(1, 2)).transpose(1, 2)       # (B,1,T) ∈ (0,1)
        # 帧级活动度门控
        act = self.act_net(h_w.transpose(1, 2)).transpose(1, 2) if self.silence_gate else None

        A_mean = A_c.mean(dim=1, keepdim=True)          # (B,1,T) 模型原始非周期度
        gate_base = gate                                # (B,1,T) 模型原始 λ_h
        A_tgt = g_tgt = None
        if voicing is not None:
            v = voicing.unsqueeze(1) if voicing.dim() == 2 else voicing
            if v.shape[-1] != A_c.shape[-1]:
                v = F.interpolate(v.float(), size=A_c.shape[-1], mode='nearest')
            v = v.to(A_c.dtype)
            a_h = self.voicing_harmonic if voicing_harmonic is None else voicing_harmonic
            a_u = self.voicing_aperiodic if voicing_aperiodic is None else voicing_aperiodic
            A_tgt = a_h * v + a_u * (1 - v)             # (B,1,T) 目标非周期度
            g_tgt = 1.0 - A_tgt                         # 目标 λ_h
        lambda_h = g_tgt if A_tgt is not None else gate
        lambda_n = 1.0 - lambda_h

        # 解压到全频段
        S_lin = self._decompress_envelope(S_c)

        # A 插值 16 → 1025
        A_lin = A_c.transpose(1, 2)        # (B, T, 16)
        A_lin = F.interpolate(A_lin, size=self.fft_size // 2 + 1, mode='linear')
        A_lin = A_lin.transpose(1, 2)      # (B, 1025, T)
        if A_tgt is not None:
            # 用目标 A 覆盖
            A_lin = A_tgt.to(A_lin.dtype).expand(-1, self.fft_size // 2 + 1, -1)

        # 谐波激励
        n_samples = T * self.encoder_hop_length
        # f0<=0 的帧不产生谐波激励
        voiced = (f0 > 0).to(f0.dtype)                                  # (B,1,T)
        voiced_up = F.interpolate(voiced, size=n_samples, mode='nearest')

        if self.harm_amp_pred:
            # 谐波域幅度预测
            amps = F.softplus(self.harm_head(h.transpose(1, 2)).transpose(1, 2))  # (B,K,T)
            x_h = self._harmonic_from_amps(f0, n_samples, amps, voiced_up)
            n_spec_frames = 1 + n_samples // self.hop_length
        elif self.harm_synth == 'td':
            # 时域谐波合成
            x_h = self._harmonic_td(f0, n_samples, S_lin, A_lin, voiced_up)
            n_spec_frames = 1 + n_samples // self.hop_length
        else:
            e_h = self.harmonic_oscillator(f0, n_samples, voiced=voiced_up)
            # Decoder STFT: fft=2048, hop=512
            spec_h = torch.stft(e_h.squeeze(1), self.fft_size, self.hop_length,
                                window=self.hann_window.to(z.device),
                                return_complex=True)
            spec_h = spec_h[:, :self.fft_size // 2 + 1, :]
            n_spec_frames = spec_h.shape[-1]

        # 帧率插值 T → n_spec_frames
        S_spec = F.interpolate(S_lin, size=n_spec_frames, mode='linear')
        A_spec = F.interpolate(A_lin, size=n_spec_frames, mode='linear')

        if (not self.harm_amp_pred) and self.harm_synth != 'td':
            # Eq(6)
            spec_h_filt = spec_h * (1.0 - A_spec) * S_spec
            x_h = torch.istft(spec_h_filt, self.fft_size, self.hop_length,
                              window=self.hann_window.to(z.device),
                              length=n_samples)

        # Eq(7)(8)
        e_n = torch.randn(B, n_samples, device=z.device)
        spec_n = torch.stft(e_n, self.fft_size, self.hop_length,
                            window=self.hann_window.to(z.device),
                            return_complex=True)
        spec_n = spec_n[:, :self.fft_size // 2 + 1, :n_spec_frames]
        spec_n_filt = spec_n * A_spec[:, :, :n_spec_frames] * S_spec[:, :, :n_spec_frames]
        x_n = torch.istft(spec_n_filt, self.fft_size, self.hop_length,
                          window=self.hann_window.to(z.device),
                          length=n_samples)
        if self.noise_scale != 1.0:
            x_n = x_n * self.noise_scale

        # Eq(9) 加权合成
        lambda_h_up = F.interpolate(lambda_h, size=n_samples, mode='linear')
        lambda_n_up = F.interpolate(lambda_n, size=n_samples, mode='linear')
        wav = lambda_h_up * x_h.unsqueeze(1) + lambda_n_up * x_n.unsqueeze(1)

        # 静音门控
        if act is not None:
            wav = wav * F.interpolate(act, size=wav.shape[-1], mode='linear')

        if target_samples is not None and wav.shape[-1] != target_samples:
            wav = wav[..., :target_samples]
        if return_aux:
            # 模型原始 A 与 gate（未被子先验覆盖）
            return wav, {'aperiodicity': A_mean, 'gate': gate_base, 'envelope': S_c, 'act': act,
                         'xh': x_h, 'xn': x_n}
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
