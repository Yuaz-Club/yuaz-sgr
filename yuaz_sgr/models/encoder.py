# YUAZ SGR
"""
models/encoder.py
论文 §3.1.1 + §4.3.1: DDSP z-encoder + F0 encoder (CREPE [30])

输入: 波形 (B, 1, T_samples)
输出: z₁ (B, 128, T_frames), F0 (B, 1, T_frames)

修复: 论文 §4.3.1 明确 "maps MFCC-based representations to a compact latent
      embedding through a GRU layer", 与原始 DDSP 一致使用 30 维 MFCC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa


class Encoder(nn.Module):
    def __init__(self, hidden_dim=512, latent_dim=128,
                 sample_rate=24000, n_mfcc=30, n_fft=1024, hop_length=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # 论文 §4.3.1: DDSP z-encoder, MFCC → Conv1D → GRU → z₁
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, 7, stride=1, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 7, stride=1, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 7, stride=1, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, hidden_dim, 7, stride=1, padding=3),
            nn.LeakyReLU(0.2),
        )

        self.gru = nn.GRU(hidden_dim, hidden_dim // 2,
                          batch_first=True, bidirectional=True)
        self.proj = nn.Conv1d(hidden_dim, latent_dim, 1)

        try:
            import torchcrepe
            self.crepe_available = True
        except ImportError:
            print("WARNING: torchcrepe 未安装")
            self.crepe_available = False

    def _compute_mfcc(self, x):
        """论文 §4.3.1: DDSP MFCC 特征 (30维)"""
        x_np = x.squeeze(1).cpu().numpy()
        B = x_np.shape[0]
        mfcc_list = []
        for b in range(B):
            m = librosa.feature.mfcc(
                y=x_np[b], sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=40,
            )
            mfcc_list.append(torch.FloatTensor(m))
        mfcc = torch.stack(mfcc_list, dim=0).to(x.device)
        return mfcc

    def _extract_f0(self, x):
        """论文 [30]: CREPE 卷积基频估计"""
        B = x.shape[0]
        hop = self.hop_length
        T_frames_target = (x.shape[-1] // hop) + 1
        f0 = torch.zeros(B, 1, T_frames_target, device=x.device)
        import torchcrepe
        with torch.no_grad():
            for i in range(B):
                audio_i = x[i, 0, :].unsqueeze(0)
                f0_i = torchcrepe.predict(
                    audio_i, self.sample_rate,
                    hop_length=hop,
                    fmin=50.0, fmax=2006.0,
                    model='full',
                    device=x.device,
                    pad=True,
                )
                n_f = min(f0_i.shape[-1], T_frames_target)
                if f0_i.dim() == 3:
                    f0[i, 0, :n_f] = f0_i[0, 0, :n_f]
                else:
                    f0[i, 0, :n_f] = f0_i[0, :n_f]
        return f0

    def forward(self, x, f0_override=None):
        mfcc = self._compute_mfcc(x)
        h = self.conv(mfcc)
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        h = h.transpose(1, 2)
        z1 = self.proj(h)

        if f0_override is not None:
            f0 = f0_override
            if f0.dim() == 2:
                f0 = f0.unsqueeze(1)
            elif f0.dim() == 3:
                f0 = f0
            else:
                f0 = f0.unsqueeze(1)
            if f0.shape[-1] != z1.shape[-1]:
                f0 = F.interpolate(f0, size=z1.shape[-1], mode='linear')
        elif self.training:
            raise RuntimeError("训练时必须提供 f0_override, 禁止每步实时 CREPE")
        else:
            f0 = self._extract_f0(x)
            if f0.shape[-1] != z1.shape[-1]:
                f0 = F.interpolate(f0, size=z1.shape[-1], mode='linear')

        return z1, f0
