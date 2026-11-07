# YUAZ SGR
"""models/encoder"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class Encoder(nn.Module):
    def __init__(self, hidden_dim=512, latent_dim=128,
                 sample_rate=24000, n_mfcc=30, n_fft=1024, hop_length=256,
                 feature='mfcc', n_mels=80):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        # feature: 'logmel' | 'mfcc'
        self.feature = str(feature).lower()
        self.n_mels = n_mels
        feature_dim = self.n_mels if self.feature == 'logmel' else n_mfcc

        mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
                                     fmin=40, htk=False, norm='slaney')
        self.register_buffer('_mel_fb', torch.from_numpy(mel_fb).float(), persistent=False)
        self.register_buffer('_hann', torch.hann_window(n_fft), persistent=False)

        # DDSP z-encoder: MFCC → Conv1D → GRU → z₁
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, 64, 7, stride=1, padding=3),
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

    def _compute_logmel(self, x):
        """修复 #9: 对数梅尔谱特征 (默认 80 维) — 比 MFCC 保留更多高频细节 优化: 全程在计算设备上用 torch"""
        wav = x.squeeze(1).float()
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.n_fft,
                          window=self._hann.to(device=wav.device, dtype=wav.dtype),
                          center=True, pad_mode='constant', return_complex=True)
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = torch.matmul(self._mel_fb.to(device=wav.device, dtype=power.dtype), power)
        return torch.log(mel.clamp(min=1e-5))

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
        feat = self._compute_logmel(x) if self.feature == 'logmel' else self._compute_mfcc(x)
        h = self.conv(feat)
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
