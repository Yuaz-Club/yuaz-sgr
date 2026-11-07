# YUAZ SGR
"""losses/gan_loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    def __init__(self, mpd, msd):
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def compute_g_loss(self, real_wav, fake_wav):
        """仅 G loss + FM loss (用于生成器 total_loss)"""
        with torch.no_grad():
            mpd_real_fmaps, _ = self.mpd(real_wav)
            msd_real_fmaps, _ = self.msd(real_wav)
        mpd_fake_fmaps, mpd_fake_scores = self.mpd.forward_fake(fake_wav)
        msd_fake_fmaps, msd_fake_scores = self.msd.forward_fake(fake_wav)

        g_loss = 0.0
        for s in mpd_fake_scores + msd_fake_scores:
            g_loss = g_loss + ((s - 1) ** 2).mean()

        fm_loss = 0.0
        for fm_r, fm_f in zip(mpd_real_fmaps + msd_real_fmaps,
                               mpd_fake_fmaps + msd_fake_fmaps):
            for r, f in zip(fm_r, fm_f):
                fm_loss = fm_loss + F.l1_loss(r, f)
        return g_loss, fm_loss

    def compute_d_loss(self, real_wav, fake_wav):
        """完整 D loss, 独立 forward (论文 §3.1.4: Σ)"""
        _, mpd_r = self.mpd(real_wav)
        _, msd_r = self.msd(real_wav)
        fake_d = fake_wav.detach()
        _, mpd_f = self.mpd.forward_fake(fake_d)
        _, msd_f = self.msd.forward_fake(fake_d)
        d_loss = 0.0
        for s in mpd_r + msd_r:
            d_loss = d_loss + ((s - 1) ** 2).mean()
        for s in mpd_f + msd_f:
            d_loss = d_loss + (s ** 2).mean()
        return d_loss
