# YUAZ SGR
"""models/discriminator"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=None):
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.periods = periods
        self.discriminators = nn.ModuleList([
            self._make_disc() for _ in periods
        ])

    def _make_disc(self):
        return nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv2d(1, 16, (5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(16, 64, (5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(64, 256, (5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(256, 256, (5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(256, 256, (5, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2),
            ),
            weight_norm(nn.Conv2d(256, 1, (3, 1), padding=(1, 0))),
        ])

    def forward(self, x):
        fmaps_real = []
        scores_real = []
        for i, disc in enumerate(self.discriminators):
            b, c, t = x.shape
            p = self.periods[i]
            if t % p != 0:
                pad_len = p - t % p
                x_pad = F.pad(x, (0, pad_len))
            else:
                x_pad = x
            x_2d = x_pad.view(b, c, -1, p).transpose(2, 3)
            h = x_2d
            fmaps = []
            for layer in disc:
                h = layer(h)
                fmaps.append(h)
            score = h
            scores_real.append(score)
            fmaps_real.append(fmaps)
        return fmaps_real, scores_real

    def forward_fake(self, x):
        fmaps_fake = []
        scores_fake = []
        for i, disc in enumerate(self.discriminators):
            b, c, t = x.shape
            p = self.periods[i]
            if t % p != 0:
                pad_len = p - t % p
                x_pad = F.pad(x, (0, pad_len))
            else:
                x_pad = x
            x_2d = x_pad.view(b, c, -1, p).transpose(2, 3)
            h = x_2d
            fmaps = []
            for layer in disc:
                h = layer(h)
                fmaps.append(h)
            score = h
            scores_fake.append(score)
            fmaps_fake.append(fmaps)
        return fmaps_fake, scores_fake

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            self._make_disc() for _ in range(3)
        ])
        self.pool = nn.AvgPool1d(4, stride=2, padding=2)

    def _make_disc(self):
        return nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(1, 32, 15, padding=7)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(32, 32, 41, stride=2, groups=4, padding=20)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(32, 64, 41, stride=2, groups=16, padding=20)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(64, 128, 41, stride=4, groups=16, padding=20)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(128, 256, 41, stride=4, groups=16, padding=20)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(256, 128, 5, padding=2)),
                nn.LeakyReLU(0.2),
            ),
            weight_norm(nn.Conv1d(128, 1, 3, padding=1)),
        ])

    def forward(self, x):
        fmaps_real = []
        scores_real = []
        for disc in self.discriminators:
            h = x
            fmaps = []
            for layer in disc:
                h = layer(h)
                fmaps.append(h)
            score = h
            scores_real.append(score)
            fmaps_real.append(fmaps)
            x = self.pool(x)
        return fmaps_real, scores_real

    def forward_fake(self, x):
        fmaps_fake = []
        scores_fake = []
        for disc in self.discriminators:
            h = x
            fmaps = []
            for layer in disc:
                h = layer(h)
                fmaps.append(h)
            score = h
            scores_fake.append(score)
            fmaps_fake.append(fmaps)
            x = self.pool(x)
        return fmaps_fake, scores_fake
