# YUAZ SGR
"""迭代残差精炼器 (IRR)"""
import torch
import torch.nn as nn

class _MSBlock(nn.Module):
    """多尺度残差块：k=3 与 k=7 并联后相加。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.c3 = nn.Conv1d(cin, cout, 3, padding=1)
        self.c7 = nn.Conv1d(cin, cout, 7, padding=3)

    def forward(self, x):
        return 0.5 * (self.c3(x) + self.c7(x))

class IterativeRefiner(nn.Module):
    def __init__(self, cond_dim=128, hidden=384, n_layers=3, max_steps=8,
                 multi_scale=False):
        super().__init__()
        self.max_steps = int(max_steps)
        self.multi_scale = bool(multi_scale)
        self.step_emb = nn.Embedding(self.max_steps + 1, 32)
        if self.multi_scale:
            first = _MSBlock(cond_dim * 2 + 32, hidden)
        else:
            first = nn.Conv1d(cond_dim * 2 + 32, hidden, 5, padding=2)
        layers = [first, nn.GELU()]
        for _ in range(max(0, n_layers - 2)):
            layers += [_MSBlock(hidden, hidden) if self.multi_scale
                       else nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU()]
        layers += [nn.Conv1d(hidden, cond_dim, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def step(self, z, mu, k):
        emb = self.step_emb(torch.full((z.shape[0],), min(k, self.max_steps), device=z.device))
        emb = emb.unsqueeze(-1).expand(-1, -1, z.shape[-1])
        return z + self.net(torch.cat([z, mu, emb], dim=1))

    def forward(self, mu, steps=4):
        z = mu
        for k in range(max(1, int(steps))):
            z = self.step(z, mu, k)
        return z
