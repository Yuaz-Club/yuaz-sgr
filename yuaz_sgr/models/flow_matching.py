# YUAZ SGR
"""
models/flow_matching.py
论文 §3.2 + §4.3.2: Flow Matching Latent Generator

架构: Matcha-TTS 风格, Conv1D(k=7) + Emformer[18] + skip + time embed 拼接
输出 129 通道: 前 128 → 速度场 v_θ, 第 129 → 精炼 F0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ddsp_decoder import _EmformerLayer


class FlowMatchingBlock(nn.Module):
    """
    单个 Conv1D + Emformer block
    修复 #6:  time embedding 拼接 (非加法)
    修复 #38: 使用 _EmformerLayer (segment-based attention, 论文 [18])
    """
    def __init__(self, hidden_dim=256, time_dim=64):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, time_dim)

        # Conv1D: 输入 hidden_dim+time_dim → 输出 hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim + time_dim, hidden_dim, 7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        # 修复 #38: Emformer layer (segment-based, 论文 [18])
        # segment_length=32, left_context=8, right_context=4
        self.emformer = _EmformerLayer(
            input_dim=hidden_dim, num_heads=8, ffn_dim=hidden_dim * 4,
            segment_length=32, left_context_length=8, right_context_length=4,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, x, t_emb):
        """
        x: (B, H, T)
        t_emb: (B, 64)
        """
        t_feat = self.time_proj(t_emb).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x_cat = torch.cat([x, t_feat], dim=1)  # (B, H+64, T)

        residual = x
        x_conv = self.conv(x_cat)
        x_out = self.norm1((x_conv + residual).transpose(1, 2)).transpose(1, 2)

        x_t = x_out.transpose(1, 2)
        x_t = self.emformer(x_t)
        x_out = x_t.transpose(1, 2)

        return x_out


class FlowMatchingGenerator(nn.Module):
    """
    论文 §4.3.2: Matcha-TTS 风格 Latent Generator

    输入: z_t (B,128,T), t (B,), μ (B,128,T)
    输出: gen_out (B,129,T) — 前 128=v_t, 第 129=F0_refined
    """
    def __init__(self, input_dim=128, hidden_dim=256,
                 num_blocks=6, time_dim=64, output_dim=129):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.time_dim = time_dim      # 修复 #29: 存储 time_dim
        self.z_dim = 128            # ODE 状态维度 (论文 §3.2: z∈ℝ^{d×T})
        self.output_dim = output_dim  # 129 = 128(v_t) + 1(F0)

        # 修复 #21: 输入 (z_t(128) + mu(128)) = 256 → hidden_dim
        self.input_proj = nn.Conv1d(self.z_dim + input_dim, hidden_dim, 1)

        # 堆叠 Conv1D + Emformer blocks
        self.blocks = nn.ModuleList([
            FlowMatchingBlock(hidden_dim=hidden_dim, time_dim=time_dim)
            for _ in range(num_blocks)
        ])

        # 修复 #28: 输出投影 MLP (Conv1d kernel=1)
        # 论文: "output features are projected by a multilayer perceptron"
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim * num_blocks, hidden_dim, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, output_dim, 1),  # 129 = 128(v_t) + 1(F0)
        )

    def forward(self, z_t, t, mu):
        """
        修复 #19: z_t (B,128,T) — 仅潜变量 ODE 状态
        t: (B,) 或 (B,1)
        mu: (B,128,T)
        返回: gen_out (B,129,T) — 前 128 通道 = v_t, 第 129 通道 = F0_refined
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)

        B, _, T = z_t.shape

        # Sinusoidal time embedding (dim=64)
        t_emb = self._sinusoidal_embedding(t)  # (B, 64)

        # 修复 #21: 输入投影 (z_t(128) + mu(128))
        x = torch.cat([z_t, mu], dim=1)         # (B, 256, T)
        x = self.input_proj(x)                   # (B, hidden_dim, T)

        # 修复 #5: 跨层 skip connections — 收集每层输出
        skip_features = []
        for block in self.blocks:
            x = block(x, t_emb)                  # (B, hidden_dim, T)
            skip_features.append(x)

        # 合并所有 skip features
        x = torch.cat(skip_features, dim=1)     # (B, hidden_dim*6, T)
        gen_out = self.output_proj(x)            # (B, 129, T)
        return gen_out

    def _sinusoidal_embedding(self, t):
        """论文 §4.3.2: sinusoidal time embedding, dim=64"""
        device = t.device
        half = self.time_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / half
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    @torch.no_grad()
    def sample(self, mu, n_steps=1):
        """
        论文 §4.3.2: 一阶 Euler 法积分 ODE (推理)

        修复 #19: 仅积分 128-dim z (非 129-dim)
        修复 #31: denoiser Euler z ← z + (v_t − z)·dt/(1−t)

        从 N(0,1) 噪声开始, ODE 逐步积分得到 z₁
        返回: z (128-dim 潜变量), f0_refined (精炼 F0)
        """
        B, _, T = mu.shape
        device = mu.device
        z = torch.randn(B, self.z_dim, T, device=device)  # 128-dim ODE 初始状态
        dt = 1.0 / n_steps
        f0_refined = None

        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            gen_out = self.forward(z, t, mu)          # (B, 129, T)
            v_z = gen_out[:, :self.z_dim, :]            # 速度场 v_θ(z_t, t)

            # 论文 §4.3.2: 一阶 Euler: z_{t+dt} = z_t + v_θ(z_t, t) · dt
            z = z + v_z * dt

            f0_refined = gen_out[:, self.z_dim:, :]     # 精炼 F0 (取最后一步)

        return z, f0_refined