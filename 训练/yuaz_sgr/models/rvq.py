# YUAZ SGR
"""
models/rvq.py
论文 §4.3.1: Residual Vector Quantization (VQ-VAE 风格 EMA 码本更新)

8 层量化器, 每层码本 1024×128
EMA 更新 (VQ-VAE [21]) 防码本坍塌.

修复 v3: 码本初始化标准差增大 10x (0.008 → 0.088)
         + 死码重启机制 (ema_size < 2.0 → 随机输入重初始化)
"""
import math, torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_quantizers=8, codebook_size=1024, codebook_dim=128,
                 commitment_weight=1.0, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps

        codebooks = []
        for _ in range(num_quantizers):
            cb = nn.Embedding(codebook_size, codebook_dim)
            # 修复 v3: std=1/sqrt(dim) ≈ 0.088, 远大于原 1/dim ≈ 0.008
            # 让码本条目的初始分布更分散, 减少多个输入同时映射到同一条目
            nn.init.normal_(cb.weight, mean=0, std=1.0 / math.sqrt(codebook_dim))
            codebooks.append(cb)
        self.codebooks = nn.ModuleList(codebooks)

        self.register_buffer('ema_size',
                             torch.zeros(num_quantizers, codebook_size))
        self.register_buffer('ema_sum',
                             torch.zeros(num_quantizers, codebook_size, codebook_dim))

    def forward(self, z):
        residual = z
        z_q_sum = 0
        all_indices = []
        commit_loss = 0

        for i, cb in enumerate(self.codebooks):
            flat = residual.transpose(1, 2).contiguous().reshape(-1, self.codebook_dim)
            dist = flat.pow(2).sum(1, keepdim=True) \
                   - 2 * flat @ cb.weight.T \
                   + cb.weight.pow(2).sum(1)
            idx = dist.argmin(-1)
            idx_2d = idx.view(z.shape[0], z.shape[2])
            all_indices.append(idx_2d)
            q = cb(idx_2d).transpose(1, 2).contiguous()

            commit_loss = commit_loss + F.mse_loss(residual, q.detach())

            if self.training:
                with torch.no_grad():
                    flat_fp32 = flat.float()
                    onehot = F.one_hot(idx, self.codebook_size).float()
                    n_new = onehot.sum(0)
                    s_new = onehot.T @ flat_fp32
                    self.ema_size[i] = self.decay * self.ema_size[i] + (1 - self.decay) * n_new
                    self.ema_sum[i] = self.decay * self.ema_sum[i] + (1 - self.decay) * s_new
                    n = self.ema_size[i].sum().clamp(min=1.0)
                    n_smooth = (self.ema_size[i] + self.eps) / (n + self.codebook_size * self.eps) * n
                    new_w = (self.ema_sum[i] / n_smooth.unsqueeze(1).clamp(min=self.eps)).float()
                    if not torch.isnan(new_w).any():
                        cb.weight.data.copy_(new_w)

                    # 修复 v3: 死码重启 — ema_size < 2.0 的码本条目
                    # 用当前批次的随机输入向量替换, 确保每个码本都被充分使用
                    dead = self.ema_size[i] < 2.0
                    if dead.any():
                        n_dead = dead.sum().item()
                        n_avail = flat_fp32.shape[0]
                        if n_dead <= n_avail:
                            perm = torch.randperm(n_avail, device=flat.device)
                            replacement = flat_fp32[perm[:n_dead]]
                        else:
                            # 死码多于可用向量, 重复采样
                            n_repeat = (n_dead + n_avail - 1) // n_avail
                            perm = torch.randperm(n_avail, device=flat.device)
                            replacement = flat_fp32[perm].repeat(n_repeat, 1)[:n_dead]
                        cb.weight.data[dead] = replacement.to(cb.weight.device)
                        self.ema_size[i][dead] = 1.0
                        self.ema_sum[i][dead] = replacement.to(self.ema_sum.device)

            q_st = residual + (q - residual).detach()
            z_q_sum = z_q_sum + q_st
            residual = residual - q_st

        loss = self.commitment_weight * commit_loss
        return z_q_sum, torch.stack(all_indices, dim=1), loss
