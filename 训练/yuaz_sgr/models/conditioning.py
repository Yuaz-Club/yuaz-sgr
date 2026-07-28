# YUAZ SGR
"""
models/conditioning.py
论文 §4.3.2: Conditioning Encoder

输入: 音素 ID + 时长 + F0 → 条件特征 μ (128-dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningEncoder(nn.Module):
    def __init__(self, num_phonemes, phoneme_dim=256, cond_dim=128):
        super().__init__()
        self.phoneme_embed = nn.Embedding(num_phonemes, phoneme_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(phoneme_dim + 1, 256, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, cond_dim, 3, padding=1),
        )

    def _length_regulate(self, emb, durations):
        B, N, D = emb.shape
        device = emb.device
        outputs = []
        for b in range(B):
            durs = durations[b]
            valid = durs > 0
            ph_ids = torch.where(valid)[0]
            reps = durs[valid].long()
            seq = torch.repeat_interleave(emb[b, ph_ids], reps, dim=0)
            outputs.append(seq)
        max_len = max(o.shape[0] for o in outputs) if outputs else 1
        result = torch.zeros(B, max_len, D, device=device)
        for b, o in enumerate(outputs):
            result[b, :o.shape[0]] = o
        return result

    def forward(self, phoneme_ids, durations, f0):
        ph_emb = self.phoneme_embed(phoneme_ids)
        ph_frames = self._length_regulate(ph_emb, durations)

        if f0.dim() == 3:
            f0 = f0.squeeze(1)

        if f0.shape[-1] != ph_frames.shape[1]:
            if ph_frames.shape[1] < f0.shape[-1]:
                pad = f0.shape[-1] - ph_frames.shape[1]
                ph_frames = torch.nn.functional.pad(ph_frames, (0, 0, 0, pad))
            else:
                ph_frames = ph_frames[:, :f0.shape[-1], :]

        f0_in = f0.unsqueeze(-1)
        x = torch.cat([ph_frames, f0_in], dim=-1)
        x = x.transpose(1, 2)
        mu = self.encoder(x)
        return mu


class TemporalPredictor(nn.Module):
    """
    论文 §3.3: Time-lag Predictor + Duration Predictor (音符/音素级)

    §3.3.1: 每音符一个 lag 标量 ĝ_n, L_lag = E||g - ĝ||²
    §3.3.2: 每音素一个 duration 标量 d̂_nk, L_dur = E||d - d̂||²

    dur_predictor 输入: ph_emb(256) + f0(1) + pos_enc(32) + note_len(1) + ph_pos(1) = 291
      关键改动: 使用音素级嵌入（与时长无关），而非帧级 mu 池化（销毁时长信号）
    lag_predictor 输入: mu_pool(128) + f0(1) + pos_enc(32) = 161
    """
    def __init__(self, cond_dim=128, num_phonemes=60, phoneme_dim=256):
        super().__init__()
        # Conv1D 预处理器: lag_predictor 用帧级 mu
        self.pre_conv = nn.Sequential(
            nn.Conv1d(cond_dim, 128, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, padding=1),
        )
        # 音素嵌入: 回退用（当不传 phoneme_emb 时）
        self.ph_embed = nn.Embedding(num_phonemes, 64)
        self.ph_proj = nn.Linear(64, phoneme_dim)  # 64 → 256 投影
        self.phoneme_dim = phoneme_dim
        self.lag_predictor = nn.Sequential(
            nn.Linear(128 + 1 + 32, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )
        # dur 输入: ph_emb(256) + f0(1) + pe(32) + note_len(1) + ph_pos(1) = 291
        DUR_IN = phoneme_dim + 1 + 32 + 1 + 1
        self.dur_predictor = nn.Sequential(
            nn.Linear(DUR_IN, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def _segment_pool(self, mu_conv, f0, boundaries, device, phoneme_ids=None):
        """对 Conv1D 预处理后的 mu 按 boundaries 分段池化
        phoneme_ids: (B, P) — 每个音素段的 ID, 用于拼接触入, 仅 dur_predictor 需要
        """
        B, D, T = mu_conv.shape
        T_f0 = f0.shape[-1] if f0.dim() == 2 else (f0.shape[-2] if f0.dim() == 3 else f0.shape[-1])
        f0_1d = f0.unsqueeze(1) if f0.dim() == 2 else f0.view(B, 1, -1)
        use_ph = phoneme_ids is not None
        feats = []
        for b in range(B):
            pos = 0
            for seg_idx in range(boundaries.shape[1]):
                seg_len = int(boundaries[b, seg_idx].item())
                if seg_len <= 0:
                    continue
                end = min(pos + seg_len, T, T_f0)
                if end <= pos:
                    continue
                mu_seg = mu_conv[b:b+1, :, pos:end].mean(dim=-1).squeeze(0)
                f0_seg = f0_1d[b:b+1, :, pos:end].mean(dim=-1).squeeze(0)
                rel = seg_idx / max(1, boundaries.shape[1])
                pos_val = torch.tensor([rel], device=device).float()
                pe = torch.cat([torch.sin(pos_val * (2**i)) for i in range(16)] +
                               [torch.cos(pos_val * (2**i)) for i in range(16)])
                base_feat = torch.cat([mu_seg, f0_seg, pe])
                if use_ph and seg_idx < phoneme_ids.shape[1]:
                    ph_id = phoneme_ids[b, seg_idx].long()
                    ph_emb = self.ph_embed(ph_id)
                    base_feat = torch.cat([base_feat, ph_emb])
                feats.append(base_feat)
                pos = end
        if not feats:
            feat_dim = 128 + 1 + 32 + (64 if use_ph else 0)
            return torch.zeros(0, feat_dim, device=device)
        return torch.stack(feats, dim=0)

    def forward(self, mu, f0, durations, note_lengths, phoneme_ids=None, phoneme_emb=None):
        """
        mu: (B, 128, T) — 帧级条件特征 (仅 lag_predictor 使用)
        f0: (B, T) — 帧级 F0
        durations: (B, P) — GT 音素帧数 (用于 f0 池化和 note_ctx 对齐)
        note_lengths: (B, N) — 音符帧数
        phoneme_ids: (B, P) — 音素 ID
        phoneme_emb: (B, P, D) — 音素级嵌入 (与时长无关，用作 dur_predictor 核心输入)
        """
        B = mu.shape[0]
        device = mu.device
        P = durations.shape[1]
        N = note_lengths.shape[1]

        # Conv1D 预处理: 仅 lag_predictor 使用帧级 mu
        mu_conv = self.pre_conv(mu)
        note_feats = self._segment_pool(mu_conv, f0, note_lengths, device)

        # ——— dur_predictor: 音素级特征 (不依赖 mu 池化) ———
        # 1. F0 池化到音素级 (沿 GT 边界)
        f0_1d = f0.unsqueeze(1) if f0.dim() == 2 else f0.view(B, 1, -1)
        ph_f0 = torch.zeros(B, P, device=device)
        for b in range(B):
            pos = 0
            for p in range(P):
                dp = int(durations[b, p].item())
                if dp <= 0:
                    continue
                end = min(pos + dp, f0_1d.shape[-1])
                if end > pos:
                    ph_f0[b, p] = f0_1d[b, 0, pos:end].mean()
                pos = end

        # 2. 音素嵌入 + 位置编码 + note_ctx
        dur_feat_list = []  # list of (b, p, note_idx, feat_tensor)
        if phoneme_emb is not None:
            ph_emb = phoneme_emb  # (B, P, 256)
        elif phoneme_ids is not None:
            ph_emb = self.ph_proj(self.ph_embed(phoneme_ids))  # (B, P, 256)
        else:
            ph_emb = torch.zeros(B, P, self.phoneme_dim, device=device)

        # note_ctx: (note_len, ph_pos_in_note)
        note_cumsum = torch.cumsum(note_lengths.float(), dim=1)
        ph_cumsum = torch.cumsum(durations.float(), dim=1)

        for b in range(B):
            note_idx = 0
            prev_note_end = 0.0
            for p in range(P):
                dp = int(durations[b, p].item())
                if dp <= 0:
                    continue
                ph_end = ph_cumsum[b, p].item()
                while note_idx < N and note_cumsum[b, note_idx].item() < ph_end - 0.5:
                    prev_note_end = note_cumsum[b, note_idx].item()
                    note_idx += 1
                if note_idx >= N:
                    break
                nl = note_lengths[b, note_idx].item()
                ph_mid = ph_end - dp / 2.0
                ph_pos = max(0.0, min(1.0, (ph_mid - prev_note_end) / max(1.0, nl)))

                # 位置编码 (全局)
                rel = p / max(1, P)
                pos_val = torch.tensor([rel], device=device).float()
                pe = torch.cat([torch.sin(pos_val * (2**i)) for i in range(16)] +
                               [torch.cos(pos_val * (2**i)) for i in range(16)])

                # 拼接: ph_emb(D_ph) + f0(1) + pe(32) + note_len(1) + ph_pos(1)
                feat = torch.cat([
                    ph_emb[b, p],
                    ph_f0[b, p].unsqueeze(0),
                    pe,
                    torch.tensor([nl], device=device),
                    torch.tensor([ph_pos], device=device),
                ])
                dur_feat_list.append((b, p, note_idx, feat))

        lag_pred = torch.zeros(B, N, device=device)
        if note_feats.shape[0] > 0:
            lag_flat = self.lag_predictor(note_feats).squeeze(-1)
            idx = 0
            for b in range(B):
                for n in range(N):
                    nl = int(note_lengths[b, n].item())
                    if nl > 0 and idx < lag_flat.shape[0]:
                        lag_pred[b, n] = lag_flat[idx]
                        idx += 1

        # ——— dur_predictor: 可微分构建带有音符内归一化的 dur_pred ———
        # 论文 Eq(21): Σd̂_nk = L̂_n 强制音素时长之和=音符长度
        # 修复 v2: 使用矩阵乘法而非 in-place 赋值, 确保梯度传播
        dur_pred = torch.zeros(B, P, device=device)
        if dur_feat_list:
            n_valid = len(dur_feat_list)
            dur_feats_t = torch.stack([f for _, _, _, f in dur_feat_list], dim=0)
            dur_flat = self.dur_predictor(dur_feats_t).squeeze(-1)  # (n_valid,) + grad

            # 收集每个有效音素的 (b, p) 坐标
            b_vals = torch.tensor([b for b, _, _, _ in dur_feat_list], device=device)
            p_vals = torch.tensor([p for _, p, _, _ in dur_feat_list], device=device)
            note_vals = torch.tensor([ni for _, _, ni, _ in dur_feat_list], device=device)

            # 按 (b, note_idx) 分组归一化 (论文 Eq 21)
            # 先构建一个 (B, P) 的 flat 索引用于扩散
            flat_idx = b_vals * P + p_vals  # (n_valid,) 每个有效位置在 (B*P) 中的索引

            # 按 (b, note) 分组 → 逐音符归一化
            # 分组键: b * max_notes + note_idx
            max_notes = N
            group_key = b_vals * max_notes + note_vals

            # 找唯一组
            unique_keys, inverse, counts = torch.unique(group_key, return_inverse=True, return_counts=True)
            n_groups = len(unique_keys)

            # 对每个组归一化: dur_flat[group] = dur_flat[group] / sum(dur_flat[group]) * note_len
            # 用 scatter_add 求每组和
            group_sum = torch.zeros(n_groups, device=device, dtype=dur_flat.dtype)
            group_sum = group_sum.scatter_add(0, inverse, dur_flat)  # differentiable!
            group_sum = group_sum + 1e-8  # 防零

            # 获取每组的 note_length
            group_nl = torch.zeros(n_groups, device=device)
            first_in_group = torch.zeros(n_groups, dtype=torch.long, device=device)
            for g in range(n_groups):
                mask = inverse == g
                # 取组中第一个元素的 note_idx
                first_in_group[g] = mask.nonzero()[0]
                # 从 note_vals 取长度
                ni = note_vals[first_in_group[g]]
                # 找对应的 b
                bi = b_vals[first_in_group[g]]
                group_nl[g] = note_lengths[bi, ni].float()

            # 归一化: dur_norm = dur_flat / group_sum * group_nl
            dur_norm = dur_flat / group_sum[inverse] * group_nl[inverse]  # (n_valid,) + grad

            # 用矩阵乘法而不是 in-place scatter 来构建 (B, P) 输出
            # 创建稀疏分配矩阵 A: (B*P, n_valid) 中 A[flat_idx[i], i] = 1.0
            rows = flat_idx  # (n_valid,)
            cols = torch.arange(n_valid, device=device)
            A = torch.zeros(B * P, n_valid, device=device, dtype=dur_flat.dtype)
            A[rows, cols] = 1.0
            # dur_pred_flat = A @ dur_norm → (B*P,) 完全可微分!
            dur_pred = (A @ dur_norm).view(B, P)

        return lag_pred, dur_pred

    def predict_durations(self, mu, f0, note_lengths, phoneme_ids, durations=None):
        """推理用: 只预测音素时长，不依赖 ground-truth durations

        输入: mu (B,128,T), f0 (B,T), note_lengths (B,N), phoneme_ids (B,P)
              durations (B, P) — 初始时长 (GT), 用于推导音素到音符的映射关系
        返回: dur_pred (B, P) — 预测的每个音素帧数，已逐音符归一化
        """
        B = mu.shape[0]
        device = mu.device
        P = phoneme_ids.shape[1]

        # 从初始 durations 推导每个音符含几个音素
        # durations(B,P) 编码了音素→音符分组: 同一组内 durations 之和 = 该音符帧数
        if durations is not None:
            phonemes_per_note = torch.zeros(B, note_lengths.shape[1], dtype=torch.long, device=device)
            for b in range(B):
                ph_idx = 0
                for n in range(note_lengths.shape[1]):
                    nl = int(note_lengths[b, n].item())
                    if nl <= 0:
                        continue
                    frame_sum = 0
                    count = 0
                    while ph_idx < P and frame_sum < nl:
                        d = int(durations[b, ph_idx].item())
                        if d <= 0:
                            ph_idx += 1
                            continue
                        frame_sum += d
                        count += 1
                        ph_idx += 1
                    phonemes_per_note[b, n] = max(1, count)
        else:
            # fallback: 均匀分配
            num_notes = (note_lengths > 0).sum(dim=1).max().item()
            avg = max(1, P // max(1, num_notes))
            phonemes_per_note = torch.full((B, note_lengths.shape[1]), avg, dtype=torch.long, device=device)
            phonemes_per_note[note_lengths <= 0] = 0

        # 初始时长: 各音符内均匀分配到所有音素
        init_durs = torch.zeros(B, P, dtype=torch.long, device=device)
        for b in range(B):
            ph_idx = 0
            for n in range(note_lengths.shape[1]):
                nl = int(note_lengths[b, n].item())
                ph_count = int(phonemes_per_note[b, n].item())
                if nl <= 0 or ph_count <= 0:
                    continue
                base = nl // ph_count
                rem = nl % ph_count
                for j in range(ph_count):
                    if ph_idx < P:
                        init_durs[b, ph_idx] = base + (1 if j < rem else 0)
                        ph_idx += 1

        # 预测: init_durs 与 phoneme_ids 形状对齐 (B, P)
        _, dur_pred = self.forward(mu, f0, init_durs, note_lengths, phoneme_ids=phoneme_ids)

        # 逐音符归一化 Σd̂_nk = L̂_n
        dur_norm = dur_pred.clone()
        for b in range(B):
            ph_idx = 0
            for n in range(note_lengths.shape[1]):
                nl = int(note_lengths[b, n].item())
                if nl <= 0 or ph_idx >= P:
                    continue
                ph_count = int(phonemes_per_note[b, n].item())
                if ph_count <= 0:
                    continue
                end = min(ph_idx + ph_count, P)
                note_sum = dur_norm[b, ph_idx:end].sum() + 1e-8
                dur_norm[b, ph_idx:end] = dur_norm[b, ph_idx:end] / note_sum * nl
                ph_idx = end

        return dur_norm  # (B, P) 与 phoneme_ids 对齐

    def compute_loss(self, lag_pred, dur_pred, lag_gt, dur_gt, note_lengths):
        mask_dur = dur_gt > 0
        mask_lag = note_lengths > 0
        lag_loss = F.mse_loss(lag_pred[mask_lag], lag_gt[mask_lag])
        # 对数域 MSE: 关注相对比例而非绝对帧数
        # log(1+x) 防止 log(0) → 兼容归一化后可能出现的零值
        dur_loss = F.mse_loss(
            torch.log1p(dur_pred[mask_dur]),
            torch.log1p(dur_gt[mask_dur])
        )
        return lag_loss, dur_loss

    @staticmethod
    def normalize_per_note(dur_pred, note_boundaries, note_lengths, lag_pred):
        """
        论文 §3.3.1: 推理时逐音符归一化 Σd̂_nk = L̂_n
        note_lengths 在预处理阶段已存为 L̂_n = L_n - lag_n，无需再减
        """
        dur_normalized = dur_pred.clone()
        for n, (start, end) in enumerate(note_boundaries):
            lag_n = lag_pred[0, start].detach()
            L_hat = max(1, note_lengths[n])
            note_sum = dur_pred[0, start:end].sum() + 1e-8
            dur_normalized[0, start:end] = dur_pred[0, start:end] / note_sum * L_hat
        return dur_normalized


class F0Predictor(nn.Module):
    """
    独立 F0 预测器 — 从条件特征 μ 直接预测帧级 F0

    论文 §4.3.2: Flow matching 输出第 129 通道作为 F0_refined 的设计
    因 loss 权重低 + 目标冲突导致预测不准。改用独立网络:
      μ (B,128,T) → Conv1D × 3 → f0_pred (B,1,T)
    L1 loss 更稳定, 权重可设更高。
    """
    def __init__(self, cond_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cond_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, 1, 3, padding=1),
        )

    def forward(self, mu):
        """
        mu: (B, 128, T) — 条件特征
        returns: (B, 1, T) — 归一化 F0 [0, 1], 对应 [0, 2000] Hz
        """
        return torch.sigmoid(self.net(mu))  # 有界输出, 防溢出
