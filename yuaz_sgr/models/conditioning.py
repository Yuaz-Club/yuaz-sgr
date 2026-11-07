# YUAZ SGR
"""models/conditioning"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditioningEncoder(nn.Module):
    def __init__(self, num_phonemes, phoneme_dim=256, cond_dim=128, use_energy=False):
        super().__init__()
        # use_energy: 额外接收 1 维帧级能量
        self.use_energy = use_energy
        in_ch = phoneme_dim + 1 + (1 if use_energy else 0)
        self.phoneme_embed = nn.Embedding(num_phonemes, phoneme_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 256, 5, padding=2),
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

    def forward(self, phoneme_ids, durations, f0, energy=None):
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
        feats = [ph_frames, f0_in]

        if self.use_energy:
            # 帧级能量（归一化 RMS）
            if energy is None:
                e_in = torch.zeros_like(f0_in)
            else:
                e_in = energy.unsqueeze(-1) if energy.dim() == 2 else energy
                if e_in.shape[-2] != f0_in.shape[-2]:
                    e_in = F.interpolate(e_in.transpose(1, 2).float(),
                                         size=f0_in.shape[-2], mode='linear').transpose(1, 2)
            feats.append(e_in.to(f0_in.dtype))

        x = torch.cat(feats, dim=-1)
        x = x.transpose(1, 2)
        mu = self.encoder(x)
        return mu

class TemporalPredictor(nn.Module):
    """论文 §3"""
    def __init__(self, cond_dim=128, num_phonemes=60, phoneme_dim=256):
        super().__init__()
        # Conv1D 预处理器
        self.pre_conv = nn.Sequential(
            nn.Conv1d(cond_dim, 128, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, padding=1),
        )
        # 音素嵌入
        self.ph_embed = nn.Embedding(num_phonemes, 64)
        self.ph_proj = nn.Linear(64, phoneme_dim)  # 64 → 256 投影
        self.phoneme_dim = phoneme_dim
        self.lag_predictor = nn.Sequential(
            nn.Linear(128 + 1 + 32, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )
        # dur 输入维度 291
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
        """对 Conv1D 预处理后的 mu 按 boundaries 分段池化 (向量化) 返回: (feats, valid) — feats 为 (n_valid, D+33[+64]) 的有效段特征, valid 为 (B,N) 掩码, 行主序与 feats 一致"""
        B, D, T = mu_conv.shape
        N = boundaries.shape[1]
        f0_1d = f0.unsqueeze(1) if f0.dim() == 2 else f0.view(B, 1, -1)
        T_e = min(T, f0_1d.shape[-1])

        bl = boundaries.float()
        ends = torch.cumsum(bl, dim=1)
        starts = ends - bl
        e = ends.clamp(max=T_e)
        s = starts.clamp(max=T_e)
        seg_len = (e - s).clamp(min=0)
        valid = (bl > 0) & (seg_len > 0)

        cmu = F.pad(mu_conv.cumsum(-1), (1, 0))
        cf0 = F.pad(f0_1d.cumsum(-1), (1, 0))
        ie = e.long().view(B, 1, N).expand(B, D, N)
        is_ = s.long().view(B, 1, N).expand(B, D, N)
        denom = seg_len.clamp(min=1).view(B, 1, N)
        mu_mean = (cmu.gather(2, ie) - cmu.gather(2, is_)) / denom
        f0_mean = (cf0.gather(2, ie[:, :1]) - cf0.gather(2, is_[:, :1])) / denom[:, :1]

        freqs = 2.0 ** torch.arange(16, device=device).float()
        rel = torch.arange(N, device=device).float() / max(1, N)
        ang = rel.view(N, 1) * freqs.view(1, 16)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1).unsqueeze(0).expand(B, N, 32)

        feats = torch.cat([mu_mean.permute(0, 2, 1), f0_mean.permute(0, 2, 1), pe], dim=-1)
        feats = feats.reshape(B * N, D + 33)
        if phoneme_ids is not None and phoneme_ids.shape[1] >= N:
            ph = self.ph_embed(phoneme_ids.long()[:, :N]).reshape(B * N, 64)
            feats = torch.cat([feats, ph], dim=-1)
        return feats[valid.reshape(-1)], valid

    def forward(self, mu, f0, durations, note_lengths, phoneme_ids=None, phoneme_emb=None):
        """mu: (B, 128, T) — 帧级条件特征 (仅 lag_predictor 使用) f0: (B, T) — 帧级 F0 durations: (B, P) — GT 音素帧数 (用于 f0 池化和 note_ctx 对齐) note_lengths: (B, N) — 音符帧数 phoneme_ids: (B, P) — 音素 ID phoneme_emb: (B, P, D) — 音素级嵌入 (与时长无关，用作 dur_predictor 核心输入)"""
        B = mu.shape[0]
        device = mu.device
        P = durations.shape[1]
        N = note_lengths.shape[1]

        # Conv1D 预处理（lag_predictor 用）
        mu_conv = self.pre_conv(mu)
        note_feats, note_valid = self._segment_pool(mu_conv, f0, note_lengths, device)

        f0_1d = f0.unsqueeze(1) if f0.dim() == 2 else f0.view(B, 1, -1)
        Tf = f0_1d.shape[-1]

        # F0 池化到音素级
        dp = durations.float()
        ends_p = torch.cumsum(dp, dim=1)
        starts_p = ends_p - dp
        e_p = ends_p.clamp(max=Tf)
        s_p = starts_p.clamp(max=Tf)
        len_p = (e_p - s_p).clamp(min=0)
        cf0 = F.pad(f0_1d.cumsum(-1), (1, 0))
        ie_p = e_p.long().view(B, 1, P)
        is_p = s_p.long().view(B, 1, P)
        ph_f0 = (cf0.gather(2, ie_p) - cf0.gather(2, is_p)).squeeze(1) / len_p.clamp(min=1)

        # 音素嵌入
        if phoneme_emb is not None:
            ph_emb = phoneme_emb  # (B, P, 256)
        elif phoneme_ids is not None:
            ph_emb = self.ph_proj(self.ph_embed(phoneme_ids))  # (B, P, 256)
        else:
            ph_emb = torch.zeros(B, P, self.phoneme_dim, device=device)

        # note_ctx: (note_len, ph_pos_in_note)
        note_cum = torch.cumsum(note_lengths.float(), dim=1)
        note_idx = torch.searchsorted(note_cum.contiguous(), (ends_p - 0.5).contiguous(), right=False)
        note_ok = note_idx < N
        ni = note_idx.clamp(max=N - 1)
        nlt = torch.gather(note_lengths.float(), 1, ni)
        prev_all = torch.cat([torch.zeros(B, 1, device=device), note_cum[:, :-1]], dim=1)
        prev_end = torch.gather(prev_all, 1, ni)
        ph_mid = ends_p - dp / 2.0
        ph_pos = ((ph_mid - prev_end) / nlt.clamp(min=1)).clamp(0.0, 1.0)
        valid_ph = (durations > 0) & note_ok

        freqs = 2.0 ** torch.arange(16, device=device).float()
        rel_p = torch.arange(P, device=device).float() / max(1, P)
        ang_p = rel_p.view(P, 1) * freqs.view(1, 16)
        pe_p = torch.cat([torch.sin(ang_p), torch.cos(ang_p)], dim=1).unsqueeze(0).expand(B, P, 32)

        dur_feats = torch.cat([
            ph_emb,
            ph_f0.unsqueeze(-1),
            pe_p,
            nlt.unsqueeze(-1),
            ph_pos.unsqueeze(-1),
        ], dim=-1)
        flat_valid = valid_ph.reshape(-1)
        dur_feats_t = dur_feats.reshape(B * P, -1)[flat_valid]

        lag_pred = torch.zeros(B, N, device=device)
        if note_feats.shape[0] > 0:
            lag_flat = self.lag_predictor(note_feats).squeeze(-1)
            lag_pred = torch.zeros(B, N, device=device, dtype=lag_flat.dtype)
            lag_pred[note_valid] = lag_flat

        # dur_predictor：Eq(21) Σd̂_nk = L̂_n
        dur_pred = torch.zeros(B, P, device=device)
        n_valid = dur_feats_t.shape[0]
        if n_valid > 0:
            dur_flat = self.dur_predictor(dur_feats_t).squeeze(-1)  # (n_valid,) + grad

            ar_b = torch.arange(B, device=device).view(B, 1).expand(B, P).reshape(-1)[flat_valid]
            ar_p = torch.arange(P, device=device).view(1, P).expand(B, P).reshape(-1)[flat_valid]
            note_vals = ni.reshape(-1)[flat_valid]

            flat_idx = ar_b * P + ar_p
            group_key = ar_b * N + note_vals

            unique_keys, inverse, counts = torch.unique(
                group_key, return_inverse=True, return_counts=True)
            n_groups = unique_keys.shape[0]

            group_sum = torch.zeros(n_groups, device=device, dtype=dur_flat.dtype)
            group_sum = group_sum.index_add(0, inverse, dur_flat) + 1e-8  # differentiable

            first_idx = torch.full((n_groups,), n_valid, dtype=torch.long, device=device)
            first_idx = first_idx.scatter_reduce(
                0, inverse, torch.arange(n_valid, device=device), reduce='amin')
            group_nl = note_lengths.float()[ar_b[first_idx], note_vals[first_idx]]

            dur_norm = dur_flat / group_sum[inverse] * group_nl[inverse]  # (n_valid,) + grad
            dur_pred = torch.zeros(B * P, device=device, dtype=dur_flat.dtype).index_add(
                0, flat_idx, dur_norm).view(B, P)

        return lag_pred, dur_pred

    def predict_durations(self, mu, f0, note_lengths, phoneme_ids, durations=None):
        """推理用: 只预测音素时长，不依赖 ground-truth durations 输入: mu (B,128,T), f0 (B,T), note_lengths (B,N), phoneme_ids (B,P) durations (B, P) — 初始时长 (GT), 用于推导音素到音符的映射关系 返回: dur_pred (B, P) — 预测的每个音素帧数，已逐音符归一化"""
        B = mu.shape[0]
        device = mu.device
        P = phoneme_ids.shape[1]

        # 由 durations 推导音符内的音素分组
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
            # 均匀分配
            num_notes = (note_lengths > 0).sum(dim=1).max().item()
            avg = max(1, P // max(1, num_notes))
            phonemes_per_note = torch.full((B, note_lengths.shape[1]), avg, dtype=torch.long, device=device)
            phonemes_per_note[note_lengths <= 0] = 0

        # 初始时长（音符内均匀分配）
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

        # 预测 dur（B, P）
        _, dur_pred = self.forward(mu, f0, init_durs, note_lengths, phoneme_ids=phoneme_ids)

        # 逐音符归一化
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
        # 对数域 MSE（log1p 防 log(0)）
        # 对数域 L1（对长音/短音更均衡，避免被长音主导）
        dur_loss = F.l1_loss(
            torch.log1p(dur_pred[mask_dur]),
            torch.log1p(dur_gt[mask_dur])
        )
        return lag_loss, dur_loss

    @staticmethod
    def normalize_per_note(dur_pred, note_boundaries, note_lengths, lag_pred):
        """论文 §3"""
        dur_normalized = dur_pred.clone()
        for n, (start, end) in enumerate(note_boundaries):
            lag_n = lag_pred[0, start].detach()
            L_hat = max(1, note_lengths[n])
            note_sum = dur_pred[0, start:end].sum() + 1e-8
            dur_normalized[0, start:end] = dur_pred[0, start:end] / note_sum * L_hat
        return dur_normalized

class F0Predictor(nn.Module):
    """独立 F0 预测器 — 从条件特征 μ 直接预测帧级 F0 F0 预测器设计（论文用 flow 第 129 通道，本版独立成网） 因 loss 权重低 + 目标冲突导致预测不准"""
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
        """mu: (B, 128, T) — 条件特征 returns: (B, 1, T) — 归一化 F0 [0, 1], 对应 [0, 2000] Hz"""
        return torch.sigmoid(self.net(mu))  # 有界输出, 防溢出

class F0PredictorV2(nn.Module):
    """增强 F0 精修器 (可选, 修复 #4: 音高表现力) v1 的不足: 只有 3 层 k=3 卷积 (感受野 ~7 帧 ≈ 75ms), 且没有任何节奏/位置信息, 无法生成颤音、滑音、连音等需要"时间结构"的基频细节 → 歌声的音高偏机械"""
    def __init__(self, cond_dim=128, hidden_dim=192, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.input_proj = nn.Conv1d(cond_dim + 3, hidden_dim, 5, padding=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2 * d, dilation=d),
            ) for d in dilations
        ])
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.out = nn.Conv1d(hidden_dim, 1, 3, padding=1)

    def forward(self, mu, f0_step=None, note_pos=None, frame_pos=None):
        """mu: (B,128,T) → (B,1,T) 归一化 F0 (对应 0~2000 Hz, 训练目标=f0/2000)"""
        B, _, T = mu.shape
        dev = mu.device

        def _prep(x, is_f0=False):
            if x is None:
                return torch.zeros(B, 1, T, device=dev, dtype=mu.dtype)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if x.shape[-1] != T:
                x = F.interpolate(x.float(), size=T, mode='linear')
            x = x.to(mu.dtype)
            if is_f0:
                x = x / 2000.0
            return x

        h = self.input_proj(torch.cat(
            [mu, _prep(f0_step, True), _prep(note_pos), _prep(frame_pos)], dim=1))
        for blk in self.blocks:
            h = h + blk(h)
        h = self.norm(h)
        return self.out(h)

def build_note_positions(note_lengths, T, device):
    """由音符帧数构造 (note_pos, frame_pos), 形状均为 (B,1,T), 归一化到 [0,1) note_pos : 当前帧在其所属音符内的相对位置 (音符起点=0, 终点→1) — 颤音相位参考 frame_pos: 全局相对位置 (整段起点=0, 终点→1) — 长程走向参考 note_lengths: (B,N) 每音符帧数; 为 0 的音符跳过"""
    B = note_lengths.shape[0]
    note_pos = torch.zeros(B, 1, T, device=device)
    frame_pos = (torch.arange(T, device=device).float() / max(1, T - 1)
                 ).view(1, 1, T).expand(B, 1, T).contiguous()
    for b in range(B):
        pos = 0
        for n in range(note_lengths.shape[1]):
            nl = int(note_lengths[b, n].item())
            if nl <= 0:
                continue
            end = min(pos + nl, T)
            if end <= pos:
                break
            note_pos[b, 0, pos:end] = (
                torch.arange(end - pos, device=device).float() / max(1, nl))
            pos = end
            if pos >= T:
                break
    return note_pos, frame_pos
