# YUAZ SGR
"""
inference/synthesizer.py
推理管线 — 虚拟歌手合成器

论文 §3.2 + §3.3: 推理流程
  音素+F0 → Conditioning Encoder → μ
  → TemporalPredictor → lag_pred, dur_pred (per-note 归一化)
  → Flow Matching ODE (128-dim Euler) → (z, f0_refined)
  → DDSP Decoder → 24kHz 波形

修复 #19: Flow Matching ODE 仅积分 128-dim z
修复 #20: DDSP Decoder 使用 Flow Matching 精炼的 F0
修复 #17: 推理时逐音符归一化时长
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.svs_model import SVSModel
from models.conditioning import TemporalPredictor


class SingingVoiceSynthesizer:
    def __init__(self, checkpoint_path, config, device='cpu'):
        self.device = device

        # 加载模型
        self.model = SVSModel(config)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.model.to(device)

        print(f"Synthesizer 加载完成 (device={device})")

    @torch.no_grad()
    def synthesize(self, phoneme_ids, phoneme_durations, f0_curve,
                   note_boundaries=None, note_lengths=None, n_steps=1):
        """
        主合成接口

        Args:
            phoneme_ids: list of int, 音素 ID 序列
            phoneme_durations: list of int, 每音素帧数
            f0_curve: list/array of float, 帧级 F0 (Hz)
            note_boundaries: list of (start, end), 音符边界帧索引
            note_lengths: list of int, 原始音符长度 (帧)
            n_steps: Flow Matching Euler 步数 (1 或 8)

        Returns:
            waveform: np.ndarray (T_samples,), 24kHz 波形
        """
        ph = torch.LongTensor(phoneme_ids).unsqueeze(0).to(self.device)
        dur = torch.LongTensor(phoneme_durations).unsqueeze(0).to(self.device)
        f0 = torch.FloatTensor(f0_curve).unsqueeze(0).to(self.device)

        mu = self.model.cond_encoder(ph, dur, f0)

        z, f0_refined = self.model.flow_gen.sample(mu, n_steps=n_steps)

        waveform = self.model.ddsp_decoder(f0_refined, z)

        return waveform[0, 0].cpu().numpy()

    def save(self, path, waveform, sample_rate=24000):
        """保存为 WAV 文件"""
        import soundfile as sf
        sf.write(path, waveform, sample_rate)
        print(f"保存: {path}")


class VirtualSinger:
    """
    便捷接口 — 支持音素字符串 + 编辑器风格参数
    """
    def __init__(self, checkpoint_path, phoneme_map_path, config, device='cpu'):
        import json
        with open(phoneme_map_path, 'r') as f:
            mapping = json.load(f)
        self.p2i = mapping['p2i']

        self.synthesizer = SingingVoiceSynthesizer(
            checkpoint_path, config, device=device
        )

    def synthesize(self, phonemes, durations, f0_curve, n_steps=1,
                   vibrato=None, breathiness=0.1, volume=-6,
                   note_boundaries=None, note_lengths=None):
        """
        Args:
            phonemes: list of str, 音素列表
            durations: list of int, 每音素帧数
            f0_curve: np.array, 帧级 F0
            vibrato: dict {rate, depth, start_frame, end_frame}
            breathiness: float, 气声强度 (0~1)
            volume: float, 音量 (dB)
        """
        ph_ids = [self.p2i.get(p, 0) for p in phonemes]

        # 应用 vibrato (可选)
        if vibrato is not None:
            f0_curve = self._apply_vibrato(f0_curve, vibrato)

        waveform = self.synthesizer.synthesize(
            ph_ids, durations, f0_curve,
            note_boundaries=note_boundaries,
            note_lengths=note_lengths,
            n_steps=n_steps,
        )

        # 音量调整
        waveform = waveform * (10 ** (volume / 20))

        return waveform

    def _apply_vibrato(self, f0_curve, vibrato):
        """在指定帧范围施加颤音"""
        rate = vibrato.get('rate', 5.5)
        depth = vibrato.get('depth', 0.5)
        start = vibrato.get('start_frame', 0)
        end = vibrato.get('end_frame', len(f0_curve))

        t = np.arange(end - start) / (24000 / 256)  # 帧率 93.75 Hz
        mod = 1.0 + depth * np.sin(2 * np.pi * rate * t) / 100.0

        f0_curve = f0_curve.copy()
        f0_curve[start:end] = f0_curve[start:end] * mod
        return f0_curve

    def save(self, path, waveform, sample_rate=24000):
        import soundfile as sf
        sf.write(path, waveform, sample_rate)