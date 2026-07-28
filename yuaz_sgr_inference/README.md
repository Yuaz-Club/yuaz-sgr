# YUAZ SGR — 推理包

基于论文 *"YUAZ SGR: An Open-Source Singing Voice Synthesis System"* 的独立推理包。

## 架构概览

```
拼音/F0/MIDI 输入
    │
    ▼
ConditioningEncoder (音素 + F0 → μ)
    │
    ▼
FlowMatchingGenerator (ODE 采样 → 潜变量 z, 128维)
    │
    ├──► F0Predictor (μ → 精炼 F0)
    │
    ▼
ResidualVectorQuantizer (8层 EMA 码本量化)
    │
    ▼
DDSPDecoder (谐波振荡器 + 噪声路径 → 24kHz 波形)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要 Python >= 3.9, PyTorch >= 2.0.

### 2. 放置模型文件

将训练好的检查点 `.pt` 文件放入 `yuaz_sgr_inference/` 目录（或任意位置，加载时指定路径）。

```
yuaz_sgr_inference/
├── yuaz_sgr/              # 模型和推理代码
├── example.py             # 使用示例
├── requirements.txt
├── checkpoint_200000.pt   # ← 放在这里
└── README.md
```

### 3. 开始合成

```python
from yuaz_sgr import YuazSinger

# 加载模型
singer = YuazSinger("checkpoint_200000.pt", device="cuda")  # 或 device="cpu"

# 方式A: 从音素+时长+F0 直接合成
import numpy as np
phoneme_ids = [33, 19, 18, 6]      # n, i, h, ao (你好)
durations   = [3, 45, 3, 45]        # 每音素帧数 (帧率 93.75Hz)
f0_curve    = np.concatenate([
    np.full(48, 262),  # C4
    np.full(48, 330),  # E4
])

wav = singer.sing(phoneme_ids, durations, f0_curve, use_predicted_f0=True)
singer.save("output.wav", wav)

# 方式B: 从音素字符串合成
wav = singer.sing_from_phonemes(
    ["n", "i", "h", "ao"], durations, f0_curve
)

# 方式C: 从 MIDI + 歌词合成
wav = singer.sing_from_midi(
    "song.mid",
    lyrics="ni hao shi jie",
    output_path="output.wav",
    use_predicted_f0=True,
)
```

## API 参考

### `YuazSinger(checkpoint_path, config_dir=None, device="cpu")`

加载模型。

| 参数 | 类型 | 说明 |
|------|------|------|
| `checkpoint_path` | `str` | 检查点 `.pt` 文件路径 |
| `config_dir` | `str\|None` | 配置目录, 默认使用内置 `assets/` |
| `device` | `str` | `"cpu"` 或 `"cuda"` |

### `sing(phoneme_ids, durations, f0_curve, ...)`

核心合成接口。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `phoneme_ids` | `list[int]` | — | 音素 ID 序列, shape (P,) |
| `durations` | `list[int]` | — | 每音素帧数, shape (P,) |
| `f0_curve` | `list[float]` | — | 帧级 F0 (Hz), shape (T_frames,) |
| `note_lengths` | `list[int]\|None` | `None` | 每音符帧数, predict_durations 时必需 |
| `predict_durations` | `bool` | `False` | 是否用 TemporalPredictor 预测时长 |
| `use_predicted_f0` | `bool` | `True` | 是否用 F0Predictor 精炼 F0 |
| `n_steps` | `int` | `1` | Flow Matching ODE 步数 |

返回: `np.ndarray`, shape (T_samples,), dtype=float32, 24kHz.

### `sing_from_midi(midi_path, lyrics=None, output_path=None, ...)`

从 MIDI + 歌词合成。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `midi_path` | `str` | — | MIDI 文件路径 |
| `lyrics` | `str\|None` | `None` | 拼音歌词, 空格分隔; None 则自动填 "a" |
| `output_path` | `str\|None` | `None` | 输出 WAV 路径 |
| `consonant_frames` | `int` | `2` | 辅音最小帧数 |

### `sing_from_phonemes(phoneme_strs, durations, f0_curve, ...)`

从音素字符串合成（无需手动查 ID）。

### `lyrics_to_score(lyrics, note_duration_frames, f0_per_note)`

编程式构建音素序列/时长/F0。

### `save(path, waveform, sample_rate=24000)`

保存 WAV 文件（静态方法）。

## 音素映射

OpenCPOP 60音素中文拼音集。常见拼音到音素分解:

| 拼音 | 音素 | | 拼音 | 音素 |
|------|------|---|------|------|
| ni | n, i | | wo | w, o |
| hao | h, ao | | shi | sh, i |
| zhong | zh, ong | | guo | g, uo |
| xue | x, ve | | yuan | y, van |
| jian | j, ian | | qing | q, ing |

完整音素表见 `yuaz_sgr/assets/phoneme_map.json`.

## 三种推理模式

| 模式 | `use_predicted_f0` | `predict_durations` | 说明 |
|------|-------------------|---------------------|------|
| 原版 | `False` | `False` | 直接使用输入的 F0 和时长 |
| 预测 F0 | `True` | `False` | F0Predictor 从 μ 精炼 F0 |
| 预测全部 | `True` | `True` | TemporalPredictor + F0Predictor |

## 依赖

```
torch>=2.0.0
numpy>=1.24.0
librosa>=0.10.0
torchcrepe>=0.0.2
soundfile>=0.12.0
pyyaml>=6.0
pretty_midi>=0.2.10    # MIDI 文件解析 (仅 sing_from_midi 需要)
```

## 项目结构

```
yuaz_sgr_inference/
├── example.py                    # 使用示例
├── requirements.txt              # 依赖
├── README.md
└── yuaz_sgr/                     # 推理包
    ├── __init__.py               # 导出 YuazSinger
    ├── synthesizer.py            # 推理核心 API
    ├── assets/
    │   ├── base.yaml             # 声学参数
    │   ├── model.yaml            # 模型超参数
    │   └── phoneme_map.json      # 60音素映射表
    └── models/
        ├── svs_model.py          # 顶层 SVSModel
        ├── encoder.py            # DDSP Encoder (MFCC+CREPE)
        ├── rvq.py                # Residual Vector Quantizer
        ├── ddsp_decoder.py       # DDSP 解码器 (谐波+噪声)
        ├── flow_matching.py      # Flow Matching Generator
        ├── conditioning.py       # 条件编码器 + F0Predictor + TemporalPredictor
        └── discriminator.py      # 判别器 (推理时未使用)
```

## 引用

```
@article{yuaz_sgr,
  title={YUAZ SGR: An Open-Source Singing Voice Synthesis System},
  author={Yu, et al.},
  year={2025}
}
```
