# YUAZ SGR — 推理包

基于 *"YUAZ SGR"* 的独立推理包。

## 架构概览

```
拼音/F0/MIDI 输入
    │
    ▼
ConditioningEncoder (音素 + F0 → μ)
    │
    ▼
IterativeRefiner (K 步残差精炼 → 潜变量 z, 128维)
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
├── checkpoint_75k_pure.pt      # 从零训练 75k（内置模型，听感最自然）
├── 示例_75k_默认.wav            # 默认配置试听（统计表时长 + 爆破音 32ms）
└── README.md
```

> 旧的 264k 微调版权重（`checkpoint_264000_pure.pt`）体积相同、音色更暗，
> 已不再随包附带；需要时从发布页单独下载，放到本目录即可用
> `YuazSinger("checkpoint_264000_pure.pt")` 加载。

> `示例_75k_默认.wav` 由 `checkpoint_75k_pure.pt` + 默认参数合成
> （《这世界那么多人》，343 音符 / 255 秒），可作为音质基准。

### 3. 开始合成

> **默认行为**：从 MIDI/UST 合成时，辅音时长由训练数据统计表
> `yuaz_sgr/assets/cons_dur_table.json` 自动决定；爆破音（b/d/g/p/t/k）
> 固定 **32ms**。不需要额外传参。

```python
from yuaz_sgr import YuazSinger

# 加载模型
singer = YuazSinger("checkpoint_75k_pure.pt", device="cuda")  # 或 device="cpu"

# 方式A: 从音素+时长+F0 直接合成
import numpy as np
phoneme_ids = [33, 19, 18, 6]      # n, i, h, ao (你好)
durations   = [12, 36, 12, 36]      # 每音素帧数 (帧率 93.75Hz; 辅音12/元音36 实测最优)
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
| `irr_steps` | `int` | `4` | IRR 迭代精炼步数 (4 默认, 8 细节更多) |

返回: `np.ndarray`, shape (T_samples,), dtype=float32, 24kHz.

### `sing_from_midi(midi_path, lyrics=None, output_path=None, ...)`

从 MIDI + 歌词合成。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `midi_path` | `str` | — | MIDI 文件路径 |
| `lyrics` | `str\|None` | `None` | 拼音歌词, 空格分隔; None 则自动填 "a" |
| `output_path` | `str\|None` | `None` | 输出 WAV 路径 |
| `cons_dur_table` | `dict\|None` | 自动加载 | 音素时长统计表; 默认读 `assets/cons_dur_table.json` |
| `cons_seconds` | `dict\|None` | 爆破音 0.032 | 按秒覆盖辅音时长, 如 `{"b": 0.032, "p": 0.04}` |
| `plosive_frames` | `int` | `0` | 爆破音帧数覆盖 (0=用 `cons_seconds`) |
| `consonant_frames` | `int\|'auto'` | `12` | 无统计表时的兜底帧数 |

### 音素时长统计表（默认启用）

`assets/cons_dur_table.json` 由训练数据统计而来，每个音素记录：

| 字段 | 含义 | 用途 |
|---|---|---|
| `ratio_p50` | 占所属音符时长的中位数比例 | 合成时作为目标时长 |
| `ratio_p90` | 90 分位 | 合成时作为占比上限 |
| `frames_p90` | 绝对帧数 90 分位 | 绝对上限，防止长音把辅音拉长 |

分配规则：`辅音帧 = round(音符帧数 × ratio_p50)`，再夹到
`min(round(音符帧数 × ratio_p90), frames_p90)`；爆破音走 32ms 固定值。

用你自己的数据重新生成：

```bash
cd yuaz_sgr
python scripts/build_cons_dur_table.py \
    --data_dir ../data/preprocessed/train \
    --output ../yuaz_sgr_inference/yuaz_sgr/assets/cons_dur_table.json
```

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

## 常见问题

**首次运行卡在 import librosa / 模型构建阶段（Windows）**

librosa 的 numba 缓存默认写在 site-packages 里；如果 AppData 被 360/OneDrive 之类重定向到慢路径，numba 生成缓存时会卡住。把缓存目录指到普通本地目录即可：

`powershell
$env:NUMBA_CACHE_DIR = "D:\numba_cache"
python example.py
`

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
    │   ├── phoneme_map.json      # 60音素映射表
    │   └── cons_dur_table.json   # 音素时长统计表（默认启用）
    └── models/
        ├── svs_model.py          # 顶层 SVSModel
        ├── encoder.py            # DDSP Encoder (MFCC+CREPE)
        ├── rvq.py                # Residual Vector Quantizer
        ├── ddsp_decoder.py       # DDSP 解码器 (谐波+噪声)
        ├── iterative_refiner.py   # 迭代残差精炼器 (IRR)
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
