# YUAZ SGR — 从零训练教程

本文档介绍如何从零开始训练 YUAZ SGR 歌声合成模型。

## 1. 环境准备

### 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA GPU 16GB VRAM | NVIDIA A800 80GB |
| RAM | 32GB | 64GB |
| 存储 | 50GB | 100GB SSD |

### 软件环境

```bash
pip install -r yuaz_sgr/requirements.txt
```

Python >= 3.9, PyTorch >= 2.0。

## 2. 数据集准备

你需要一个包含以下文件的数据集（每首歌三个文件）：

```
your_dataset/
├── song_001.wav        # 人声干声, 单声道 (推荐 44.1kHz, 预处理会自动降采样到 24kHz)
├── song_001.TextGrid   # Praat TextGrid 格式, 含 phonemes 层
├── song_001.mid        # MIDI 伴奏/旋律 (含音符音高和起止时间)
├── song_002.wav
├── song_002.TextGrid
├── song_002.mid
└── ...
```

### TextGrid 格式要求

TextGrid 文件需要包含一个 `phones` 或 `phonemes` 层，标注每个音素的时间边界。例如：

```
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 30.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 30.5
        intervals: size = 10
        intervals [1]:
            xmin = 0.0
            xmax = 0.05
            text = "d"
        intervals [2]:
            xmin = 0.05
            xmax = 0.3
            text = "a"
        ...
```

### 数据集划分

代码默认按字母顺序划分：前 N 首 → 测试集，接下来 N 首 → 验证集，其余 → 训练集。

例如 5 首歌、`test_songs=1, val_songs=1`：
- 测试: song_001
- 验证: song_002
- 训练: song_003 ~ song_005

## 3. 数据预处理

预处理脚本将 WAV + TextGrid + MIDI 合并对齐、提取 F0 (CREPE)、分段为 4 秒片段、构建音素映射表：

```bash
python yuaz_sgr/scripts/preprocess_opencpop.py \
    --data_dir /path/to/your_dataset \
    --output_dir data/preprocessed \
    --test_songs 1 \
    --val_songs 1
```

**参数说明**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data_dir` | 必填 | 数据集根目录 (含 .wav, .TextGrid, .mid) |
| `--output_dir` | `data/preprocessed` | 预处理输出目录 |
| `--test_songs` | `3` | 测试集歌曲数 |
| `--val_songs` | `3` | 验证集歌曲数 |

**预处理做了什么**：
1. 音频降采样到 24kHz，-26dBFS 归一化
2. 用 CREPE 从音频提取帧级真实 F0 (50~2006 Hz)
3. TextGrid 音素与 MIDI 音符对齐，计算 time-lag 和 音素时长
4. 按 4 秒分段（词级边界对齐）
5. 构建音素映射表 (`phoneme_map.json`)
6. 输出 `.npz` 分段文件到 `{output_dir}/{train,val,test}/`

**输出目录结构**：
```
data/preprocessed/
├── phoneme_map.json
├── train/
│   ├── song_003_0000.npz
│   ├── song_003_0001.npz
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

每个 `.npz` 文件包含：
- `audio`: (T_samples,) float32
- `phonemes`: 音素字符串列表
- `durations`: 每音素帧数
- `f0`: CREPE 帧级 F0 曲线
- `lags`: 帧级 time-lag
- `note_lengths`: 每音符帧数

## 4. 训练

### 4.1 配置文件

训练需要三个配置文件（在 `yuaz_sgr/config/` 目录下）：

**`base.yaml`** — 声学参数：
```yaml
sample_rate: 24000
hop_length: 256          # 帧率 93.75 Hz
n_fft: 1024
n_mfcc: 30
segment_seconds: 4
normalize_db: -26
num_phonemes: 60
```

**`model.yaml`** — 模型架构：
```yaml
encoder:
  hidden_dim: 512
  latent_dim: 128
  n_mfcc: 30
  n_fft: 1024
  hop_length: 256

rvq:
  num_quantizers: 8
  codebook_size: 1024
  codebook_dim: 128
  commitment_weight: 1.0

ddsp:
  n_harmonics: 60
  fft_size: 2048
  hop_length: 512
  hidden_dim: 128
  emformer_num_layers: 4
  emformer_num_heads: 8
  emformer_ffn_dim: 512
  emformer_segment_length: 32
  emformer_left_context: 8
  emformer_right_context: 4

flow_matching:
  input_dim: 128
  output_dim: 129
  hidden_dim: 256
  num_blocks: 6
  time_dim: 64

conditioning:
  num_phonemes: 60       # 根据 phoneme_map.json 调整
  cond_dim: 128
  phoneme_dim: 256

temporal:
  cond_dim: 128

discriminator:
  mpd_periods: [2, 3, 5, 7, 11]
  msd_scales: 3
```

**`train.yaml`** — 训练超参数：
```yaml
optimizer:
  name: Adam
  lr: 1.0e-5
  betas: [0.8, 0.999]

batch_size: 8
total_steps: 200000
precision: fp16

stages:
  ae:
    end: 20000              # 阶段1: AE预训练 0~20k
  gan:
    end: 50000              # 阶段2: 引入判别器 20k~50k
  joint:
    end: 200000             # 阶段3: 全联合优化 50k~200k

loss_weights:
  stft: 1.0
  rvq: 0.5
  gan: 0.5
  fm: 1.0
  feature: 0.1
  lag: 0.02
  dur: 0.05

log_interval: 100
save_interval: 1000
eval_interval: 5000
```

### 4.2 启动训练

```bash
python yuaz_sgr/train/train.py \
    --config_dir yuaz_sgr/config \
    --data_dir data/preprocessed \
    --phoneme_map data/preprocessed/phoneme_map.json \
    --output_dir checkpoints \
    --device cuda
```

**参数说明**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config_dir` | `config` | 配置文件目录 |
| `--data_dir` | 必填 | 预处理后的数据目录 |
| `--phoneme_map` | 必填 | 音素映射表路径 |
| `--output_dir` | `checkpoints` | 模型保存目录 |
| `--device` | `cuda` | 训练设备 |
| `--resume` | `None` | 从 checkpoint 恢复 |
| `--log_dir` | `logs` | 日志目录 |

### 4.3 从检查点恢复训练

```bash
python yuaz_sgr/train/train.py \
    --config_dir yuaz_sgr/config \
    --data_dir data/preprocessed \
    --phoneme_map data/preprocessed/phoneme_map.json \
    --output_dir checkpoints \
    --resume checkpoints/model_128000.pt
```

## 5. 训练三阶段

论文采用三阶段渐近训练策略：

### 阶段 1 — AE 预训练 (0 ~ 20,000 步)

```
目标: 学习音频自重建
损失: L_STFT + L_RVQ

仅训练 Encoder + RVQ + DDSP Decoder。
模型学习将波形编码为潜变量、量化并解码回波形。
```

### 阶段 2 — GAN 训练 (20,000 ~ 50,000 步)

```
目标: 提升合成音质
损失: + L_GAN + L_FM

引入 MPD + MSD 判别器，用对抗损失提升波形自然度。
Feature Matching 损失稳定训练。
```

### 阶段 3 — 全联合优化 (50,000 ~ 200,000 步)

```
目标: 完整 SVS 系统
损失: + L_feature + L_lag + L_dur + L_CFM + L_F0

接入:
  - Conditioning Encoder (音素 → 条件特征 μ)
  - Flow Matching Generator (μ → 潜变量 z)
  - F0Predictor (μ → 精炼 F0)
  - TemporalPredictor (lag + duration)
  - Flow-through-Decoder (CFM 预测 → Decoder → STFT)

全部子网络联合优化。
```

## 6. 训练监控

### 日志输出

训练日志实时打印到终端和 `logs/training_log.jsonl`：

```
Step 012500 [joint]  rvq=0.382  stft=7.12  total=15.92  gan=2.21  fm=1.85
  feature=0.094  lag=0.0002  dur=0.27  cfm=0.96  f0_refined=0.031
  stft_flow=7.87  f0_pred=0.060  d_loss=2.98  [0h12<3h45 0.5it/s]
```

**关键指标解读**：
- `rvq`: K-means 量化损失 (越低越好)
- `stft`: 多分辨率频谱重建损失 (越低越好)
- `gan`: 生成器对抗损失
- `cfm`: Flow Matching CFM 损失
- `f0_pred`: F0预测器 L1 损失
- `dur`: 时长预测损失
- `d_loss`: 判别器损失 (过低 = 训练崩溃)
- `total`: 生成器总损失

### 验证评估

每 5,000 步自动运行验证集评估，输出 `val_*` 指标。最佳验证模型保存为 `best_val_model.pt`。

## 7. 检查点管理

检查点每 1,000 步保存一次：

```
checkpoints/
├── model_001000.pt
├── model_002000.pt
├── ...
├── model_200000.pt       # 最终模型
└── best_val_model.pt     # 最佳验证模型
```

每个检查点文件 (~263MB) 包含：
- `model`: 模型权重
- `optim_g` / `optim_d`: 优化器状态
- `scaler`: AMP GradScaler 状态
- `step`: 当前步数

## 8. 训练后推理

训练完成后，将 `model_200000.pt` 复制到推理包 `yuaz_sgr_inference/` 目录：

```bash
cp checkpoints/model_200000.pt ../yuaz_sgr_inference/checkpoint.pt

cd ../yuaz_sgr_inference
python example.py
```

推理包使用详见 [推理包 README](../yuaz_sgr_inference/README.md)。

## 9. 常见问题

### Q: GPU 内存不够

减小 `batch_size` 在 `train.yaml` 中。注意：
- batch_size=4 约需 8GB VRAM
- batch_size=8 约需 16GB VRAM
- batch_size=32 约需 60GB+ VRAM

### Q: 训练 loss 不下降

1. 检查数据预处理是否正确（F0 范围、音素映射）
2. 确认 `num_phonemes` 与 `phoneme_map.json` 一致
3. 尝试增大 AE 阶段步数（如 40k）
4. 检查数据集是否太少（建议至少 3 首）

### Q: 码本坍缩

如果 RVQ 码本使用率低（如 < 50%），检查 `rvq.py` 中：
- 初始化 std = `1.0 / sqrt(codebook_dim)`（约 0.088）
- 死码重启机制 ema_size < 2.0

### Q: 合成音频响度过大/削波

推理包已移除峰值归一化。如果仍有削波，检查：
- 数据预处理 normalize_db 参数
- DDSP 解码器输出幅值

## 10. 项目结构

```
训练/
├── README.md                          # 本教程
└── yuaz_sgr/
    ├── __init__.py
    ├── requirements.txt
    ├── config/
    │   ├── base.yaml             # 声学参数
    │   ├── model.yaml            # 模型超参数
    │   └── train.yaml            # 训练超参数
    ├── data/
    │   ├── dataset.py            # CPU 数据集类
    │   └── gpu_dataset.py        # GPU 预载入数据集
    ├── losses/
    │   ├── stft_loss.py          # 多分辨率 STFT 损失
    │   └── gan_loss.py           # LS-GAN + Feature Matching
    ├── models/
    │   ├── svs_model.py          # 顶层 SVSModel
    │   ├── encoder.py            # DDSP Encoder
    │   ├── rvq.py                # Residual VQ
    │   ├── ddsp_decoder.py       # DDSP 谐波振荡器
    │   ├── flow_matching.py      # Flow Matching ODE
    │   ├── conditioning.py       # 条件编码 + F0/Temporal 预测器
    │   └── discriminator.py      # MPD + MSD 判别器
    ├── train/
    │   ├── train.py              # 训练入口
    │   └── trainer.py            # 训练器 + 损失计算
    ├── utils/
    │   ├── audio.py              # 音频工具
    │   └── metrics.py            # 评估指标
    └── scripts/
        └── preprocess_opencpop.py  # 数据预处理
```

## 11. 引用

```
@article{yuaz_sgr,
  title={YUAZ SGR: An Open-Source Singing Voice Synthesis System},
  author={Yu, et al.},
  year={2025}
}
```
