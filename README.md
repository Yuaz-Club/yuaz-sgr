# Yuaz SGR 1.2

Yuaz SGR (Singing Generative Refinement) — 基于 DDSP + **迭代残差精炼（IRR）** 的中文歌唱声音合成系统。

## 项目说明

本项目以 DDSP（Differentiable Digital Signal Processing）为声码器骨架，用**可监督的迭代残差精炼器**替代潜变量生成，实现高保真中文歌唱合成。

原始论文为本项目的参考基线；本版起生成器已**不再使用**论文的 Flow Matching，详见 `paper_diff.md`。

### 论文（原始参考）

> 论文标题: *Singing voice synthesis via latent flow matching and differentiable digital signal processing*
> 详见 `论文/paper15_full.txt`

## 本次更新（1.2）

### 内置模型

本版内置**从零训练 75k 步**的 IRR 模型 `checkpoint_75k_pure.pt`。
75k 位于收敛最好的一段；继续训练会让 `rvq` 码本 commitment 漂移、非周期度上升，
爆破音逐渐变噪（见 `TRAINING.md` §6.3）。

### 生成器由 Flow Matching 变更为 IRR

论文与旧版用 CFM 训练速度场、推理时 ODE 积分得到潜变量 `z`。V1.2版改为：

```
z₀ = μ
z_{k+1} = z_k + g(z_k, μ, k)      k = 0..K-1,  K = 4（训练时固定）
```

- `g`：3 层 1D 卷积 + 步数嵌入，约 1.34M 参数
- 没有噪声采样、 ODE、 CFM 损失；`z` 直接接受解码器梯度
- 推理只需 K 次小卷积，K 可在推理时调整（4 默认，8 细节更多）

IRR 使 `z` 端到端对齐了解码器与目标音频，经过我们的测试：

| 指标 | 224k Flow | 75k IRR |
| --- | --- | --- |
| 谱平坦度 SFM | 0.0268 / 0.0463 | **0.0253 / 0.0410** |
| 高频能量占比 HF>4k | 0.0702 / 0.1217 | 0.0686 / 0.1095 |

感官上：咬字更加清晰、整体更加自然。

### 其他改动

| # | 问题 | 修复 | 涉及 |
| --- | --- | --- | --- |
| 1 | 大约 28% 的 f0 标签饱和在 2006Hz | 进行峰值归一化，添加置信度门控，同时`fmax` 收紧到 1100Hz | `scripts/preprocess_opencpop_v2.py` |
| 2 | 擦音、塞擦音高频细节差 | 编码器新增 log-mel 表征 | `models/encoder.py` |
| 3 | 解码器无音素条件 | 新增 `decoder_cond` | `models/ddsp_decoder.py` |
| 4 | F0 无滑音/颤音结构 | F0Predictor v2和帧间平滑损失 | `models/conditioning.py` |
| 5 | joint 阶段合成路径损失溢出 NaN | 进行`z_pred` 钳制和各损失 `isfinite` 保护 | `train/trainer.py` |
| 6 | 包络分辨率不足 | 包络 mel 128提升至256 | `models/ddsp_decoder.py` |
| 7 | 生成结果抖动 | 生成器权重 EMA | `train/trainer.py` |
| 8 | 训练信号精度 | STFT 损失变为 fp32，增加mel 感知损失 | `train/trainer.py` |
| 9 | `lag` 、`note_lengths` 标签错位 | 按 (note, note_dur) 归并去重 | `scripts/preprocess_opencpop_v2.py` |
| 10 | 谐波支路宽带电 | Nyquist 软滚降、可学习谐波权重、时域谐波合成等开关 | `models/ddsp_decoder.py` |
| 11 | 爆破音前面多一段噪声 | 使用音素时长统计表和爆破音固定 32ms | `yuaz_sgr/synthesizer.py` |

### 推理与工程

| 新增 | 说明 |
| --- | --- |
| 谐波抗混叠 | 进行逐谐波累加和Nyquist 掩蔽，显存由 O(K·N) 变为 O(N) |
| 静音门控 | `f0=0` 处不产生谐波激励 |
| 分块解码 | 进行`max_chunk_frames` 分块并进行交叉淡化 |
| 清浊先验 | `voicing_prior=True`（默认关闭） |
| 软量化 | `rvq_residual_alpha` 回补 RVQ 连续残差 |
| UST 直读 | `infer_ust.py` 直接解析 Utau或OpenUtau 工程 |
| 检查点能力自动识别 | 读取 `flags`（`generator` / `irr_steps` / `decoder_cond` / `f0_predictor` / `encoder_feature` / `dec_mels`） |
| 断点续训 | `--resume auto`、`--max_hours`、`--keep_ckpts` |
| 训练进度面板 | 无 |

## 兼容性：不兼容旧版检查点

**本版内置 `checkpoint_75k_pure.pt`（从零训练），为 IRR 架构，与所有 Flow Matching 版检查点不兼容**（旧版权重没有 `refiner.*`，且生成器代码已移除）。

推理端读取检查点 `flags` 中的 `generator` / `irr_steps`，本版及以后有效。

## 项目结构

```
.
├── yuaz_sgr/                     # 核心训练代码
│   ├── models/                   # Encoder / RVQ / DDSP Decoder / IterativeRefiner / Conditioning / Discriminator
│   ├── train/                    # train.py / trainer.py / progress.py
│   ├── data/ scripts/ losses/ utils/
│   ├── config/                   # base.yaml / model.yaml / train.yaml
│   └── inference/                # 训练包内推理入口 (YuazSinger)
├── yuaz_sgr_inference/           # 推理包
│   ├── yuaz_sgr/                 # 模型代码 + synthesizer.py + assets/
│   ├── checkpoint_75k_pure.pt    # 内置模型（从零训练 75k，34.3MB）
│   ├── 示例_75k_默认.wav          # 默认参数试听
│   ├── infer_ust.py              # UST 工程推理
│   ├── infer_vsqx.py             # VSQX 推理
│   └── example.py                # 推理示例
├── 论文/                         # 原始论文
├── 示例.wav                      # 音频示例
├── paper_diff.md                 # 与论文的差异对照
└── README.md
```

## 模型架构

- **Encoder**：160 维的 log-mel 
-**RVQ**：Residual Vector Quantization（8 × 1024 × 128）
- **Conditioning Encoder**：音素 + 时长 + F0 → μ
- **Iterative Refiner**：`z₀=μ`，K 步残差精炼（1.34M）
- **DDSP Decoder**：谐波 + 噪声，weight_net 门控、Nyquist 软抗混叠、静音门控，包络 256 mel
- **F0Predictor v2**：从 μ + 阶梯 F0 预测帧级 F0（当前模型未收敛，推理请用 `use_predicted_f0=False`）

## 音频示例

> 歌曲：这世界那么多人 —— 本版模型（`checkpoint_75k_pure.pt`）合成，IRR K=4，使用统计表时长和爆破音 32ms推理

音频文件：`示例.wav`（约 4 分 15 秒）

## 推理性能

| 平台 | 配置 | RTF |
|------|------|-----|
| CPU（4 核） | K=4 | 0.1–0.2 |

IRR 每步只是一个小卷积，K 从 1 到 8 几乎不增加耗时。

## 预训练模型

| 模型 | 文件 | 大小 | 说明 |
|------|------|------|------|
| 基础模型 | `yuaz_sgr_inference/checkpoint_75k_pure.pt` | 34.3MB | 75k 步（IRR，K=4） |

## 推理

```bash
cd yuaz_sgr_inference/
pip install -r requirements.txt
python example.py            # 基础示例
python infer_ust.py 歌曲.ust  # UST 直接合成
```

```python
from yuaz_sgr import YuazSinger
singer = YuazSinger("checkpoint_75k_pure.pt", device="cuda")
wav = singer.sing(
    phoneme_ids, durations, f0_curve,
    irr_steps=4,             # 精炼步数: 4 默认, 8 细节更多
    use_predicted_f0=False,  # 保持 False：F0 预测器尚未收敛
    consonant_gain=0.5,      # 辅音增益
    f0_portamento_frames=4,  # 滑音帧数
    # 辅音时长默认走 assets/cons_dur_table.json（爆破音固定 32ms）
)
```

## 训练

配置见 `yuaz_sgr/config/`：

```bash
python train/train.py --config_dir config --data_dir <preprocessed> \
    --phoneme_map <preprocessed>/phoneme_map.json \
    --output_dir checkpoints_irr --resume auto
```

关键配置：`generator.irr_steps`（训练步数，默认 4）、`loss_weights.irr_steps`。

## 数据集

[OpenCPOP](https://wenet-e2e.github.io/opencpop/) 中文歌唱数据集（24kHz，3450/110/206 划分）。

## 开源协议

MIT License。

## 致谢

- [OpenCPOP](https://wenet-e2e.github.io/opencpop/)
- [DDSP](https://github.com/magenta/ddsp)
- [原始论文 *Singing voice synthesis via latent flow matching and DDSP*]([ceur-ws.org/Vol-4160/paper15.pdf](https://ceur-ws.org/Vol-4160/paper15.pdf))（作为基线参考）




