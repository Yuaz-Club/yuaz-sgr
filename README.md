# Yuaz SGR

Yuaz SGR (Singing Generative Refinement) — 基于 DDSP + Flow Matching 的歌唱声音合成系统。

## 项目说明

本项目基于 DDSP（Differentiable Digital Signal Processing）和 Flow Matching 技术，实现了高保真的中文歌唱声音合成。

**全球首个基于中文数据集完整复现并开源的项目。**

### 论文

> 论文标题: *Singing voice synthesis via latent flow matching and differentiable digital signal processing*
> 详见 `论文/paper15_full.txt`

## 项目结构

```
yuaz-sgr/
├── yuaz_sgr/                           # 核心训练代码
│   ├── models/                         # 模型定义 (Encoder, RVQ, DDSP, Flow Matching)
│   ├── train/                          # 训练脚本
│   ├── data/                           # 数据处理与 OpenCPOP 数据集
│   ├── scripts/                        # 预处理/推理脚本
│   ├── losses/                         # 损失函数
│   └── utils/                          # 工具函数
├── yuaz_sgr_inference/                 # 推理包 (轻量级，可独立部署)
│   ├── yuaz_sgr/                       # 模型代码
│   ├── checkpoint_300k.pt              # 预训练模型 (300K steps, 94MB)
│   ├── infer_vsqx.py                   # VSQX 文件推理脚本
│   └── example.py                      # 推理示例
├── yuaz_sgr_finetune_inference/        # 微调推理包
│   ├── yuaz_sgr/                       # 模型代码
│   ├── freeze_final.pt                 # 微调模型 (94MB)
│   ├── train_freeze.py                 # 微调脚本
│   └── example.py                      # 推理示例
├── 示例.wav                            # VSQX 推理输出示例
├── 论文/                               # 参考论文
├── README.md                           # 中文文档
└── README.en.md                        # English docs
```

## 模型架构

- **Encoder**: MFCC 特征提取 + CREPE F0 估计
- **RVQ**: Residual Vector Quantization (4 级量化)
- **DDSP Decoder**: 谐波+噪声合成，带 weight_net 门控
- **Conditioning Encoder**: 音素嵌入 + F0 → 条件向量 μ
- **Flow Matching Generator**: ODE 积分生成潜变量
- **F0Predictor**: 从 μ 独立预测精炼 F0

## 音频示例

<audio controls src="示例.wav">
  您的浏览器不支持 audio 标签，请下载 [示例.wav](示例.wav) 收听。
</audio>

音频直链：`https://github.com/user-attachments/files/30454901/default.wav`

> 歌曲: 这世界那么多人 (VSQX 调音版)，使用原始 F0 合成，n_steps=8

## 推理性能

在 **CPU (AMD Ryzen 5 3500U)** 上实测性能：

| 指标 | 数值 |
|------|------|
| 模型大小 | 94MB (提纯后仅含权重) |
| 推理速度 (n_steps=8) | RTF = 0.821 (4分15秒歌曲 → 3分29秒合成, 比实时快) |
| CPU 推理内存占用 | ~283 MB (加载+推理增量) |
| ODE 步数 | 支持 1~16 步, 8步即可获得高质量结果 |
| 采样率 | 24 kHz |

> RTF (Real-Time Factor) < 1 表示合成速度快于音频实际时长，即使在 CPU 上也能实现实时或超实时合成。

## 预训练模型

本仓库包含两个预训练模型：

| 模型 | 文件 | 大小 | 说明 |
|------|------|------|------|
| 基础模型 | `yuaz_sgr_inference/checkpoint_300k.pt` | 94MB | 300K 步训练，已提纯(仅保留权重) |
| 微调模型 | `yuaz_sgr_finetune_inference/freeze_final.pt` | 94MB | 基于基础模型微调，冻结解码器优化发音清晰度 |

## 推理

```bash
# 使用基础模型推理 (从音素合成)
cd yuaz_sgr_inference/
pip install -r requirements.txt
python example.py

# 使用微调模型推理
cd yuaz_sgr_finetune_inference/
pip install -r requirements.txt
python example.py
```

## 数据集

本项目使用 [OpenCPOP](https://github.com/yxlllc/OpenCPOP) 中文歌唱数据集进行训练。

## 开源协议

本项目遵循 MIT License，欢迎自由使用和修改。

## 致谢

- [OpenCPOP](https://github.com/yxlllc/OpenCPOP) 中文 singing 数据集
- DDSP: Differentiable Digital Signal Processing
- Flow Matching for Generative Modeling
