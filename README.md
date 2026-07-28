# Yuaz SGR

Yuaz SGR (Singing Generative Refinement) — 基于 DDSP + Flow Matching 的歌唱声音合成系统。

## 项目说明

本项目基于 DDSP（Differentiable Digital Signal Processing）和 Flow Matching 技术，实现了高保真的中文歌唱声音合成。

### 论文

本项目基于以下论文复现（见 [论文/paper15.pdf](论文/paper15.pdf)）：

> **论文标题**: 详见 `论文/paper15.pdf`

## 项目结构

```
yuaz-sgr/
├── yuaz_sgr/                    # 核心训练代码
│   ├── models/                  # 模型定义 (Encoder, RVQ, DDSP, Flow Matching)
│   ├── train/                   # 训练脚本
│   ├── data/                    # 数据处理
│   ├── scripts/                 # 预处理/推理脚本
│   ├── losses/                  # 损失函数
│   └── utils/                   # 工具函数
├── yuaz_sgr_inference/          # 推理包 (轻量级，可独立部署)
│   └── checkpoint_300k.pt       # 最终提纯模型 (94MB)
├── yuaz_sgr_finetune_inference/ # 微调相关代码
├── 训练/                         # 训练配置和脚本
├── 论文/                         # 参考论文
└── vsqx/                        # VSQX 测试文件
```

## 模型架构

- **Encoder**: MFCC 特征提取 + CREPE F0 估计
- **RVQ**: Residual Vector Quantization (4 级量化)
- **DDSP Decoder**: 谐波+噪声合成，带 weight_net 门控
- **Conditioning Encoder**: 音素嵌入 + F0 → 条件向量 μ
- **Flow Matching Generator**: ODE 积分生成潜变量
- **F0Predictor**: 从 μ 独立预测精炼 F0

## 推理

```bash
cd yuaz_sgr_inference/
pip install -r requirements.txt
python example.py
```

## 开源协议

本项目遵循 MIT License，欢迎自由使用和修改。

## 致谢

- [OpenCPOP](https://github.com/yxlllc/OpenCPOP) 中文 singing 数据集
- DDSP: Differentiable Digital Signal Processing
- Flow Matching for Generative Modeling
