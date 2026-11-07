# 与论文的差异对照

- 论文：`论文/paper15_full.txt` — *Singing voice synthesis via latent flow matching and differentiable digital signal processing*
- 实现：YuazSGR 本仓库（v1.2 起生成器已改为 IRR）

---

## 一、YuazSGR 参考论文的地方

| 论文 | YuazSGR 实现 |
| --- | --- |
| §3.1.1 Encoder：CREPE 输出 F0、MFCC→GRU 输出 z，省略 loudness encoder | Conv1D×4 → GRU → proj，没有 loudness encoder |
| §3.1.3 Decoder：Emformer blocks，使用 F0 / S / A 三分量 | `_EmformerLayer`，`env_net` 输出 256 mel（S）、`ap_net` 输出 16 通道（A） |
| Eq(4)(5) `e_h=Σ(1/k)sin(φ_k)`，`φ_k=2πk∫F0` | `harmonic_oscillator` |
| Eq(6) `x_h=F⁻¹((1−A)S·F(e_h))`，STFT 2048、75% overlap | `fft_size=2048`、`hop=512`、1025 频点 |
| Eq(7)(8) `x_n=F⁻¹(AS·F(e_n))` | 一致 |
| Eq(9) `x̂=λ_h x_h+λ_n x_n` | weight_net 互补 sigmoid 门控 |
| Eq(10)(11) S 的 log-mel 压解压，`M₀=max(M⁻¹,0)` | `mel_basis` 256 mels；`mel_pinv_clamp=false` 不把伪逆负值截零 |
| Eq(12) 多分辨率 STFT：6 档、`2^(5+i)`、hop 25% | `fft_sizes=[64,128,256,512,1024,2048]`、`hop=s//4` |
| Eq(13) RVQ commitment | `rvq` 损失 |
| Eq(14)(15) MPD+MSD、最小二乘 GAN + FM | LS-GAN（MPD+MSD）+ FM |
| Eq(19) `L_feature=‖μ−z₁‖²` | `feature_loss` MSE |
| Eq(20)(22) lag 与 dur 均为 MSE | lag 用 MSE；dur 改为对数域 L1（长短音更均衡） |
| Eq(21) `Σ_k d̂_nk = L̂_n` 求和约束 | `normalize_per_note` 按 note_sum 归一 |
| §4.3.1 RVQ 8×1024×128 | `num_quantizers 8`、`codebook_size 1024`、`codebook_dim 128` |
| §4.1 24kHz、−26dB 归一、4 秒分段、词级边界切分 | 24kHz、−26dB、约 4 秒、在音素边界切分 |
| 损失权重 λ_stft 1.0、rvq 0.5、gan 0.5、fm 1.0、feature 0.1、lag 0.02 | 新增权重 mel 4.0、stft_syn_scale 2.0 |

## 二、YuazSGR 新增的地方

| 新增 | 说明 |
| --- | --- |
| **IRR 迭代残差精炼器** | 生成器：`z₀=μ, z_{k+1}=z_k+g(z_k,μ,k)`，K=4，约 1.34M 参数；取代 CFM 与 ODE 采样 |
| F0Predictor v1/v2 | 独立 F0 预测网络 |
| 清浊监督 | 对 `A` 与 `gate` 的 BCE 监督开关；重训置 `ap=0` |
| `f0_smooth` | F0 帧间平滑损失，权重 0.05 |
| log-mel 编码器表征 | 编码器可选用 log-mel 替代 MFCC |
| `decoder_cond` | 把 μ 送入 weight_net |
| Nyquist 抗混叠 | 谐波逐项累加并按 Nyquist 掩蔽 |
| 静音门控 | `f0=0` 处不产生谐波激励 |
| 分块解码 | 长音频分块解码并交叉淡化 |
| 软量化 | `rvq_residual_alpha` 回补 RVQ 连续残差 |
| 推理期清浊先验 | `voicing_prior` 由音素序列注入清浊掩码 |
| f0 预处理修复 | CREPE 峰值归一化、置信度门控、`fmax` 钳制 |
| lag / note_lengths 修复 | 按音节归并去重、`lag` 取辅音前置量 |
| 谐波软抗混叠 | Nyquist 顶部 10% 软滚降替代硬门限 |
| 生成器 EMA | 训练维护权重指数滑动平均，推理优先使用 |
| mel 感知损失 | 训练目标新增 log-mel L1（权重 4.0） |
| 判别器降频 | `gan_every=2` |
| 检查点精简 | 剥离优化器/scaler 并把 EMA 叠加进 `model`（34.3MB） |
| UST 直读 | `infer_ust.py` 解析 Utau/OpenUtau 工程 |

## 三、YuazSGR 改动论文的地方

| 项 | 论文 | YuazSGR |
| --- | --- | --- |
| **潜变量生成** | Flow Matching：CFM 训练速度场 + ODE 积分（Eq.17/18） | **IRR**：无 CFM、ODE、噪声采样，K 步残差直接沿解码器梯度优化 |
| 训练调度 | AE 0–20k → 加判别器 20k–200k → 全联合 200k–250k | AE 0–20k → GAN 20k–63k → JOINT 63k–200k（200k 步 @batch16，bf16） |
| `λ_dur` | 0.02 | 0.2 |
| 数据集 | Tohoku Kiritan（日语） | OpenCPOP（中文） |
| A 插值目标维度 | §3.1.3 s 513（fft 1024） | 1025（fft 2048） |
| 包络维度 | 论文的 mel 谱 | 解码器包络 256 mel；编码器 z 输入 log-mel 160 维 |
| 参数量 | 报告 18.6M | 实测 G 8.6M（含 IRR 1.34M）、D 4.6M |

## 四、YuazSGR 舍弃的地方

- **Flow Matching 整体移除**：`flow_matching.py` 与 CFM 训练路径已从代码中删除；旧版检查点不再兼容。
- flow 第 129 通道的 F0 输出随之消失，F0 改由独立的 F0Predictor 负责。
- loudness encoder 保持省略；另加了更轻的 `use_energy` 可选条件。

## 五、小结

骨架与公式层面仍大量参考论文：Encoder、Decoder、RVQ、时值模型、损失公式与权重。

最大的差异是**生成器**：论文用 latent flow matching，本版用可监督的迭代残差精炼（IRR）。实测在同一前端下，IRR 的噪声感更低、咬字更清晰；推理 RTF 约 0.1–0.2（4 核 CPU）。


