# 训练教程

本教程覆盖：数据预处理 → 训练 → 验证 → 导出 → 推理

> 本版配方 = **从零训练 IRR（OpenCPOP 全量）**

---

## 0. 环境

```bash
conda create -n yuaz python=3.10 -y
conda activate yuaz
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r yuaz_sgr_inference/requirements.txt
pip install torchcrepe pyworld librosa soundfile pretty_midi pyyaml
```

- 显存：batch 16（bf16）约需 30GB；不够降到 8。
- 磁盘：预处理数据约 1.6GB；每个检查点约 190MB。
- 速度参考：RTX 4080 SUPER、batch 16、bf16 ≈ **0.83 it/s** → 200k 步约 **67 小时**。

---

## 1. 数据预处理

### 1.1 OpenCPOP

```
data/opencpop/
└── segments/
    ├── wavs/                 # 2004000110.wav ...
    ├── train.txt / test.txt
    └── transcriptions.txt
```

### 1.2 运行预处理

```bash
cd yuaz_sgr
python scripts/preprocess_opencpop_v2.py \
    --data_dir ../data/opencpop \
    --output_dir ../data/preprocessed \
    --test_songs 3 --val_songs 3
```

输出：`train/ 3450 段`、`val/ 110 段`、`test/ 206 段`，每段约 4 秒，外加 `phoneme_map.json`。

每段 npz：`audio / f0 / phonemes / durations / lags / note_lengths`。

> F0 用 CREPE 重算（不用 MIDI 的平坦 F0），清辅音帧置 0；音频按 RMS 归一化到 −26 dB。

---

## 2. 训练配置

`yuaz_sgr/config/` 下三份配置：

| 文件 | 内容 |
|---|---|
| `base.yaml` | 数据、音频、路径 |
| `model.yaml` | 模型结构（encoder / rvq / ddsp / generator / conditioning） |
| `train.yaml` | 优化器、步数、阶段、损失权重 |

### 2.1 模型（model.yaml）

```yaml
generator:
  input_dim: 128
  irr_steps: 4          # 训练时固定的精炼步数
  irr_hidden: 384
  irr_layers: 3
  irr_max_steps: 8
ddsp:
  n_harmonics: 60
  fft_size: 2048
  hop_length: 512
  n_mels: 256
build_discriminator: true   # 训练必须 true；推理端可设 false
```

### 2.2 训练（train.yaml）：最终配方

```yaml
batch_size: 16
total_steps: 200000
precision: bf16              # 关键：不要用 fp16

optimizer:
  lr: 1.5e-4
  schedule: {warmup_steps: 1000, min_lr_ratio: 0.05}

stages:
  ae:    {start: 0,     end: 20000}    # AE 重建
  gan:   {start: 20000, end: 63000}    # 加判别器（40k 步足够，不用等到 80k）
  joint: {start: 63000, end: 200000}   # 全联合（含 IRR 生成）

loss_weights:
  ap: 0.0             # 实测开启(ap>0)反而更差，保持 0
  hf_spec: 0.3        # 合成路径 >4kHz 对数幅度（治高频过亮/电流音）
  hf_spec_ae: 0.3
  cons_energy: 1.0    # 清辅音能量监督（治辅音被吞）
  f0_loss: cents      # F0 预测用音分损失（务必）
  f0_pred: 0.5
  dur: 0.2
  irr_steps: 4
```

**阶段比例**：AE 10% / GAN 30% / JOINT 60%。AE 阶段别省（早期只给 4k 步时解码器明显欠训）。

---

## 3. 启动训练

```bash
cd yuaz_sgr
python train/train.py \
    --config_dir config \
    --data_dir ../data/preprocessed \
    --phoneme_map ../data/preprocessed/phoneme_map.json \
    --output_dir ../checkpoints \
    --keep_ckpts 8 \
    --log_dir ../logs
```

| 参数 | 说明 |
|---|---|
| `--resume auto` | 从 `--output_dir` 最新断点续训 |
| `--resume <ckpt>` | 从指定检查点续训 |
| `--max_hours 8` | 到时存盘退出（配合外层循环做长训练） |
| `--device cpu` | 无 GPU 调试（极慢） |

---

## 4. 训练指标

| 字段 | 含义 | 正常趋势 |
|---|---|---|
| `stft` / `mel` | AE 重建损失 | 稳定下降（4s 段约 5.8–6.0） |
| `stft_syn` / `mel_syn` | 合成路径（IRR）损失 | 下降；与 `stft` 差距越小越好 |
| `hf_spec_ae/syn` | >4kHz 对数幅度 | 0.04–0.08 |
| `cons` / `cons_syn` | 清辅音能量 | 0.1–0.25 |
| `f0_pred` | F0 预测（×100 音分） | 持续下降；>1.5 说明还没学会 |
| `rvq` | 码本 commitment | 0.02–0.3 |
| `val` | 验证集 | 与训练同步 |

> 旧日志里 `stft_syn` 、 `mel_syn` 叫 `stft_flow` 、 `mel_flow`

---

## 5. 导出与推理

```python
import torch
ck = torch.load('checkpoints/model_200000.pt', map_location='cpu', weights_only=False)
sd = ck.get('ema') or ck['model']
torch.save({'step': ck['step'], 'model': sd, 'flags': ck['flags']},
           'yuaz_sgr_inference/checkpoint_200000_pure.pt')
```

导出后约 34MB（无判别器/优化器）。

```bash
cd yuaz_sgr_inference
python example.py
python infer_ust.py 歌曲.ust --ckpt checkpoint_200000_pure.pt
```

GT 条件重建（输入原音频 F0 和真实音素时长）：

```bash
python scripts/syn_gt.py checkpoints/model_200000.pt config out_prefix 3
```

---

## 6. 你可能会踩的坑（务必看）

### 6.1 不要用 fp16

从零训练在 joint 阶段会出现激活溢出，表现为：

```
CUDA error: device-side assert triggered  (BCE 输入越界)
```

而且 NaN 会写进权重。**用 bf16**（范围同 fp32、速度同 fp16）。已 NaN 的检查点要回退到上一个干净存档。

### 6.2 磁盘

每个检查点约 190MB，务必定期清理。

### 6.3 训练到 ~105k 之后开始退化

使用Opencpop数据时，从零训练在 105k 附近听感最好，之后爆破音逐渐变成"噪声"。验证集指标能对上：

| step | stft_syn | rvq（码本 commitment） | ap_syn（非周期度） |
|---|---|---|---|
| 70k | 10.09 | **0.74** | 2.77 |
| 105k | 10.22 | 1.66 | 2.85 |
| 120k | 10.42 | **2.01** | 3.00 |
| 135k | 10.39 | 1.88 | **3.03** |

指标上升：

- `rvq` 上涨，码本漂移，量化残差变大，瞬态（爆破音）首当其冲；
- `ap_syn` 上涨，解码器非周期度（噪声占比）持续变大，听感就是"爆破音变为噪声"。

**对策**：
 **根据Tensorboard验证集的听感及时停止训练**，不要一路跑到 200k。

### 6.4 检查点务必留够

`--keep_ckpts 3` 会在你还没决定用哪版之前把旧检查点删掉。

- 全量训练建议 `--keep_ckpts 8`；
- 一旦听感满意，立刻单独复制一份：`cp model_105000.pt model_best_105k.pt`；
- TensorBoard 里的 `val/syn` 音频可以事后导出（见 `scripts/tb_extract.py`），但那只是声音，**权重无法恢复**。


---

## 7. 常见问题

**Q：训练崩了怎么恢复？**
使用`--resume auto` 取最新断点；若最新检查点含 NaN，回退一个存档再续。




