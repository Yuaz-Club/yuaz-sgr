#!/bin/bash
# YUAZ SGR 训练启动脚本
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 激活 Conda 环境
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/bin/activate

# 路径
YUAZ_DIR="/root/autodl-tmp/yuaz_sgr"
OPENCPOP_DIR="/root/autodl-tmp/opencpop"
PREPROCESSED_DIR="$YUAZ_DIR/data/preprocessed"
OUTPUT_DIR="/root/autodl-tmp/checkpoints"
LOG_DIR="/root/autodl-tmp/logs"
CONFIG_DIR="$YUAZ_DIR/config"

mkdir -p "$PREPROCESSED_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# 清理旧的测试数据
rm -rf "$PREPROCESSED_DIR/train" "$PREPROCESSED_DIR/val" "$PREPROCESSED_DIR/test" "$PREPROCESSED_DIR/phoneme_map.json"
# 移除 segments 目录避免干扰
mv "$OPENCPOP_DIR/segments" "$OPENCPOP_DIR/segments.bak" 2>/dev/null

cd "$YUAZ_DIR"

echo "=========================================="
echo "YUAZ SGR - OpenCPOP 训练"
echo "=========================================="
echo "时间: $(date)"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# ── 步骤1: 预处理 ──
echo ">> [1/3] 开始预处理 OpenCPOP 数据集..."
python scripts/preprocess_opencpop_full.py \
    --data_dir "$OPENCPOP_DIR" \
    --output_dir "$PREPROCESSED_DIR" \
    --test_songs 5 \
    --val_songs 5 2>&1 | tee "$LOG_DIR/preprocess.log"

echo ""
echo "预处理完成: $(date)"
ls -R "$PREPROCESSED_DIR" | head -20
echo ""

# ── 步骤2: 数据检查 ──
echo ">> [2/3] 数据验证..."
python -c "
import json, os
from pathlib import Path

pp_dir = Path('$PREPROCESSED_DIR')
for split in ['train', 'val', 'test']:
    split_dir = pp_dir / split
    if split_dir.exists():
        files = list(split_dir.glob('*.npz'))
        print(f'  {split}: {len(files)} 段')
    else:
        print(f'  {split}: 不存在')

# 检查音素映射
pm = json.load(open(str(pp_dir / 'phoneme_map.json')))
print(f'  音素数: {pm[\"num_phonemes\"]}')
print(f'  音素表: {list(pm[\"p2i\"].keys())[:20]}...')

# 检查样本
if list((pp_dir / 'train').glob('*.npz')):
    import numpy as np
    sample = np.load(list((pp_dir / 'train').glob('*.npz'))[0])
    print(f'  样本 keys: {list(sample.keys())}')
    print(f'  音频长度: {len(sample[\"audio\"])} samples')
"

# ── 步骤3: 启动训练 ──
echo ""
echo ">> [3/3] 启动训练..."
echo ""

# 重新创建模型 + 加载新的音素数
python -c "
import json
pm = json.load(open('$PREPROCESSED_DIR/phoneme_map.json'))
print(f'配置文件音素数: {pm[\"num_phonemes\"]}')
"

python train/train.py \
    --config_dir "$CONFIG_DIR" \
    --data_dir "$PREPROCESSED_DIR" \
    --phoneme_map "$PREPROCESSED_DIR/phoneme_map.json" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --device cuda 2>&1 | tee "$LOG_DIR/training.log"

echo ""
echo "训练完成: $(date)"
