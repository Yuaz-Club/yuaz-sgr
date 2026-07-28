#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/yuaz_sgr

# 1. 训 2000 步 (约 10 分钟), checkpoint 在 5000 步自动保存
#    但我们需要更早验证, 所以用 5000 步
echo "训到 5000 步..."
python train/train.py \
    --config_dir config \
    --data_dir data/preprocessed \
    --phoneme_map data/preprocessed/phoneme_map.json \
    --output_dir /root/autodl-tmp/checkpoints \
    --log_dir /root/autodl-tmp/logs \
    --device cuda 2>&1 | head -50 &

TRAIN_PID=$!

# 等 checkpoint 出现
echo "等待 checkpoint..."
while [ ! -f /root/autodl-tmp/checkpoints/model_005000.pt ]; do
    sleep 30
    LATEST=$(ls -t /root/autodl-tmp/checkpoints/model_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "发现: $LATEST"
        CKPT=$LATEST
        break
    fi
done

# 2. 停训练
kill $TRAIN_PID 2>/dev/null
sleep 2

# 3. 跑快速验证
python3 /root/autodl-tmp/yuaz_sgr/scripts/quick_test.py 2>&1

echo "=== 验证完成 ==="
