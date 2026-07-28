#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Kill old
pids=$(ps aux | grep "train/train.py" | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then kill -9 $pids 2>/dev/null; fi
sleep 3

cd /root/autodl-tmp/yuaz_sgr
# 取最新的 checkpoint (按时间排序)
LATEST_CKPT=$(ls -t /root/autodl-tmp/checkpoints/model_*.pt 2>/dev/null | head -1)
echo "Resume from: $LATEST_CKPT"

nohup python -u train/train.py \
    --config_dir /root/autodl-tmp/yuaz_sgr/config \
    --data_dir /root/autodl-tmp/yuaz_sgr/data/preprocessed \
    --phoneme_map /root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json \
    --output_dir /root/autodl-tmp/checkpoints \
    --log_dir /root/autodl-tmp/logs \
    --device cuda \
    --resume "$LATEST_CKPT" \
    > /root/autodl-tmp/logs/train_nohup.out 2>&1 &
echo "TRAIN_PID=$!"
