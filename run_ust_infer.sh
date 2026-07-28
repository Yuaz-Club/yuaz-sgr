#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
cd /root/autodl-tmp/yuaz_sgr
LATEST_CKPT=$(ls -t /root/autodl-tmp/checkpoints/model_*.pt 2>/dev/null | head -1)
UST_FILE="/root/autodl-tmp/yuaz_sgr/inference_input/玫瑰少年-autosave_FL Keys_04.ust"
python ust_infer.py \
    --checkpoint "$LATEST_CKPT" \
    --config_dir /root/autodl-tmp/yuaz_sgr/config \
    --ust "$UST_FILE" \
    --phoneme_map /root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json \
    --output /root/autodl-tmp/inference_output \
    --device cuda 2>&1
