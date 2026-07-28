#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
cd /root/autodl-tmp/yuaz_sgr
python midi_infer.py \
    --checkpoint /root/autodl-tmp/checkpoints/model_100000.pt \
    --config_dir /root/autodl-tmp/yuaz_sgr/config \
    --midi /root/autodl-tmp/yuaz_sgr/inference_input/思华年.mid \
    --phoneme_map /root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json \
    --output /root/autodl-tmp/inference_output \
    --device cuda 2>&1
