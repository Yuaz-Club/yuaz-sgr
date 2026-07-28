#!/bin/bash
CKPT=$(ls -t /root/autodl-tmp/checkpoints/model_*.pt | head -1)
echo "使用: $CKPT"
cd /root/autodl-tmp/yuaz_sgr
python -u midi_infer.py --checkpoint "$CKPT" --config_dir /root/autodl-tmp/yuaz_sgr/config --midi /root/autodl-tmp/yuaz_sgr/inference_input/思华年.mid --phoneme_map /root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json --output /root/autodl-tmp/inference_output/
python -u ust_infer.py --checkpoint "$CKPT" --config_dir /root/autodl-tmp/yuaz_sgr/config --ust "/root/autodl-tmp/yuaz_sgr/inference_input/玫瑰少年-autosave_FL Keys_04.ust" --phoneme_map /root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json --output /root/autodl-tmp/inference_output/
