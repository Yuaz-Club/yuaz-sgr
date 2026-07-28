export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
cd /root/autodl-tmp/yuaz_sgr

# 自动获取最新 checkpoint
CKPT=$(ls -t /root/autodl-tmp/checkpoints/model_*.pt | head -1)
echo "使用: $CKPT"

python midi_infer.py \
  --checkpoint "$CKPT" \
  --config_dir /root/autodl-tmp/yuaz_sgr/config \
  --midi "/root/autodl-tmp/yuaz_sgr/inference_input/思华年.mid" \
  --phoneme_map /root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json \
  --output /root/autodl-tmp/inference_output \
  --device cuda 2>&1

python ust_infer.py \
  --checkpoint "$CKPT" \
  --config_dir /root/autodl-tmp/yuaz_sgr/config \
  --ust "/root/autodl-tmp/yuaz_sgr/inference_input/玫瑰少年-autosave_FL Keys_04.ust" \
  --phoneme_map /root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json \
  --output /root/autodl-tmp/inference_output \
  --device cuda 2>&1
