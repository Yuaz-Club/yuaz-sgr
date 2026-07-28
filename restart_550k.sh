export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin

# 最新的 checkpoint
CKPT=/root/autodl-tmp/checkpoints/model_440000.pt

cd /root/autodl-tmp/yuaz_sgr
nohup python -u train/train.py \
  --config_dir /root/autodl-tmp/yuaz_sgr/config \
  --data_dir /root/autodl-tmp/yuaz_sgr/data/preprocessed \
  --phoneme_map /root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json \
  --output_dir /root/autodl-tmp/checkpoints \
  --log_dir /root/autodl-tmp/logs \
  --device cuda \
  --resume $CKPT \
  > /root/autodl-tmp/logs/train_nohup.out 2>&1 &

sleep 5
echo "重启完成"
tail -1 /root/autodl-tmp/logs/train_nohup.out
