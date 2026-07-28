#!/bin/bash
# run_train.sh — 使用 conda yuaz 环境运行
export PATH="/root/miniconda3/envs/yuaz/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
cd /root/autodl-tmp

ACTION=${1:-preprocess}

if [ "$ACTION" = "preprocess" ]; then
    echo "=== 预处理 ==="
    # 如存在基座音素映射，传入以实现索引对齐
    BASE_MAP="yuaz_sgr/data/preprocessed/phoneme_map.json"
    if [ -f "$BASE_MAP" ]; then
        echo "  使用基座音素映射: $BASE_MAP"
        python yuaz_sgr/scripts/preprocess_li.py \
            --data_dir li_data \
            --output_dir data/preprocessed \
            --val_songs 1 --test_songs 1 --device cuda \
            --base_phoneme_map "$BASE_MAP"
    else
        echo "  未找到基座音素映射, 独立构建"
        python yuaz_sgr/scripts/preprocess_li.py \
            --data_dir li_data \
            --output_dir data/preprocessed \
            --val_songs 1 --test_songs 1 --device cuda
    fi
elif [ "$ACTION" = "train" ]; then
    echo "=== 微调训练 ==="
    python yuaz_sgr/train/train.py \
        --config_dir yuaz_sgr/config \
        --data_dir data/preprocessed \
        --phoneme_map data/preprocessed/phoneme_map.json \
        --output_dir finetune_checkpoints \
        --log_dir logs \
        --resume base_checkpoint.pt \
        --device cuda
elif [ "$ACTION" = "overwrite" ]; then
    echo "=== 覆盖音素数后训练 ==="
    python -c "
import json, yaml
with open('data/preprocessed/phoneme_map.json') as f:
    pm = json.load(f)
n_ph = pm['num_phonemes']
print(f'音素数: {n_ph}')

with open('yuaz_sgr/config/finetune.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['num_phonemes'] = n_ph
with open('yuaz_sgr/config/finetune.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('finetune.yaml 已更新')
"
    python yuaz_sgr/train/train.py \
        --config_dir yuaz_sgr/config \
        --data_dir data/preprocessed \
        --phoneme_map data/preprocessed/phoneme_map.json \
        --output_dir finetune_checkpoints \
        --log_dir logs \
        --resume base_checkpoint.pt \
        --device cuda
fi
