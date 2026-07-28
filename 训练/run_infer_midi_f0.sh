#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

cd /root/autodl-tmp

CKPT=${1:-finetune_checkpoints/model_012000.pt}

# 修改 infer_midi.py 使用用 MIDI 原始 F0
python -c "
import re

script_path = 'yuaz_sgr/scripts/infer_midi.py'
with open(script_path, 'r') as f:
    content = f.read()

# 在 forward_synthesis 调用中插入 use_predicted_f0=False
old = \"predict_durations=True\"
new = \"predict_durations=True, use_predicted_f0=False\"
content = content.replace(old, new)

with open(script_path, 'w') as f:
    f.write(content)
print('已改为 use_predicted_f0=False')
"

python yuaz_sgr/scripts/infer_midi.py \
    --checkpoint "$CKPT" \
    --midi ni_hao_wo_shi_li_lao_shi.mid \
    --phoneme_map data/preprocessed/phoneme_map.json \
    --config_dir yuaz_sgr/config \
    --output ni_hao_wo_shi_li_lao_shi_midi_f0.wav \
    --n_steps 1 \
    --device cuda

# 改回来
python -c "
script_path = 'yuaz_sgr/scripts/infer_midi.py'
with open(script_path, 'r') as f:
    content = f.read()
content = content.replace('predict_durations=True, use_predicted_f0=False', 'predict_durations=True')
with open(script_path, 'w') as f:
    f.write(content)
print('已恢复 use_predicted_f0=True')
"

echo ""
echo "=== 完成 ==="
ls -la ni_hao_wo_shi_li_lao_shi_midi_f0.wav
