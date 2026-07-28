#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yuaz

cd /root/autodl-tmp

CKPT=${1:-finetune_checkpoints/model_012000.pt}

# 1. 创建 MIDI 文件: "ni hao wo shi li lao shi" (7 个音节)
python -c "
import mido
import os

mid = mido.MidiFile(ticks_per_beat=480)
track = mido.MidiTrack()
mid.tracks.append(track)

# 设置 tempo (BPM=120)
track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))

# 音符: 每个 1 拍 (D4、E4、F4、G4、A4、G4、F4)
lyrics = ['ni', 'hao', 'wo', 'shi', 'li', 'lao', 'shi']
notes = [62, 64, 65, 67, 69, 67, 65]  # D4 E4 F4 G4 A4 G4 F4
ticks_per_beat = 480
beat_dur = 480  # 1 拍

tick = 0
for lyr, note in zip(lyrics, notes):
    track.append(mido.MetaMessage('lyrics', text=lyr, time=0))
    track.append(mido.Message('note_on', note=note, velocity=80, time=0))
    track.append(mido.Message('note_off', note=note, velocity=0, time=beat_dur))
    tick += beat_dur

# 保存
midi_path = 'ni_hao_wo_shi_li_lao_shi.mid'
mid.save(midi_path)
print(f'MIDI 已创建: {midi_path}')
for l, n in zip(lyrics, notes):
    f0 = 440.0 * (2 ** ((n - 69) / 12))
    print(f'  {l}: MIDI={n} F0={f0:.1f}Hz')
"

# 2. 修改 infer_midi.py 补充缺失的拼音映射
python -c "
import re

script_path = 'yuaz_sgr/scripts/infer_midi.py'
with open(script_path, 'r') as f:
    content = f.read()

# 添加缺失的拼音：hao, lao, ma, ba, da, xiao, hui, jiu, zhi, de, ge
new_mapping = '''
    'hao': ['h', 'ao'], 'lao': ['l', 'ao'], 'ma': ['m', 'a'],
    'ba': ['b', 'a'],  'da': ['d', 'a'],  'xiao': ['x', 'iao'],
    'hui': ['h', 'ui'], 'jiu': ['j', 'iu'], 'zhi': ['zh', 'i'],
    'de': ['d', 'e'],  'ge': ['g', 'e'],  'tian': ['t', 'ian'],
    'sheng': ['sh', 'eng'], 'ri': ['r', 'i'],
'''

# 在 PY_TO_PH 字典最后插入
old_last = \"    'na': ['n', 'a'],  'bu': ['b', 'u'],\"
new_last = old_last + new_mapping
content = content.replace(old_last, new_last)

with open(script_path, 'w') as f:
    f.write(content)
print('PY_TO_PH 已更新')
"

# 3. 运行推理
python yuaz_sgr/scripts/infer_midi.py \
    --checkpoint "$CKPT" \
    --midi ni_hao_wo_shi_li_lao_shi.mid \
    --phoneme_map data/preprocessed/phoneme_map.json \
    --config_dir yuaz_sgr/config \
    --output ni_hao_wo_shi_li_lao_shi.wav \
    --n_steps 1 \
    --device cuda

echo ""
echo "=== 完成 ==="
ls -la ni_hao_wo_shi_li_lao_shi.wav
