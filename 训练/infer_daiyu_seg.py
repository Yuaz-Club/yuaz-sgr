"""
黛玉葬花 分段推理脚本
把长 MIDI 按帧数分段，每段约 15 秒，带交叉淡入淡出拼接
三种模式:
  1. 原版 F0 + 原版音素
  2. 预测 F0 + 原版音素
  3. 原版 F0 + 预测音素
"""
import sys, os, warnings, numpy as np
sys.path.insert(0, r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr_inference")
from yuaz_sgr import YuazSinger

MIDI_PATH = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\黛玉葬花.mid"
CKPT_PATH = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr_inference\checkpoint.pt"
CONFIG_DIR = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr_inference\yuaz_sgr\assets"
OUTPUT_DIR = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr_inference"

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"设备: {device}")

singer = YuazSinger(CKPT_PATH, config_dir=CONFIG_DIR, device=device)

# 解析 MIDI 和歌词
import pretty_midi
midi = pretty_midi.PrettyMIDI(MIDI_PATH)
notes = []
for inst in midi.instruments:
    for n in inst.notes:
        notes.append((n.pitch, n.start, n.end))
notes.sort(key=lambda x: x[1])

import mido
mid = mido.MidiFile(MIDI_PATH)
all_lyrics = []
for track in mid.tracks:
    for msg in track:
        if msg.type == 'lyrics' and msg.text.strip():
            all_lyrics.append(msg.text.strip())

print(f"共 {len(notes)} 音符, {len(all_lyrics)} 歌词")

# 分段: 每段约 15 秒 (~1400 帧)
MAX_SEG_FRAMES = 1400
segments = []  # [(note_start, note_end), ...]
seg_start = 0
while seg_start < len(notes):
    total_frames = 0
    seg_end = seg_start
    while seg_end < len(notes):
        dur = notes[seg_end][2] - notes[seg_end][1]
        f = max(1, int(dur * singer.frame_rate))
        if total_frames + f > MAX_SEG_FRAMES and seg_end > seg_start:
            break
        total_frames += f
        seg_end += 1
    segments.append((seg_start, seg_end))
    seg_start = seg_end

print(f"分段: {len(segments)} 段")
for i, (s, e) in enumerate(segments):
    dur = notes[e-1][2] - notes[s][1]
    print(f"  段{i}: 音符[{s}:{e}] ({e-s}个), {dur:.1f}s")

# 交叉淡入淡出 (50ms)
CROSSFADE_SAMPLES = int(0.05 * singer.sample_rate)

def crossfade(a, b):
    """叠加两个波形，用 50ms 线性交叉淡入淡出"""
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    cf = min(CROSSFADE_SAMPLES, len(a), len(b))
    fade_out = np.linspace(1, 0, cf)
    fade_in = np.linspace(0, 1, cf)
    a[-cf:] *= fade_out
    b[:cf] *= fade_in
    return np.concatenate([a[:-cf], a[-cf:] + b[:cf], b[cf:]])

def run_segmented(predict_durations, use_predicted_f0, label):
    print(f"\n{'='*60}")
    print(f"{label} (pred_dur={predict_durations}, pred_f0={use_predicted_f0})")
    print('='*60)
    full_wav = np.array([], dtype=np.float32)
    for seg_i, (s, e) in enumerate(segments):
        seg_notes = notes[s:e]
        seg_lyrics = all_lyrics[s:e]
        seg_lyrics_str = " ".join(seg_lyrics)
        
        # 找到该段的开始时间, 将音符时间归一化到段起始
        seg_start_time = seg_notes[0][1]
        norm_notes = [(p, st - seg_start_time, et - seg_start_time) for p, st, et in seg_notes]
        
        # 构建本段的 score
        ph_ids, durs, f0, nl = singer._build_score(norm_notes, seg_lyrics, consonant_frames=2)
        
        # 推理
        wav = singer.sing(
            ph_ids, durs, f0,
            note_lengths=nl,
            predict_durations=predict_durations,
            use_predicted_f0=use_predicted_f0,
            n_steps=1,
        )
        
        elapsed = len(full_wav) / singer.sample_rate
        print(f"  段{seg_i}: {len(seg_notes)} 音符, {len(ph_ids)} 音素, {len(wav)/singer.sample_rate:.1f}s 音频")
        full_wav = crossfade(full_wav, wav)
    
    out_path = os.path.join(OUTPUT_DIR, f"黛玉葬花_分段_{label}.wav")
    singer.save(out_path, full_wav)
    print(f"  总时长: {len(full_wav)/singer.sample_rate:.1f}s")
    return full_wav

# 运行三种模式
wav1 = run_segmented(predict_durations=False, use_predicted_f0=False, label="原版F0+原版音素")
wav2 = run_segmented(predict_durations=False, use_predicted_f0=True,  label="预测F0+原版音素")
wav3 = run_segmented(predict_durations=True,  use_predicted_f0=False, label="原版F0+预测音素")

print(f"\n全部完成!")
for f in os.listdir(OUTPUT_DIR):
    if f.startswith("黛玉葬花_分段"):
        fp = os.path.join(OUTPUT_DIR, f)
        print(f"  {f}: {os.path.getsize(fp)//1024} KB")
