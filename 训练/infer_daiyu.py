"""
黛玉葬花 推理脚本
使用本地 yuaz_sgr_inference 包
三种模式:
  1. 原版 F0 + 原版音素 (MIDI F0, MIDI durations)
  2. 预测 F0 + 原版音素 (模型预测 F0, MIDI durations)
  3. 原版 F0 + 预测音素 (MIDI F0, 模型预测 durations)
"""
import sys, os
sys.path.insert(0, r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr_inference")
from yuaz_sgr import YuazSinger

MIDI_PATH = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\黛玉葬花.mid"
CKPT_PATH = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr_inference\checkpoint.pt"
CONFIG_DIR = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr_inference\yuaz_sgr\assets"
OUTPUT_DIR = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\yuaz_sgr_inference"

# 初始化
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"设备: {device}")

singer = YuazSinger(CKPT_PATH, config_dir=CONFIG_DIR, device=device)

# MIDI 信息
import mido
mid = mido.MidiFile(MIDI_PATH)
lyrics = []
for track in mid.tracks:
    for msg in track:
        if msg.type == 'lyrics' and msg.text.strip():
            lyrics.append(msg.text.strip())
print(f"歌词: {len(lyrics)} 音节")

# ─── 模式 1: 原版 F0 + 原版音素 ───
print("\n" + "="*60)
print("模式1: 原版 F0 + 原版音素 (predict_durations=False, use_predicted_f0=False)")
print("="*60)
wav1 = singer.sing_from_midi(
    MIDI_PATH,
    lyrics=" ".join(lyrics),
    output_path=os.path.join(OUTPUT_DIR, "黛玉葬花_原版F0+原版音素.wav"),
    predict_durations=False,
    use_predicted_f0=False,
    n_steps=1,
)

# ─── 模式 2: 预测 F0 + 原版音素 ───
print("\n" + "="*60)
print("模式2: 预测 F0 + 原版音素 (predict_durations=False, use_predicted_f0=True)")
print("="*60)
wav2 = singer.sing_from_midi(
    MIDI_PATH,
    lyrics=" ".join(lyrics),
    output_path=os.path.join(OUTPUT_DIR, "黛玉葬花_预测F0+原版音素.wav"),
    predict_durations=False,
    use_predicted_f0=True,
    n_steps=1,
)

# ─── 模式 3: 原版 F0 + 预测音素 ───
print("\n" + "="*60)
print("模式3: 原版 F0 + 预测音素 (predict_durations=True, use_predicted_f0=False)")
print("="*60)
wav3 = singer.sing_from_midi(
    MIDI_PATH,
    lyrics=" ".join(lyrics),
    output_path=os.path.join(OUTPUT_DIR, "黛玉葬花_原版F0+预测音素.wav"),
    predict_durations=True,
    use_predicted_f0=False,
    n_steps=1,
)

print("\n全部完成!")
for f in ["黛玉葬花_原版F0+原版音素.wav", "黛玉葬花_预测F0+原版音素.wav", "黛玉葬花_原版F0+预测音素.wav"]:
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(path):
        sz = os.path.getsize(path)
        print(f"  {f}: {sz//1024} KB")
