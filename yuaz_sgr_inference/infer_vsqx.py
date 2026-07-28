"""
infer_vsqx.py — 从 VSQX 文件推理合成歌声

用法:
    python infer_vsqx.py
"""

import os
import sys
import xml.etree.ElementTree as ET
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yuaz_sgr import YuazSinger


def parse_vsqx(vsqx_path):
    """解析 VSQX 文件 → [(pitch, start_sec, end_sec, pinyin), ...]

    VSQX 是 VOCALOID 4 项目文件 (XML 格式).
    resolution = 480 ticks/quarter
    tempo 单位 = BPM * 100 → 7200 = 72.00 BPM
    """
    tree = ET.parse(vsqx_path)
    root = tree.getroot()

    # 命名空间
    ns = {"vsq4": "http://www.yamaha.co.jp/vocaloid/schema/vsq4/"}

    # 解析 tempo 和 resolution
    resolution = int(root.find(".//vsq4:resolution", ns).text)
    tempo_elems = root.findall(".//vsq4:tempo", ns)
    # 默认取第一个 tempo (假设单 tempo)
    tempo_raw = int(tempo_elems[0].find("vsq4:v", ns).text) if tempo_elems else 7200
    bpm = tempo_raw / 100.0
    ticks_per_sec = resolution * bpm / 60.0

    # 提取所有音符
    notes = []
    for note_elem in root.findall(".//vsq4:note", ns):
        tick = int(note_elem.find("vsq4:t", ns).text)
        dur_ticks = int(note_elem.find("vsq4:dur", ns).text)
        pitch = int(note_elem.find("vsq4:n", ns).text)
        lyric_elem = note_elem.find("vsq4:y", ns)
        lyric = lyric_elem.text if lyric_elem is not None else "a"

        start_sec = tick / ticks_per_sec
        end_sec = (tick + dur_ticks) / ticks_per_sec

        # 跳过休止符 "-"
        if lyric.strip() == "-":
            continue

        notes.append((pitch, start_sec, end_sec, lyric.strip()))

    print(f"VSQX 解析完成: {bpm:.1f} BPM, {len(notes)} 个音符")
    print(f"  歌曲时长: {notes[-1][2]:.1f}s")
    return notes


def create_temp_midi(notes, output_midi):
    """从 VSQX 音符创建 MIDI 文件"""
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("请安装 pretty_midi: pip install pretty_midi")

    midi = pretty_midi.PrettyMIDI(initial_tempo=72.0)
    inst = pretty_midi.Instrument(program=0)

    for pitch, start_sec, end_sec, lyric in notes:
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start_sec,
            end=end_sec,
        )
        inst.notes.append(note)

    midi.instruments.append(inst)
    midi.write(output_midi)
    return output_midi


def main():
    # 路径配置
    vsqx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..",
                             "【调音者：想吃素的BeaR】【熊】这世界那么多人【无参】.vsqx")
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "checkpoint_300k.pt")
    output_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "示例.wav")

    if not os.path.exists(vsqx_path):
        print(f"错误: 找不到 VSQX 文件 {vsqx_path}")
        sys.exit(1)
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型 {checkpoint_path}")
        sys.exit(1)

    # 1. 解析 VSQX
    print("=" * 60)
    print("YUAZ SGR — VSQX 推理")
    print("=" * 60)
    notes = parse_vsqx(vsqx_path)

    # 构建歌词字符串
    lyrics = " ".join(n[3] for n in notes)
    print(f"歌词: {lyrics[:80]}...")

    # 2. 创建临时 MIDI
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        temp_midi = f.name
    try:
        create_temp_midi(notes, temp_midi)

        # 3. 加载模型并合成
        print("\n加载模型...")
        singer = YuazSinger(checkpoint_path, device="cpu")

        print("合成中 (n_steps=8, use_predicted_f0=True)...")
        import time
        t0 = time.time()

        wav = singer.sing_from_midi(
            temp_midi,
            lyrics=lyrics,
            output_path=output_wav,
            predict_durations=False,
            use_predicted_f0=True,
            n_steps=8,
        )
        elapsed = time.time() - t0
        duration = len(wav) / 24000
        print(f"\n完成! 耗时 {elapsed:.1f}s, 生成 {duration:.1f}s 音频")
        print(f"RTF = {elapsed/duration:.3f}")
        print(f"输出: {output_wav}")
    finally:
        os.unlink(temp_midi)


if __name__ == "__main__":
    main()
