# YUAZ SGR
"""scripts/create_synthetic_opencpop"""
import os, sys
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

SR = 44100

def make_sine_f0(f0_hz, duration_sec, sr=SR):
    """生成单频正弦波"""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return 0.3 * np.sin(2 * np.pi * f0_hz * t)

def make_note_sequence(notes, sr=SR):
    """notes: [(f0, duration_sec), ...]"""
    segments = [make_sine_f0(f0, dur, sr) for f0, dur in notes]
    return np.concatenate(segments)

def make_wav_file(path, notes, sr=SR):
    audio = make_note_sequence(notes, sr).astype(np.float32)
    wavfile.write(path, sr, audio)

def make_midi_file(path, notes, tempo=120):
    """生成简单 MIDI 文件 (最小实现)"""
    import struct

    def midi_to_f0(pitch):
        return 440.0 * (2 ** ((pitch - 69) / 12))

    def f0_to_midi(f0):
        return max(0, min(127, int(round(69 + 12 * np.log2(max(1, f0) / 440.0)))))

    # 计算每拍秒数
    spb = 60.0 / tempo
    ticks_per_beat = 480
    us_per_beat = int(spb * 1_000_000)

    cumulative_ticks = 0
    track_events = bytearray()

    for f0, dur_sec in notes:
        pitch = f0_to_midi(f0)
        dur_ticks = int(dur_sec / spb * ticks_per_beat)

        # Note On
        track_events.extend(_midi_delta_time(cumulative_ticks))
        track_events.extend(bytes([0x90, pitch, 80]))
        cumulative_ticks = 0

        # Note Off
        track_events.extend(_midi_delta_time(dur_ticks))
        track_events.extend(bytes([0x80, pitch, 0]))
        cumulative_ticks = 0

    # End of Track
    track_events.extend(bytes([0x00, 0xFF, 0x2F, 0x00]))

    # MThd header
    header = struct.pack('>4sIHHH', b'MThd', 6, 1, 1, ticks_per_beat)
    # MTrk
    track_header = struct.pack('>4sI', b'MTrk', len(track_events))

    with open(path, 'wb') as f:
        f.write(header + track_header + track_events)

def _midi_delta_time(ticks):
    """MIDI 可变长度编码 delta time"""
    ticks = int(ticks)
    buf = bytearray()
    buf.append(ticks & 0x7F)
    ticks >>= 7
    while ticks > 0:
        buf.append(0x80 | (ticks & 0x7F))
        ticks >>= 7
    return bytes(reversed(buf))

def make_textgrid_file(path, notes, phonemes_per_note):
    """生成 TextGrid 标注文件"""
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        'xmin = 0',
        f'xmax = {sum(d for _, d in notes):.6f}',
        'tiers? <exists>',
        'size = 1',
        'item []:',
        '    item [1]:',
        '        class = "IntervalTier"',
        '        name = "phones"',
        '        xmin = 0',
        f'        xmax = {sum(d for _, d in notes):.6f}',
    ]

    total_intervals = sum(len(ph) for ph in phonemes_per_note)
    lines.append(f'        intervals: size = {total_intervals}')

    interval_idx = 1
    time_cursor = 0.0
    for i, ((f0, dur), ph_list) in enumerate(zip(notes, phonemes_per_note)):
        ph_dur = dur / len(ph_list)
        for ph in ph_list:
            lines.append(f'        intervals [{interval_idx}]:')
            lines.append(f'            xmin = {time_cursor:.6f}')
            time_cursor += ph_dur
            lines.append(f'            xmax = {time_cursor:.6f}')
            lines.append(f'            text = "{ph}"')
            interval_idx += 1

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ── 5 首合成歌曲 (中文儿歌) ──
SONGS = [
    {
        'id': 'song_001',
        'notes': [
            (262, 0.5), (294, 0.5), (330, 0.5), (349, 0.5),  # do re mi fa
            (392, 0.5), (349, 0.5), (330, 0.5), (294, 0.5),  # sol fa mi re
            (262, 0.5), (294, 0.5), (330, 0.5), (349, 0.5),  # do re mi fa
            (392, 0.5), (440, 0.5), (392, 0.5), (330, 0.5),  # sol la sol mi
            (262, 0.5), (294, 0.5), (330, 0.5), (294, 0.5),  # do re mi re
            (262, 0.5), (220, 0.5), (262, 0.5), (294, 0.5),  # do si do re
            (330, 0.5), (349, 0.5), (392, 0.5), (440, 0.5),  # mi fa sol la
            (392, 1.0),                                         # sol
        ],
        'phonemes': [
            ['d', 'uo'], ['l', 'ai'], ['m', 'i'], ['f', 'a'],
            ['s', 'uo'], ['f', 'a'], ['m', 'i'], ['l', 'ai'],
            ['d', 'uo'], ['l', 'ai'], ['m', 'i'], ['f', 'a'],
            ['s', 'uo'], ['l', 'a'], ['s', 'uo'], ['m', 'i'],
            ['d', 'uo'], ['l', 'ai'], ['m', 'i'], ['l', 'ai'],
            ['d', 'uo'], ['x', 'i'], ['d', 'uo'], ['l', 'ai'],
            ['m', 'i'], ['f', 'a'], ['s', 'uo'], ['l', 'a'],
            ['s', 'uo'],
        ],
    },
    {
        'id': 'song_002',
        'notes': [
            (330, 0.5), (330, 0.5), (349, 0.5), (392, 0.5),
            (392, 0.5), (349, 0.5), (330, 0.5), (294, 0.5),
            (262, 0.5), (262, 0.5), (294, 0.5), (330, 0.5),
            (330, 0.75), (294, 0.25), (294, 1.0),
            (330, 0.5), (330, 0.5), (349, 0.5), (392, 0.5),
            (392, 0.5), (349, 0.5), (330, 0.5), (294, 0.5),
            (262, 0.5), (262, 0.5), (294, 0.5), (330, 0.5),
            (294, 0.75), (262, 0.25), (262, 1.0),
        ],
        'phonemes': [
            ['m', 'i'], ['m', 'i'], ['f', 'a'], ['s', 'uo'],
            ['s', 'uo'], ['f', 'a'], ['m', 'i'], ['l', 'ai'],
            ['d', 'uo'], ['d', 'uo'], ['l', 'ai'], ['m', 'i'],
            ['m', 'i'], ['l', 'ai'], ['l', 'ai'],
            ['m', 'i'], ['m', 'i'], ['f', 'a'], ['s', 'uo'],
            ['s', 'uo'], ['f', 'a'], ['m', 'i'], ['l', 'ai'],
            ['d', 'uo'], ['d', 'uo'], ['l', 'ai'], ['m', 'i'],
            ['l', 'ai'], ['d', 'uo'], ['d', 'uo'],
        ],
    },
    {
        'id': 'song_003',
        'notes': [
            (262, 0.5), (330, 0.5), (392, 0.5), (523, 0.5),
            (440, 0.5), (392, 0.5), (349, 0.5), (330, 0.5),
            (294, 0.5), (349, 0.5), (440, 0.5), (587, 0.5),
            (523, 0.5), (440, 0.5), (392, 0.5), (349, 0.5),
            (330, 0.5), (392, 0.5), (523, 0.5), (659, 0.5),
            (587, 1.0),
        ],
        'phonemes': [
            ['d', 'uo'], ['m', 'i'], ['s', 'uo'], ['d', 'uo', '_h'],
            ['l', 'a'], ['s', 'uo'], ['f', 'a'], ['m', 'i'],
            ['l', 'ai'], ['f', 'a'], ['l', 'a'], ['l', 'ai', '_h'],
            ['d', 'uo', '_h'], ['l', 'a'], ['s', 'uo'], ['f', 'a'],
            ['m', 'i'], ['s', 'uo'], ['d', 'uo', '_h'], ['m', 'i', '_h'],
            ['l', 'a', '_h'],
        ],
    },
    {
        'id': 'song_004',
        'notes': [
            (392, 0.75), (440, 0.25), (523, 0.5),
            (440, 0.5), (392, 0.5), (349, 0.5),
            (330, 0.5), (349, 0.5), (392, 0.5),
            (294, 0.5), (330, 0.5), (349, 0.5),
            (392, 0.5), (349, 0.5), (330, 0.5),
            (294, 0.5), (262, 0.5), (294, 0.5),
            (330, 0.5), (349, 0.5), (392, 0.5),
            (440, 0.5), (523, 0.5), (440, 0.5),
            (392, 1.0),
        ],
        'phonemes': [
            ['s', 'uo'], ['l', 'a'], ['d', 'uo', '_h'],
            ['l', 'a'], ['s', 'uo'], ['f', 'a'],
            ['m', 'i'], ['f', 'a'], ['s', 'uo'],
            ['l', 'ai'], ['m', 'i'], ['f', 'a'],
            ['s', 'uo'], ['f', 'a'], ['m', 'i'],
            ['l', 'ai'], ['d', 'uo'], ['l', 'ai'],
            ['m', 'i'], ['f', 'a'], ['s', 'uo'],
            ['l', 'a'], ['d', 'uo', '_h'], ['l', 'a'],
            ['s', 'uo'],
        ],
    },
    {
        'id': 'song_005',
        'notes': [
            (349, 0.5), (392, 0.5), (440, 0.5), (392, 0.5),
            (349, 0.5), (330, 0.5), (294, 0.5), (262, 0.5),
            (294, 0.5), (330, 0.5), (349, 0.5), (330, 0.5),
            (294, 0.5), (262, 0.5), (247, 0.5), (262, 0.5),
            (349, 0.5), (392, 0.5), (440, 0.5), (392, 0.5),
            (349, 0.5), (330, 0.5), (294, 0.5), (262, 0.5),
            (294, 0.5), (330, 0.5), (349, 0.5), (294, 0.5),
            (262, 1.0),
        ],
        'phonemes': [
            ['f', 'a'], ['s', 'uo'], ['l', 'a'], ['s', 'uo'],
            ['f', 'a'], ['m', 'i'], ['l', 'ai'], ['d', 'uo'],
            ['l', 'ai'], ['m', 'i'], ['f', 'a'], ['m', 'i'],
            ['l', 'ai'], ['d', 'uo'], ['x', 'i'], ['d', 'uo'],
            ['f', 'a'], ['s', 'uo'], ['l', 'a'], ['s', 'uo'],
            ['f', 'a'], ['m', 'i'], ['l', 'ai'], ['d', 'uo'],
            ['l', 'ai'], ['m', 'i'], ['f', 'a'], ['l', 'ai'],
            ['d', 'uo'],
        ],
    },
]

def main():
    output_dir = Path('/workspace/yuaz_sgr/data/opencpop_raw')
    (output_dir / 'wavs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'midis').mkdir(parents=True, exist_ok=True)
    (output_dir / 'textgrids').mkdir(parents=True, exist_ok=True)

    for song in SONGS:
        sid = song['id']
        notes = song['notes']
        phonemes = song['phonemes']
        assert len(notes) == len(phonemes), f"mismatch in {sid}"

        # WAV
        wav_path = output_dir / 'wavs' / f'{sid}.wav'
        make_wav_file(str(wav_path), notes)

        # MIDI
        midi_path = output_dir / 'midis' / f'{sid}.mid'
        make_midi_file(str(midi_path), notes)

        # TextGrid
        tg_path = output_dir / 'textgrids' / f'{sid}.TextGrid'
        make_textgrid_file(str(tg_path), notes, phonemes)

        dur = sum(d for _, d in notes)
        print(f"  {sid}: {len(notes)} notes, {dur:.1f}s, {len(phonemes)} phonemes")

    print(f"\n合成数据集生成完毕: {output_dir}")
    print(f"  wavs/: {len(SONGS)} 个 WAV 文件")
    print(f"  midis/: {len(SONGS)} 个 MIDI 文件")
    print(f"  textgrids/: {len(SONGS)} 个 TextGrid 文件")

if __name__ == '__main__':
    main()