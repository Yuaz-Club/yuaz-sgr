"""
infer_ust.py — 从 UST (Utau/OpenUtau 工程) 直接合成歌曲

用法:
    python infer_ust.py 歌曲.ust --ckpt checkpoint_75k_pure.pt --out out.wav

说明:
    - 解析 [#SETTING] 的 Tempo 与每个 [#XXXX] 音符块的 Length / Lyric / NoteNum
    - Lyric 为拼音 (如 zhe / shi) 时直接使用; 为汉字时尝试 pypinyin 自动转拼音
    - 休止符 (R) 与延音 (-) 跳过, 但保留时值
    - 生成临时 MIDI 后复用 sing_from_midi
"""
import os
import re
import sys
import tempfile
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yuaz_sgr import YuazSinger

def parse_ust(path):
    txt = open(path, encoding='utf-8', errors='ignore').read()
    m = re.search(r'Tempo=([0-9.]+)', txt)
    tempo = float(m.group(1)) if m else 120.0
    blocks = re.split(r'\[#(\d+)\]', txt)
    notes = []
    for i in range(1, len(blocks) - 1, 2):
        body = blocks[i + 1]
        lm = re.search(r'Lyric=(.*)', body)
        nm = re.search(r'NoteNum=(-?\d+)', body)
        lenm = re.search(r'Length=(\d+)', body)
        if not (lm and nm and lenm):
            continue
        notes.append((int(nm.group(1)), int(lenm.group(1)), lm.group(1).strip()))
    return tempo, notes

def to_pinyin(word):
    if not re.search(r'[\u4e00-\u9fff]', word):
        return word.lower()
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        raise SystemExit(f'歌词 "{word}" 是汉字, 需要 pypinyin: pip install pypinyin')
    return ''.join(lazy_pinyin(word)).lower()

def build_midi(tempo, notes, out_path):
    import pretty_midi
    ticks_per_beat = 480
    sec_per_tick = 60.0 / tempo / ticks_per_beat
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    lyrics = []
    for pitch, length, lyric in notes:
        dur = length * sec_per_tick
        if lyric not in ('R', '-', ''):
            inst.notes.append(pretty_midi.Note(velocity=95, pitch=pitch,
                                               start=t, end=max(t + 0.05, t + dur)))
            lyrics.append(to_pinyin(lyric))
        t += dur
    pm.instruments.append(inst)
    pm.write(out_path)
    return ' '.join(lyrics)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ust')
    ap.add_argument('--ckpt', default='checkpoint_75k_pure.pt')
    ap.add_argument('--out', default=None)
    ap.add_argument('--irr-steps', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--predicted-f0', action='store_true')
    ap.add_argument('--consonant-gain', type=float, default=0.5)
    args = ap.parse_args()

    tempo, notes = parse_ust(args.ust)
    print(f'UST: tempo={tempo} 音符={len(notes)}')
    out = args.out or (os.path.splitext(args.ust)[0] + '_irr.wav')
    with tempfile.TemporaryDirectory() as td:
        midi = os.path.join(td, 'ust.mid')
        lyr = build_midi(tempo, notes, midi)
        print(f'歌词数={len(lyr.split())}')
        singer = YuazSinger(args.ckpt, device=args.device)
        wav = singer.sing_from_midi(
            midi, lyrics=lyr, output_path=out,
            use_predicted_f0=args.predicted_f0,
            irr_steps=args.irr_steps,
            consonant_frames='auto',
            consonant_gain=args.consonant_gain,
            f0_portamento_frames=4,
            max_chunk_frames=2048,
        )
    print(f'完成: {out} ({len(wav) / 24000:.1f}s)')

if __name__ == '__main__':
    main()

