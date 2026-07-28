# YUAZ SGR
"""
MIDI 全流程推理脚本 — 最终修复版

核心修正:
  1. 音符保持原始 on_dur (note_on → note_off), 不向间隙扩展
  2. 休止间隙作为独立 REST 段插入 (SP 音素 + F0=0)
  3. 帧对齐: 所有 segment 的 F0 和音素展开帧数完全一致，无需插值
"""
import os, sys, json, argparse
import numpy as np
import torch
import mido
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.svs_model import SVSModel


def load_config(config_dir):
    config = {}
    for name in ['base', 'model']:
        with open(os.path.join(config_dir, f'{name}.yaml')) as f:
            import yaml
            config.update(yaml.safe_load(f))
    return config


def parse_midi_notes(midi_path):
    """
    解析 MIDI 返回:
      notes: [(start_sec, end_sec, midi_note), ...]
      lyrics: [lyric_str, ...]  按 note_on 时间对齐
      tempo: 实际 BPM 对应的微秒/拍
    """
    mid = mido.MidiFile(str(midi_path))
    tempo = 500000
    ticks_per_beat = mid.ticks_per_beat

    # 读取 tempo (Track 0 的第一个 set_tempo)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        break

    # 收集所有事件 (含歌词) 及其 tick
    raw_notes = []   # (tick, note, is_on)
    lyrics_events = []  # (tick, text)

    for track in mid.tracks:
        now_ticks = 0
        for msg in track:
            now_ticks += msg.time
            if msg.type == 'lyrics' and msg.text.strip():
                lyrics_events.append((now_ticks, msg.text.strip()))
            if msg.type == 'note_on' and msg.velocity > 0:
                raw_notes.append((now_ticks, msg.note, True))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                raw_notes.append((now_ticks, msg.note, False))

    # 排序
    raw_notes.sort(key=lambda x: x[0])
    lyrics_events.sort(key=lambda x: x[0])

    # 音符解析: note_on → note_off 配对
    active = {}
    note_list = []  # [(start_tick, end_tick, note)]
    for tick, note, is_on in raw_notes:
        if is_on:
            active[note] = tick
        else:
            start_tick = active.pop(note, None)
            if start_tick is not None:
                note_list.append((start_tick, tick, note))

    note_list.sort(key=lambda x: x[0])

    # 歌词按 tick 与音符对齐
    if len(lyrics_events) != len(note_list):
        print(f"WARNING: 歌词({len(lyrics_events)}) 与音符({len(note_list)}) 数量不匹配")
        # 尝试按顺序截断/补齐
        min_len = min(len(lyrics_events), len(note_list))
        lyrics = [l for _, l in lyrics_events[:min_len]]
        note_list = note_list[:min_len]
    else:
        lyrics = [l for _, l in lyrics_events]

    # 转为秒
    def ticks_to_sec(t):
        return mido.tick2second(t, ticks_per_beat, tempo)

    notes_sec = [(ticks_to_sec(s), ticks_to_sec(e), n) for s, e, n in note_list]

    return notes_sec, lyrics, tempo


# 拼音 → 音素映射 (扩展)
PY_TO_PH = {
    'yue': ['y', 'ue'], 'sui': ['s', 'ui'], 'zai': ['z', 'ai'],
    'le': ['l', 'e'],  'mei': ['m', 'ei'], 'jian': ['j', 'ian'],
    'meng': ['m', 'eng'], 'ye': ['y', 'e'], 'zui': ['z', 'ui'],
    'qu': ['q', 'v'], 'zun': ['z', 'un'], 'qian': ['q', 'ian'],
    'za': ['z', 'a'],  'xiang': ['x', 'iang'], 'ming': ['m', 'ing'],
    'he': ['h', 'e'],  'zhong': ['zh', 'ong'], 'kan': ['k', 'an'],
    'yi': ['y', 'i'],  'yan': ['y', 'an'], 'dao': ['d', 'ao'],
    'mi': ['m', 'i'],  'li': ['l', 'i'], 'ren': ['r', 'en'],
    'si': ['s', 'i'],  'shi': ['sh', 'i'], 'ni': ['n', 'i'],
    'wo': ['w', 'o'],  'ta': ['t', 'a'], 'zhe': ['zh', 'e'],
    'na': ['n', 'a'],  'bu': ['b', 'u'],
}


def get_phonemes(lyric, last_vowel=None):
    """歌词 → 音素列表
    '+' 表示 portamento/连音，延续前一音节尾元音
    """
    if lyric in PY_TO_PH:
        return list(PY_TO_PH[lyric])
    elif lyric == '+':
        if last_vowel is not None:
            return [last_vowel]  # 连音延续前一音节尾元音
        else:
            return ['a']  # 回退默认元音
    else:
        print(f"WARNING: 未映射歌词 '{lyric}', 使用 SP")
        return ['SP']


def build_timeline(notes_sec, lyrics):
    """
    构建 segment 时间线:
    每个音符作为 'note' 段, 音符间间隙作为 'rest' 段.
    返回 segments: [(type, dur_sec, f0_hz), ...]
    rest 段 f0=0, note 段 f0 根据 MIDI 音高计算
    """
    segments = []
    for i, ((start, end, midi_note), lyric) in enumerate(zip(notes_sec, lyrics)):
        on_dur = end - start
        f0 = 440.0 * (2 ** ((midi_note - 69) / 12))
        segments.append(('note', on_dur, f0, lyric))

        # 到下一音符的间隙
        if i < len(notes_sec) - 1:
            next_start = notes_sec[i+1][0]
            rest_dur = next_start - end
            if rest_dur > 0.01:  # >10ms 的间隙作为独立 rest 段
                segments.append(('rest', rest_dur, 0.0, None))

    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--midi', type=str, required=True)
    parser.add_argument('--phoneme_map', type=str, required=True)
    parser.add_argument('--config_dir', type=str, default='config')
    parser.add_argument('--output', type=str, default='synthesis.wav')
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # ── 加载配置 ──
    config = load_config(args.config_dir)
    with open(args.phoneme_map) as f:
        pm = json.load(f)
    config['conditioning']['num_phonemes'] = pm['num_phonemes']

    sr = config.get('sample_rate', 24000)
    hop_length = config.get('hop_length', 256)
    FRAME_RATE = sr / hop_length

    # ── 加载模型 ──
    model = SVSModel(config).to(args.device)
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_sd = model.state_dict()
    for k, v in ckpt['model'].items():
        clean_k = k.replace('_orig_mod.', '')
        if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
            model_sd[clean_k] = v
    model.load_state_dict(model_sd, strict=False)
    print(f"模型加载完成 (step {ckpt.get('step', 0)})")

    # ── 解析 MIDI → 时间线 segments ──
    notes_sec, lyrics, tempo = parse_midi_notes(args.midi)
    segments = build_timeline(notes_sec, lyrics)

    total_midi_sec = notes_sec[-1][1]  # 最后一个音符的结束时间
    total_seg_sec = sum(d for _, d, _, _ in segments)
    print(f"MIDI: {len(lyrics)} 歌词, {len(notes_sec)} 音符, {len(segments)} 段")
    print(f"MIDI 总时长: {total_midi_sec:.3f}s, Segment 总时长: {total_seg_sec:.3f}s")

    for s_idx, (typ, d, f0, lyr) in enumerate(segments):
        lyr_str = lyr or ''
        print(f"  seg[{s_idx:2d}]: {typ:4s}  dur={d:.4f}s  f0={f0:.1f}  lyric={lyr_str}")

    # ── 构建音素序列 ──
    all_phonemes = []
    seg_ph_counts = []  # 每个 segment 的音素数
    last_vowel = None  # 跟踪前一音节的尾元音, 用于 + 连音
    for typ, dur, f0_val, lyr in segments:
        if typ == 'note':
            phs = get_phonemes(lyr, last_vowel)
            # 更新 last_vowel: 取音素列表的最后一个元素（尾元音）
            if lyr != '+' and len(phs) > 0:
                last_vowel = phs[-1]
        else:
            phs = ['SP']
        all_phonemes.extend(phs)
        seg_ph_counts.append(len(phs))

    ph_ids = [pm['p2i'].get(p, 0) for p in all_phonemes]
    print(f"总音素数: {len(all_phonemes)}, 音素: {all_phonemes}")

    # ── 分帧 ──
    # 每个 segment 的帧数
    seg_frames = [max(1, int(dur * FRAME_RATE)) for _, dur, _, _ in segments]
    total_frames = sum(seg_frames)

    # 元音/辅音分类（元音=长帧，辅音=短帧，按权重分配）
    VOWELS = {'ue','ui','ai','e','ei','ian','eng','v','un','iang','ing',
              'ong','an','i','ao','en','a','o','er','u','ang','ou'}
    CONSONANT_WEIGHT = 0.2   # 辅音占音符 20%
    VOWEL_WEIGHT = 0.8        # 元音占音符 80%

    # 音素展开帧数 (按辅音/元音比例分配，非均分)
    ph_dur_frames = []
    for s_idx, n_ph in enumerate(seg_ph_counts):
        nf = seg_frames[s_idx]
        if n_ph <= 1:
            ph_dur_frames.append(nf)
            continue
        # 从 all_phonemes 取当前 segment 对应的音素
        ph_start = sum(seg_ph_counts[:s_idx])
        ph_list = all_phonemes[ph_start:ph_start + n_ph]

        # 分类: 多少辅音/元音
        n_cons = sum(1 for p in ph_list if p not in VOWELS and p != 'SP')
        n_vows = n_ph - n_cons

        if n_cons == 0 or n_vows == 0:
            # 全辅音或全元音: 均分
            base = nf // n_ph
            rem = nf % n_ph
            for j in range(n_ph):
                ph_dur_frames.append(base + (1 if j < rem else 0))
        else:
            # 辅音帧 + 元音帧, 按权重分配
            cons_weight_total = n_cons * CONSONANT_WEIGHT
            vow_weight_total = n_vows * VOWEL_WEIGHT
            total_weight = cons_weight_total + vow_weight_total
            cons_frames_total = max(1, int(nf * cons_weight_total / total_weight))
            vow_frames_total = nf - cons_frames_total

            # 辅音间均分
            cons_per = cons_frames_total // n_cons
            cons_rem = cons_frames_total % n_cons
            cons_durs = [cons_per + (1 if i < cons_rem else 0) for i in range(n_cons)]

            # 元音间均分
            vow_per = vow_frames_total // n_vows
            vow_rem = vow_frames_total % n_vows
            vow_durs = [vow_per + (1 if i < vow_rem else 0) for i in range(n_vows)]

            # 按原始顺序插回
            ci = 0; vi = 0
            for p in ph_list:
                if p not in VOWELS and p != 'SP':
                    ph_dur_frames.append(cons_durs[ci]); ci += 1
                else:
                    ph_dur_frames.append(vow_durs[vi]); vi += 1

    total_ph_frames = sum(ph_dur_frames)

    # F0 帧 (完全对齐: 每个 segment 的帧数就是 seg_frames)
    f0_frames = []
    for s_idx, (typ, dur, f0_val, lyr) in enumerate(segments):
        nf = seg_frames[s_idx]
        f0_frames.extend([f0_val] * nf)

    f0_frames = np.array(f0_frames[:total_ph_frames], dtype=np.float32)
    assert len(f0_frames) == total_ph_frames, f"F0帧数({len(f0_frames)}) != 音素帧数({total_ph_frames})"

    # 打印详细信息
    print(f"总帧数: {total_ph_frames}")
    print(f"seg_frames: {seg_frames}")
    print(f"音素时长(帧): {ph_dur_frames}")
    print(f"帧率: {FRAME_RATE:.2f} Hz")
    print(f"预计音频时长: {total_ph_frames / FRAME_RATE:.3f}s")

    # ── 构建 tensor ──
    phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).to(args.device)
    dur_t = torch.LongTensor(ph_dur_frames).unsqueeze(0).to(args.device)
    f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).to(args.device)
    # note_lengths 仅包含音符段 (不含休止), 对齐训练时 batch['note_lengths']
    note_frames = [seg_frames[i] for i, (typ, _, _, _) in enumerate(segments) if typ == 'note']
    nl_t = torch.LongTensor(note_frames).unsqueeze(0).to(args.device)

    # ── 推理 ──
    print(f"推理中 (n_steps={args.n_steps})...")
    with torch.no_grad():
        waveform, mu, z, f0_refined, lag_pred, dur_pred = model.forward_synthesis(
            phoneme_t, dur_t, f0_t, n_steps=args.n_steps, note_lengths=nl_t,
            predict_durations=True
        )

    wav = waveform.squeeze().cpu().numpy()
    wav = wav / (np.abs(wav).max() + 1e-8)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    sf.write(args.output, wav, sr)
    duration_sec = len(wav) / sr
    print(f"合成完成: {args.output} ({duration_sec:.3f}s, sr={sr}Hz)")


if __name__ == '__main__':
    main()
