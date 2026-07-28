"""
适配 OpenCPOP 完整数据集的预处理脚本
OpenCPOP 目录结构:
  wavs/       — 100 首完整歌曲 WAV (44.1kHz)
  textgrids/  — 100 个音素标注 TextGrid
  midis/      — 100 个音符 MIDI

输出: 24kHz 重采样 → -26dB 归一化 → 4秒分段 → .npz
"""
import os, sys, argparse, json
import numpy as np
import librosa
import textgrid
import pretty_midi
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_RATE = 24000
HOP_LENGTH = 256
SEGMENT_SEC = 4
NORM_DB = -26
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH


def load_audio(wav_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    rms = np.sqrt(np.mean(audio ** 2))
    audio *= (10 ** (NORM_DB / 20)) / (rms + 1e-8)
    return audio


def load_phonemes(tg_path):
    tg = textgrid.TextGrid.fromFile(tg_path)
    phones_tier = tg.getFirst('phones')
    if phones_tier is None:
        for name in ['phones', 'phoneme', 'phonemes', 'dian', 'text', '音素', 'pinyin']:
            try:
                phones_tier = tg.getFirst(name)
                if phones_tier is not None:
                    break
            except:
                continue
    if phones_tier is None:
        raise ValueError(f"找不到音素层: {tg_path}")
    phonemes = []
    intervals = []
    for iv in phones_tier:
        if iv.mark and iv.mark.strip():
            phonemes.append(iv.mark.strip())
            intervals.append((iv.minTime, iv.maxTime))
    return phonemes, intervals


def load_notes(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            notes.append({
                'f0': 440.0 * (2 ** ((note.pitch - 69) / 12)),
                'start': note.start,
                'end': note.end,
                'pitch': note.pitch,
            })
    notes.sort(key=lambda n: n['start'])
    return notes


def align_phonemes_to_notes(phonemes, phone_intervals, notes):
    aligned = []
    phone_idx = 0
    for note in notes:
        note_start = note['start']
        note_end = note['end']
        note_f0 = note['f0']
        note_phones = []
        while phone_idx < len(phonemes):
            ph_start, ph_end = phone_intervals[phone_idx]
            if ph_end <= note_start:
                phone_idx += 1
                continue
            if ph_start >= note_end:
                break
            note_phones.append({
                'phoneme': phonemes[phone_idx],
                'start': ph_start,
                'end': ph_end,
            })
            phone_idx += 1
        if note_phones:
            note_dur_sec = note_end - note_start
            first_phone_start = note_phones[0]['start']
            lag = first_phone_start - note_start  # 论文 §3.3.1: 辅音前置（负 lag 正常）
            aligned.append({
                'f0': note_f0, 'pitch': note['pitch'],
                'note_start': note_start, 'note_end': note_end,
                'note_dur_sec': note_dur_sec, 'lag': lag,
                'phones': note_phones,
            })
    return aligned


def compute_phone_durations(aligned):
    frame_phonemes = []
    frame_durations = []
    frame_f0 = []
    frame_lags = []
    note_lengths = []
    for note in aligned:
        note_f0 = note['f0']; lag = note['lag']
        note_dur_sec = note['note_dur_sec']
        note_len_frames = max(1, int(note_dur_sec * FRAME_RATE))
        lag_frames = int(lag * FRAME_RATE)
        l_hat_frames = note_len_frames - lag_frames
        if note['phones']:
            raw_frames = [max(1, int((ph['end'] - ph['start']) * FRAME_RATE)) for ph in note['phones']]
            total_raw = sum(raw_frames)
            remainder = l_hat_frames
            for j, ph in enumerate(note['phones']):
                if j == len(note['phones']) - 1:
                    s = remainder
                else:
                    s = max(1, int(raw_frames[j] / total_raw * l_hat_frames))
                    remainder -= s
                frame_phonemes.append(ph['phoneme'])
                frame_durations.append(s)
        else:
            frame_phonemes.append('_')
            frame_durations.append(l_hat_frames)
        frame_f0.extend([note_f0] * l_hat_frames)
        frame_lags.extend([lag_frames] * l_hat_frames)
        note_lengths.append(note_len_frames)
    return frame_phonemes, frame_durations, frame_f0, frame_lags, note_lengths


def segment_audio(audio, phoneme_ids, durations, f0_curve, lags, note_lengths,
                  aligned, seg_sec=SEGMENT_SEC, sr=SAMPLE_RATE):
    seg_samples_target = seg_sec * sr
    seg_frames_target = seg_sec * int(FRAME_RATE)
    segments = []
    current_note_idx = 0
    total_notes = len(aligned)
    while current_note_idx < total_notes:
        note_start_idx = current_note_idx
        accum_frames = 0
        while current_note_idx < total_notes:
            note = aligned[current_note_idx]
            note_dur_frames = max(1, int(note['note_dur_sec'] * FRAME_RATE))
            if accum_frames > 0 and accum_frames + note_dur_frames > seg_frames_target * 1.5:
                break
            accum_frames += note_dur_frames
            current_note_idx += 1
        note_end_idx = current_note_idx
        # 计算采样点范围
        note_start_sample = 0
        for n in range(note_start_idx):
            note_start_sample += max(1, int(aligned[n]['note_dur_sec'] * sr))
        accum_samples = 0
        for n in range(note_start_idx, note_end_idx):
            accum_samples += max(1, int(aligned[n]['note_dur_sec'] * sr))
        note_end_sample = note_start_sample + accum_samples
        seg_audio = audio[note_start_sample:note_end_sample]
        if len(seg_audio) < HOP_LENGTH * 10:
            continue
        # 帧级计算
        accum_frames = 0
        for n in range(note_start_idx):
            accum_frames += max(1, int(aligned[n]['note_dur_sec'] * FRAME_RATE))
        seg_frames = sum(
            max(1, int(aligned[n]['note_dur_sec'] * FRAME_RATE))
            for n in range(note_start_idx, note_end_idx)
        )
        f0_start = accum_frames
        f0_end = f0_start + seg_frames
        f0_start = min(f0_start, len(f0_curve) - 1)
        f0_end = min(f0_end, len(f0_curve))
        seg_f0 = f0_curve[f0_start:f0_end] if f0_start < f0_end else np.zeros(seg_frames, dtype=np.float32)
        seg_lags = lags[f0_start:f0_end] if f0_start < len(lags) and f0_end <= len(lags) else np.zeros(seg_frames, dtype=np.float32)
        seg_note_lengths = [
            max(1, int(aligned[n]['note_dur_sec'] * FRAME_RATE))
            for n in range(note_start_idx, note_end_idx)
        ]
        ph_start = sum(len(aligned[n]['phones']) for n in range(note_start_idx) if aligned[n]['phones'])
        ph_end = ph_start + sum(len(aligned[n]['phones']) for n in range(note_start_idx, note_end_idx) if aligned[n]['phones'])
        segments.append({
            'audio': seg_audio,
            'phoneme_ids': phoneme_ids[ph_start:ph_end],
            'durations': durations[ph_start:ph_end],
            'f0_curve': np.array(seg_f0, dtype=np.float32),
            'lags': np.array(seg_lags, dtype=np.float32),
            'note_lengths': np.array(seg_note_lengths, dtype=np.int32),
        })
    return segments


def build_phoneme_map(all_phonemes):
    unique = sorted(set(all_phonemes))
    return {p: i for i, p in enumerate(unique)}, {i: p for i, p in enumerate(unique)}


def preprocess_song(wav_path, tg_path, midi_path, output_dir, song_id):
    audio = load_audio(wav_path)
    phonemes, phone_intervals = load_phonemes(tg_path)
    notes = load_notes(midi_path)
    if not notes:
        print(f"  [WARN] {song_id}: MIDI 无音符, 跳过")
        return 0, []
    aligned = align_phonemes_to_notes(phonemes, phone_intervals, notes)
    if not aligned:
        print(f"  [WARN] {song_id}: 对齐后无音符, 跳过")
        return 0, []
    frame_phones, durations, f0_curve, lags, note_lengths = compute_phone_durations(aligned)
    segments = segment_audio(audio, frame_phones, durations, f0_curve, lags, note_lengths, aligned)
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for i, seg in enumerate(segments):
        if len(seg['audio']) < HOP_LENGTH * 10:
            continue
        out_path = os.path.join(output_dir, f"{song_id}_{i:04d}.npz")
        np.savez(out_path,
                 audio=seg['audio'].astype(np.float32),
                 phonemes=np.array(seg['phoneme_ids'], dtype=object),
                 durations=np.array(seg['durations'], dtype=np.int32),
                 f0=np.array(seg['f0_curve'], dtype=np.float32),
                 lags=np.array(seg['lags'], dtype=np.float32),
                 note_lengths=np.array(seg['note_lengths'], dtype=np.int32),
                 song_id=song_id)
        saved += 1
    return saved, frame_phones


def main():
    parser = argparse.ArgumentParser(description='OpenCPOP 完整预处理')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='OpenCPOP 根目录 (含 wavs/ textgrids/ midis/ 子目录)')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed',
                        help='预处理输出目录')
    parser.add_argument('--test_songs', type=int, default=5,
                        help='测试集歌曲数')
    parser.add_argument('--val_songs', type=int, default=5,
                        help='验证集歌曲数')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 只从 wavs/ 目录读取完整歌曲
    wav_dir = data_dir / 'wavs'
    tg_dir = data_dir / 'textgrids'
    midi_dir = data_dir / 'midis'

    if not wav_dir.exists():
        print(f"错误: {wav_dir} 不存在")
        return
    if not tg_dir.exists():
        print(f"错误: {tg_dir} 不存在")
        return
    if not midi_dir.exists():
        print(f"错误: {midi_dir} 不存在")
        return

    wav_files = sorted(wav_dir.glob('*.wav'))
    print(f"找到 {len(wav_files)} 首歌曲 WAV")

    songs = []
    for wav_path in wav_files:
        song_id = wav_path.stem
        tg_path = tg_dir / f'{song_id}.TextGrid'
        if not tg_path.exists():
            tg_path = tg_dir / f'{song_id}.textgrid'
        midi_path = midi_dir / f'{song_id}.midi'
        if not midi_path.exists():
            midi_path = midi_dir / f'{song_id}.mid'
        if tg_path.exists() and midi_path.exists():
            songs.append((song_id, str(wav_path), str(tg_path), str(midi_path)))
        else:
            print(f"  [SKIP] {song_id}: 缺标注 (tg={tg_path.exists()}, mid={midi_path.exists()})")

    print(f"有效歌曲: {len(songs)}/{len(wav_files)}")

    # 划分数据集
    test_ids = set(s[0] for s in songs[:args.test_songs])
    val_ids = set(s[0] for s in songs[args.test_songs:args.test_songs + args.val_songs])

    all_phonemes = []
    stats = {'train': 0, 'val': 0, 'test': 0}

    for idx, (song_id, wav_path, tg_path, midi_path) in enumerate(songs):
        if song_id in test_ids:
            split = 'test'
        elif song_id in val_ids:
            split = 'val'
        else:
            split = 'train'
        out_subdir = output_dir / split
        try:
            n_segs, phones = preprocess_song(wav_path, tg_path, midi_path,
                                             str(out_subdir), song_id)
            stats[split] += n_segs
            all_phonemes.extend(phones)
            print(f"  [{idx+1}/{len(songs)}] [{split}] {song_id}: {n_segs} 段")
        except Exception as e:
            print(f"  [ERROR] {song_id}: {e}")

    # 音素映射
    p2i, i2p = build_phoneme_map(all_phonemes)
    map_path = output_dir / 'phoneme_map.json'
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump({'p2i': p2i, 'i2p': i2p, 'num_phonemes': len(p2i)}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n预处理完成!")
    print(f"  训练: {stats['train']} 段")
    print(f"  验证: {stats['val']} 段")
    print(f"  测试: {stats['test']} 段")
    print(f"  音素数: {len(p2i)}")
    print(f"  音素表: {map_path}")


if __name__ == '__main__':
    main()
