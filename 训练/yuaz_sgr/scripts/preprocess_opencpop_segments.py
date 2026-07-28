# YUAZ SGR
import os, sys, argparse, json
import numpy as np
import librosa
from pathlib import Path

SAMPLE_RATE = 24000
HOP_LENGTH = 256
SEGMENT_SEC = 4
NORM_DB = -26
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH


NOTE_TO_MIDI = {
    'C': 60, 'C#': 61, 'Db': 61,
    'D': 62, 'D#': 63, 'Eb': 63,
    'E': 64, 'Fb': 64,
    'F': 65, 'F#': 66, 'Gb': 66,
    'G': 67, 'G#': 68, 'Ab': 68,
    'A': 69, 'A#': 70, 'Bb': 70,
    'B': 71, 'Cb': 71,
}


def pitch_to_hz(pitch_str):
    pitch_str = pitch_str.strip()
    if pitch_str.lower() == 'rest':
        return 0.0
    if '/' in pitch_str:
        pitch_str = pitch_str.split('/')[0]
    if len(pitch_str) < 2:
        return 0.0
    note_name = pitch_str[0].upper()
    accidental = ''
    rest = pitch_str[1:]
    if rest and rest[0] in '#b':
        accidental = rest[0]
        octave_str = rest[1:]
    else:
        octave_str = rest
    full_note = note_name + accidental
    octave = int(octave_str)
    midi = NOTE_TO_MIDI.get(full_note, 60) + (octave - 4) * 12
    return 440.0 * (2 ** ((midi - 69) / 12))


def load_audio_norm(wav_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-8:
        audio *= (10 ** (NORM_DB / 20)) / rms
    return audio


def parse_transcriptions_line(line):
    parts = line.strip().split('|')
    if len(parts) != 7:
        return None
    wav_name = parts[0]
    phonemes = parts[2].split()
    notes = parts[3].split()
    note_durations = [float(x) for x in parts[4].split()]
    ph_durations = [float(x) for x in parts[5].split()]
    slurs = [int(x) for x in parts[6].split()]
    phonemes = ['_h' if p in ('AP', 'SP') else p for p in phonemes]
    n_ph = len(phonemes)
    n_notes = len(notes)
    n_nd = len(note_durations)
    n_pd = len(ph_durations)
    n_sl = len(slurs)
    if not (n_ph == n_notes == n_nd == n_pd == n_sl):
        print(f"  [WARN] {wav_name}: field length mismatch ph={n_ph} notes={n_notes} nd={n_nd} pd={n_pd} sl={n_sl}")
        return None
    return {
        'wav_name': wav_name,
        'phonemes': phonemes,
        'notes': notes,
        'note_durations': note_durations,
        'ph_durations': ph_durations,
        'slurs': slurs,
    }


def build_note_groups(notes, slurs):
    groups = []
    current = []
    for i in range(len(slurs)):
        if slurs[i] == 0:
            if current:
                groups.append(current)
            current = [i]
        else:
            current.append(i)
    if current:
        groups.append(current)
    return groups


def process_segment(data, sr=SAMPLE_RATE):
    phonemes = data['phonemes']
    notes = data['notes']
    note_durations = data['note_durations']
    ph_durations = data['ph_durations']
    slurs = data['slurs']
    groups = build_note_groups(notes, slurs)
    frame_phonemes = []
    frame_durations = []
    frame_f0 = []
    frame_lags = []
    note_lengths_frames = []
    cum_note_dur = 0.0
    cum_ph_dur = 0.0
    for group in groups:
        total_note_dur = sum(note_durations[i] for i in group)
        note_len_frames = max(1, int(round(total_note_dur * FRAME_RATE)))
        note_group_start = cum_note_dur
        first_ph_start = cum_ph_dur
        lag = first_ph_start - note_group_start  # 论文 §3.3.1: 辅音可在音符 onset 前（负 lag 有意义）
        lag_frames = int(round(lag * FRAME_RATE))
        if lag_frames >= note_len_frames:
            lag_frames = note_len_frames - 1 if note_len_frames > 1 else 0
        l_hat_frames = max(1, note_len_frames - lag_frames)
        note_name = notes[group[0]]
        if note_name.lower() == 'rest':
            note_freq = 0.0
        else:
            note_freq = pitch_to_hz(note_name)
        group_ph_durs = [ph_durations[i] for i in group]
        total_ph_dur_raw = sum(group_ph_durs)
        raw_frames = [max(1, int(round(pd * FRAME_RATE))) for pd in group_ph_durs]
        total_raw = sum(raw_frames)
        remainder = l_hat_frames
        for j, idx in enumerate(group):
            if j == len(group) - 1:
                s = remainder
            else:
                s = max(1, int(round(raw_frames[j] / total_raw * l_hat_frames)))
                remainder -= s
            frame_phonemes.append(phonemes[idx])
            frame_durations.append(s)
        frame_f0.extend([note_freq] * l_hat_frames)
        frame_lags.extend([float(lag_frames)] * l_hat_frames)
        note_lengths_frames.append(note_len_frames)
        cum_note_dur += total_note_dur
        cum_ph_dur += total_ph_dur_raw
    frame_phonemes_arr = np.array(frame_phonemes, dtype=object)
    durations_arr = np.array(frame_durations, dtype=np.int32)
    f0_arr = np.array(frame_f0, dtype=np.float32)
    lags_arr = np.array(frame_lags, dtype=np.float32)
    note_lengths_arr = np.array(note_lengths_frames, dtype=np.int32)
    return frame_phonemes_arr, durations_arr, f0_arr, lags_arr, note_lengths_arr


def pad_to_segment(audio, f0, lags, target_samples, target_frames):
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        audio = audio[:target_samples]
    if len(f0) < target_frames:
        f0 = np.pad(f0, (0, target_frames - len(f0)))
    else:
        f0 = f0[:target_frames]
    if len(lags) < target_frames:
        lags = np.pad(lags, (0, target_frames - len(lags)))
    else:
        lags = lags[:target_frames]
    return audio, f0, lags


def main():
    parser = argparse.ArgumentParser(description='OpenCPOP Segments 预处理')
    parser.add_argument('--opencpop_dir', type=str, required=True,
                        help='OpenCPOP 数据集根目录 (含 segments/ 和 wavs/)')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed',
                        help='预处理输出目录')
    parser.add_argument('--sr', type=int, default=SAMPLE_RATE,
                        help='目标采样率 (默认 24000)')
    args = parser.parse_args()
    opencpop_dir = Path(args.opencpop_dir)
    output_dir = Path(args.output_dir)
    sr = args.sr
    target_samples = SEGMENT_SEC * sr
    target_frames = int(round(SEGMENT_SEC * FRAME_RATE))
    seg_dir = opencpop_dir / 'segments'
    wav_dir = seg_dir / 'wavs'
    trans_path = seg_dir / 'transcriptions.txt'
    train_list_path = seg_dir / 'train.txt'
    test_list_path = seg_dir / 'test.txt'
    if not trans_path.exists():
        print(f"错误: {trans_path} 不存在")
        return
    if not train_list_path.exists():
        print(f"错误: {train_list_path} 不存在")
        return
    if not test_list_path.exists():
        print(f"错误: {test_list_path} 不存在")
        return
    if not wav_dir.exists():
        print(f"错误: {wav_dir} 不存在")
        return
    print(f"加载标注: {trans_path}")
    transcriptions = {}
    with open(trans_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_transcriptions_line(line)
            if parsed is not None:
                transcriptions[parsed['wav_name']] = parsed
    print(f"  共 {len(transcriptions)} 条标注")
    print(f"加载划分: {train_list_path}")
    train_names = []
    with open(train_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip().split('|')[0]
            if name:
                train_names.append(name)
    print(f"  训练集: {len(train_names)} 条")
    print(f"加载划分: {test_list_path}")
    test_names = []
    with open(test_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip().split('|')[0]
            if name:
                test_names.append(name)
    print(f"  测试集: {len(test_names)} 条")
    all_phonemes = set()
    stats = {'train': 0, 'test': 0}

    def save_split(wav_names, split_name):
        split_dir = output_dir / split_name
        os.makedirs(split_dir, exist_ok=True)
        saved = 0
        for wav_name in wav_names:
            if wav_name not in transcriptions:
                print(f"  [SKIP] {wav_name}: 不在标注中")
                continue
            wav_path = wav_dir / f'{wav_name}.wav'
            if not wav_path.exists():
                alt_path = wav_dir / f'{wav_name}.WAV'
                if alt_path.exists():
                    wav_path = alt_path
                else:
                    print(f"  [SKIP] {wav_name}: WAV 文件不存在 {wav_path}")
                    continue
            try:
                audio = load_audio_norm(str(wav_path), sr=sr)
            except Exception as e:
                print(f"  [ERROR] {wav_name}: 加载音频失败 {e}")
                continue
            data = transcriptions[wav_name]
            exp_audio_len = sum(data['ph_durations'])
            actual_audio_len = len(audio) / sr
            if abs(actual_audio_len - exp_audio_len) > 0.5:
                print(f"  [WARN] {wav_name}: 音频时长 {actual_audio_len:.3f}s 与标注 {exp_audio_len:.3f}s 差异较大")
            try:
                phonemes_arr, durations_arr, f0_arr, lags_arr, note_lengths_arr = process_segment(data, sr=sr)
            except Exception as e:
                print(f"  [ERROR] {wav_name}: 处理失败 {e}")
                continue
            audio_pad, f0_pad, lags_pad = pad_to_segment(
                audio, f0_arr, lags_arr, target_samples, target_frames
            )
            out_path = split_dir / f'{wav_name}.npz'
            np.savez(out_path,
                     audio=audio_pad.astype(np.float32),
                     phonemes=phonemes_arr,
                     durations=durations_arr,
                     f0=f0_pad,
                     lags=lags_pad,
                     note_lengths=note_lengths_arr,
                     song_id=wav_name)
            for p in phonemes_arr.tolist():
                all_phonemes.add(p)
            saved += 1
            if saved % 100 == 0:
                print(f"  [{split_name}] 已处理 {saved} 段...")
        return saved

    print("\n处理训练集...")
    stats['train'] = save_split(train_names, 'train')
    print(f"\n处理测试集...")
    stats['test'] = save_split(test_names, 'test')
    p2i = {p: i for i, p in enumerate(sorted(all_phonemes))}
    i2p = {str(i): p for p, i in p2i.items()}
    map_path = output_dir / 'phoneme_map.json'
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump({'p2i': p2i, 'i2p': i2p, 'num_phonemes': len(p2i)}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n预处理完成!")
    print(f"  训练: {stats['train']} 段")
    print(f"  测试: {stats['test']} 段")
    print(f"  音素数: {len(p2i)}")
    print(f"  音素表: {map_path}")
    total = stats['train'] + stats['test']
    avg_ph_per_seg = sum(len(transcriptions[n]['phonemes']) for n in train_names if n in transcriptions) / max(1, stats['train'])
    print(f"  平均每段音素数: {avg_ph_per_seg:.1f}")
    print(f"  总段数: {total}")


if __name__ == '__main__':
    main()
