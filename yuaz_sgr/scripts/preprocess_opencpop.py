# YUAZ SGR
"""
scripts/preprocess_opencpop.py
OpenCPOP 数据集预处理流水线

论文 §4.1: 24kHz 降采样, -26dB 归一化, 4秒分段, 词级边界对齐
数据流: WAV(44.1k) + TextGrid(音素) + MIDI(音符) → 对齐 → 分段 → .npy
"""
import os, sys, argparse, json
import numpy as np
import librosa
import textgrid
import pretty_midi
import torch
import torchcrepe
from pathlib import Path
from collections import defaultdict

SAMPLE_RATE = 24000
HOP_LENGTH = 256
SEGMENT_SEC = 4
NORM_DB = -26
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # 93.75 Hz


def extract_crepe_f0(audio):
    """用 CREPE 从音频提取真实 F0 曲线（4秒分片防OOM）
    CREPE 自带置信度：未检测到时输出 <50Hz，训练时由 >50 滤波"""
    audio_t = torch.FloatTensor(audio).unsqueeze(0)
    n_frames_full = len(audio) // HOP_LENGTH + 1
    f0_full = torch.zeros(n_frames_full, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    chunk_sec = 4
    chunk_samples = chunk_sec * SAMPLE_RATE
    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, len(audio))
            chunk = audio_t[:, start:end]
            f0_chunk = torchcrepe.predict(chunk, SAMPLE_RATE, hop_length=HOP_LENGTH,
                                          fmin=50.0, fmax=2006.0,
                                          model='full', device='cuda', pad=True)
            f0_chunk = f0_chunk[0]
            c_start = start // HOP_LENGTH
            c_end = c_start + f0_chunk.shape[0]
            if c_end > n_frames_full:
                c_end = n_frames_full
                f0_chunk = f0_chunk[:c_end - c_start]
            f0_full[c_start:c_end] = f0_chunk
    
    return f0_full.cpu().numpy().astype(np.float32)


def load_audio(wav_path, sr=SAMPLE_RATE):
    """加载音频, 降采样到 24kHz, -26dB 归一化"""
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    rms = np.sqrt(np.mean(audio ** 2))
    audio *= (10 ** (NORM_DB / 20)) / (rms + 1e-8)
    return audio


def load_phonemes(tg_path):
    """从 TextGrid 提取音素序列 + 时间边界"""
    tg = textgrid.TextGrid.fromFile(tg_path)
    phones_tier = tg.getFirst('phones')
    if phones_tier is None:
        # 尝试其他可能的 tier 名称
        for name in ['phones', 'phoneme', 'phonemes', 'dian', 'text', '音素', '音节']:
            try:
                phones_tier = tg.getFirst(name)
                if phones_tier is not None:
                    break
            except:
                continue
    if phones_tier is None:
        raise ValueError(f"无法在 {tg_path} 中找到音素层")

    phonemes = []
    intervals = []
    for iv in phones_tier:
        if iv.mark and iv.mark.strip():
            phonemes.append(iv.mark.strip())
            intervals.append((iv.minTime, iv.maxTime))
    return phonemes, intervals


def load_notes(midi_path):
    """从 MIDI 提取音符 F0 + 起止时间"""
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
    """
    将音素对齐到音符, 计算 time-lag 和 note_length

    论文 §3.3.1: time-lag = 音符 onset 与实际音素起始的偏移
    论文 §3.3.2: L̂_n = L_n - lag_n
    """
    aligned = []
    phone_idx = 0

    for note in notes:
        note_start = note['start']
        note_end = note['end']
        note_f0 = note['f0']
        note_phones = []

        while phone_idx < len(phonemes):
            ph_start, ph_end = phone_intervals[phone_idx]
            # 音素在音符范围内
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
            # 音符真实时长 (秒)
            note_dur_sec = note_end - note_start
            # time-lag: 第一个音素起始 vs 音符起始
            first_phone_start = note_phones[0]['start']
            lag = first_phone_start - note_start  # 论文 §3.3.1: 辅音前置（负 lag 正常）

            aligned.append({
                'f0': note_f0,
                'pitch': note['pitch'],
                'note_start': note_start,
                'note_end': note_end,
                'note_dur_sec': note_dur_sec,
                'lag': lag,
                'phones': note_phones,
            })

    return aligned


def compute_phone_durations(aligned):
    """
    计算帧级音素时长和 time-lag
    返回: 帧对齐的 phoneme_ids, durations(帧), F0(帧), lags(帧), note_lengths(帧)
    """
    frame_phonemes = []
    frame_durations = []
    frame_f0 = []
    frame_lags = []
    note_lengths = []

    for note in aligned:
        note_f0 = note['f0']
        lag = note['lag']
        note_dur_sec = note['note_dur_sec']

        note_len_frames = max(1, int(note_dur_sec * FRAME_RATE))
        lag_frames = int(lag * FRAME_RATE)
        l_hat_frames = max(1, note_len_frames - lag_frames)

        if note['phones']:
            raw_frames = [max(1, int((ph['end'] - ph['start']) * FRAME_RATE)) for ph in note['phones']]
            total_raw = sum(raw_frames)
            scaled = []
            remainder = max(1, l_hat_frames)
            for j, rf in enumerate(raw_frames):
                if j == len(raw_frames) - 1:
                    s = max(1, remainder)
                else:
                    s = max(1, int(rf / total_raw * l_hat_frames))
                    remainder -= s
                scaled.append(s)
                frame_phonemes.append(note['phones'][j]['phoneme'])
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
    """
    论文 §4.1: 4秒分段, 边界对齐词级

    修复 #40: 在音符边界处切分, 收集连续音符直到累计时长接近 seg_sec,
    在最近音符边界切割, 确保词级边界对齐

    论文: "segmentation boundaries aligned to word-level timing
          and natural pauses to preserve linguistic and prosodic coherence"
    """
    seg_samples_target = seg_sec * sr
    seg_frames_target = seg_sec * int(FRAME_RATE)

    segments = []
    current_note_idx = 0
    total_notes = len(aligned)

    while current_note_idx < total_notes:
        # 收集音符直到累计帧数接近目标
        note_start_idx = current_note_idx
        accum_frames = 0
        accum_samples = 0

        while current_note_idx < total_notes:
            note = aligned[current_note_idx]
            note_dur_frames = max(1, int(note['note_dur_sec'] * FRAME_RATE))
            note_dur_samples = max(1, int(note['note_dur_sec'] * sr))

            # 如果加上当前音符会超过目标太多, 且已有足够内容, 则在当前音符前切割
            if accum_frames > 0 and accum_frames + note_dur_frames > seg_frames_target * 1.5:
                break

            accum_frames += note_dur_frames
            accum_samples += note_dur_samples
            current_note_idx += 1

        # 切割范围: note_start_idx ~ current_note_idx
        note_end_idx = current_note_idx

        # 计算音符累计的 frame 和 sample 范围
        # 音频边界截取 — 用 absolute note_start/note_end 而非累计时长
        note_start_sec = aligned[note_start_idx]['note_start']
        note_end_sec = aligned[note_end_idx - 1]['note_end']
        note_start_sample = max(0, int(note_start_sec * sr))
        note_end_sample = min(len(audio), int(note_end_sec * sr))
        seg_audio = audio[note_start_sample:note_end_sample]
        if len(seg_audio) < HOP_LENGTH * 10:
            continue

        # 累计帧数对齐 — 使用 L̂_n = L_n - lag_n 匹配 compute_phone_durations
        accum_frames = 0
        for n in range(note_start_idx):
            note_dur_sec = aligned[n]['note_dur_sec']
            note_len_frames = max(1, int(note_dur_sec * FRAME_RATE))
            lag_frames = int(aligned[n]['lag'] * FRAME_RATE)
            l_hat_frames = max(1, note_len_frames - lag_frames)
            accum_frames += l_hat_frames

        seg_frames = 0
        for n in range(note_start_idx, note_end_idx):
            note_dur_sec = aligned[n]['note_dur_sec']
            note_len_frames = max(1, int(note_dur_sec * FRAME_RATE))
            lag_frames = int(aligned[n]['lag'] * FRAME_RATE)
            l_hat_frames = max(1, note_len_frames - lag_frames)
            seg_frames += l_hat_frames

        # F0 索引用绝对时间对齐（修复：CREPE F0 按音频帧索引，非音符累计帧）
        f0_start = int(note_start_sec * FRAME_RATE)
        f0_end = int(note_end_sec * FRAME_RATE)
        # lags 索引用音符累计帧（lags 来源 compute_phone_durations，同体系）
        lag_f0_start = accum_frames
        lag_f0_end = lag_f0_start + seg_frames

        f0_start = min(f0_start, len(f0_curve) - 1)
        f0_end = min(f0_end, len(f0_curve))
        lag_f0_start = min(lag_f0_start, len(lags) - 1)
        lag_f0_end = min(lag_f0_end, len(lags))

        seg_f0 = f0_curve[f0_start:f0_end] if f0_start < f0_end else np.zeros(seg_frames, dtype=np.float32)
        seg_lags = lags[lag_f0_start:lag_f0_end] if lag_f0_start < len(lags) and lag_f0_end <= len(lags) else np.zeros(seg_frames, dtype=np.float32)

        seg_note_lengths = [
            max(1, int(aligned[n]['note_dur_sec'] * FRAME_RATE)
                - int(aligned[n]['lag'] * FRAME_RATE))
            for n in range(note_start_idx, note_end_idx)
        ]

        # 计算当前段涉及哪些音素
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
    """构建音素到 ID 的映射表"""
    unique = sorted(set(all_phonemes))
    return {p: i for i, p in enumerate(unique)}, {i: p for i, p in enumerate(unique)}


def preprocess_song(wav_path, tg_path, midi_path, output_dir, song_id):
    """处理单首歌曲"""
    # 1. 加载音频
    audio = load_audio(wav_path)

    # 2. 用 CREPE 从音频提取真实 F0（修复：替换平坦 MIDI F0）
    f0_curve = extract_crepe_f0(audio)

    # 3. 加载音素
    phonemes, phone_intervals = load_phonemes(tg_path)

    # 4. 加载音符
    notes = load_notes(midi_path)

    # 5. 对齐
    aligned = align_phonemes_to_notes(phonemes, phone_intervals, notes)

    # 6. 计算帧级特征（f0_curve 忽略 compute_phone_durations 的平坦值，用 CREPE）
    frame_phones, durations, _, lags, note_lengths = compute_phone_durations(aligned)

    # 7. 分段 (词级边界对齐)
    segments = segment_audio(audio, frame_phones, durations, f0_curve, lags, note_lengths, aligned)

    # 7. 保存
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
    parser = argparse.ArgumentParser(description='OpenCPOP 预处理')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='OpenCPOP 数据集根目录')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed',
                        help='预处理输出目录')
    parser.add_argument('--test_songs', type=int, default=3,
                        help='测试集歌曲数')
    parser.add_argument('--val_songs', type=int, default=3,
                        help='验证集歌曲数')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 收集所有歌曲
    wav_files = sorted(data_dir.glob('**/*.wav'))
    songs = []
    for wav_path in wav_files:
        song_id = wav_path.stem
        tg_path = wav_path.with_suffix('.TextGrid')
        if not tg_path.exists():
            # 尝试其他扩展名和子目录
            for ext in ['.textgrid', '.TextGrid', '.tg']:
                for search_dir in [wav_path.parent] + list(data_dir.glob('*/')):
                    candidate = search_dir / f'{song_id}{ext}'
                    if candidate.exists():
                        tg_path = candidate
                        break
                if tg_path.exists():
                    break
        midi_path = wav_path.with_suffix('.mid')
        if not midi_path.exists():
            for ext in ['.mid', '.midi']:
                for search_dir in [wav_path.parent] + list(data_dir.glob('*/')):
                    candidate = search_dir / f'{song_id}{ext}'
                    if candidate.exists():
                        midi_path = candidate
                        break
                if midi_path.exists():
                    break

        if tg_path.exists() and midi_path.exists():
            songs.append((song_id, str(wav_path), str(tg_path), str(midi_path)))

    print(f"找到 {len(songs)} 首带标注的歌曲")

    # 划分数据集
    test_ids = set(s[0] for s in songs[:args.test_songs])
    val_ids = set(s[0] for s in songs[args.test_songs:args.test_songs + args.val_songs])

    all_phonemes = []
    stats = {'train': 0, 'val': 0, 'test': 0}

    for song_id, wav_path, tg_path, midi_path in songs:
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
            print(f"  [{split}] {song_id}: {n_segs} 段")
        except Exception as e:
            print(f"  [ERROR] {song_id}: {e}")

    # 构建音素映射
    p2i, i2p = build_phoneme_map(all_phonemes)
    map_path = output_dir / 'phoneme_map.json'
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump({'p2i': p2i, 'i2p': i2p, 'num_phonemes': len(p2i)}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n预处理完成: 训练 {stats['train']} 段, 验证 {stats['val']} 段, 测试 {stats['test']} 段")
    print(f"音素数: {len(p2i)}, 音素表: {map_path}")


if __name__ == '__main__':
    main()