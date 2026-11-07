# YUAZ SGR
"""scripts/preprocess_opencpop_v2"""
import os, sys, argparse, json
import numpy as np
import librosa
import torch
import torchcrepe
from pathlib import Path
from collections import defaultdict, OrderedDict

SAMPLE_RATE = 24000
HOP_LENGTH = 256
SEGMENT_SEC = 4
NORM_DB = -26
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # 93.75 Hz
TARGET_SAMPLES = SEGMENT_SEC * SAMPLE_RATE
TARGET_FRAMES = int(SEGMENT_SEC * FRAME_RATE) + 1  # 376

def extract_crepe_f0(audio, fmin=50.0, fmax=1100.0, confidence=0.30):
    """用 CREPE 从音频提取真实 F0 曲线 (4秒分片防 OOM) 关键修复 (否则约 28% 的帧会饱和在 fmax 上成为坏标签, 导致合成音高严重偏高): 1) 输入峰值归一化到 [-1,1] —— CREPE 实际期望的输入范围 2) return_periodicity 置信度门控: 低置信帧 f0=0 (清音/静音), 解码器谐波通路自动 gate 关 3) fmax 收到人声上限, 杜绝 argmax 落到频率轴端点"""
    n_frames_full = len(audio) // HOP_LENGTH + 1
    if len(audio) < 2048:
        return np.zeros(n_frames_full, dtype=np.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    peak = float(np.max(np.abs(audio))) + 1e-8
    audio_t = torch.FloatTensor(audio / peak).unsqueeze(0).to(device)

    chunk_sec = 4
    chunk_samples = chunk_sec * SAMPLE_RATE
    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    f0_full = np.zeros(n_frames_full, dtype=np.float32)

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, len(audio))
            chunk = audio_t[:, start:end]
            f0_chunk, pd_chunk = torchcrepe.predict(
                chunk, SAMPLE_RATE, hop_length=HOP_LENGTH,
                fmin=fmin, fmax=fmax,
                model='full', device=device,
                return_periodicity=True, pad=True)
            f0_chunk = f0_chunk[0].cpu().numpy().astype(np.float32)
            pd_chunk = pd_chunk[0].cpu().numpy()
            f0_chunk = np.where(pd_chunk >= confidence, f0_chunk, 0.0).astype(np.float32)
            f0_chunk = np.where((f0_chunk >= fmin) & (f0_chunk <= fmax),
                                f0_chunk, 0.0).astype(np.float32)
            c_start = start // HOP_LENGTH
            c_end = min(c_start + len(f0_chunk), n_frames_full)
            f0_full[c_start:c_end] = f0_chunk[:c_end - c_start]

    return f0_full

def load_audio(wav_path, sr=SAMPLE_RATE):
    """加载音频, 降采样到 24kHz, -26dB 归一化"""
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    rms = np.sqrt(np.mean(audio ** 2))
    audio *= (10 ** (NORM_DB / 20)) / (rms + 1e-8)
    return audio

def parse_transcription(line):
    """解析转录行 格式: id|lyrics|phonemes|notes|note_durations|phoneme_durations|slur_flags 返回: (seg_id, phonemes, note_names, note_durs_sec, phoneme_durs_sec)"""
    parts = line.strip().split('|')
    if len(parts) < 6:
        return None
    seg_id = parts[0]
    phonemes = parts[2].split()
    note_names = parts[3].split()
    note_durs = [float(x) for x in parts[4].split()]
    ph_durs = [float(x) for x in parts[5].split()]
    return seg_id, phonemes, note_names, note_durs, ph_durs

def note_to_f0(note_name):
    """音符名 → F0 (Hz). 如 'G#4/Ab4' → 415.3 Hz. rest → 0"""
    if note_name.lower() == 'rest':
        return 0.0
    name = note_name.split('/')[0]
    # 解析音名，如 C4 / G#4 / A#4
    if len(name) >= 2 and name[1] == '#':
        pitch_class = name[:2]
        octave = int(name[2:])
    elif len(name) >= 2 and name[1] == 'b':
        pitch_class = name[:2]
        octave = int(name[2:])
    else:
        pitch_class = name[0]
        octave = int(name[1:])

    semitones = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    midi = 12 * (octave + 1) + semitones.get(pitch_class, 0)
    return 440.0 * (2 ** ((midi - 69) / 12))

def compute_frame_features(transcription, f0_curve, audio):
    """从转录数据计算帧级特征 返回: frame_phonemes, frame_durations, frame_f0, frame_lags, note_lengths"""
    seg_id, phonemes, note_names, note_durs_sec, ph_durs_sec = transcription
    audio_dur_sec = len(audio) / SAMPLE_RATE
    frame_phonemes = []
    frame_durations = []
    frame_f0 = []       # MIDI note F0
    frame_lags = []
    note_lengths = []

    pos_sec = 0.0
    cum_note_sec = 0.0   # 音符时间线累计
    cum_ph_sec = 0.0     # 音素时间线累计

    # 每音素帧数
    ph_frames = []
    for i in range(len(phonemes)):
        if i < len(ph_durs_sec):
            ph_frames.append(max(1, int(round(ph_durs_sec[i] * FRAME_RATE))))
        else:
            ph_frames.append(1)

    groups = []
    i = 0
    n_ph = len(phonemes)
    while i < n_ph:
        j = i + 1
        while (j < n_ph and note_names[j] == note_names[i]
               and abs(note_durs_sec[j] - note_durs_sec[i]) < 1e-9):
            j += 1
        groups.append((i, j, note_durs_sec[i]))
        i = j

    for (s, e, note_dur) in groups:
        note_len_frames = max(1, int(round(note_dur * FRAME_RATE)))
        # lag：辅音前置量（除末音素外所占帧数）
        lag_frames = sum(ph_frames[s:e - 1]) if (e - s) > 1 else 0
        lag_frames = min(lag_frames, max(0, note_len_frames - 1))
        note_f0 = note_to_f0(note_names[s]) if note_names[s].lower() != 'rest' else 0.0
        frame_phonemes.extend(phonemes[s:e])
        frame_durations.extend(ph_frames[s:e])
        frame_f0.extend([note_f0] * note_len_frames)
        frame_lags.extend([float(lag_frames)] * note_len_frames)
        note_lengths.append(note_len_frames)

    # 对齐音频与 F0 长度
    max_frames = audio.shape[-1] // HOP_LENGTH + 1
    if len(f0_curve) > 0:
        max_frames = min(max_frames, len(f0_curve))
    frame_f0 = np.array(frame_f0[:max_frames], dtype=np.float32)
    frame_lags = np.array(frame_lags[:max_frames], dtype=np.float32)
    if len(frame_f0) < max_frames:
        frame_f0 = np.pad(frame_f0, (0, max_frames - len(frame_f0)))
    if len(frame_lags) < max_frames:
        frame_lags = np.pad(frame_lags, (0, max_frames - len(frame_lags)))

    # F0 用 CREPE
    return (frame_phonemes, frame_durations,
            f0_curve[:max_frames] if len(f0_curve) > 0 else frame_f0,
            frame_lags, np.array(note_lengths, dtype=np.int32))

def _note_lengths_in_range(note_lengths, seg_start, seg_end):
    """修复 #6: 取与片段帧区间 [seg_start, seg_end) 相交的音符, 返回片段内实际帧数 原实现只用前 5 个音符按比例缩放, 使 note_lengths 与片段严重错位, 而它正是 F0-step 模拟与逐音符时长归一化的依据 → 训练信号错配"""
    out = []
    pos = 0
    for nl in note_lengths:
        nl = int(nl)
        if nl <= 0:
            continue
        n_start, n_end = pos, pos + nl
        pos = n_end
        ov = min(n_end, seg_end) - max(n_start, seg_start)
        if ov > 0:
            out.append(max(1, int(ov)))
        if n_start >= seg_end:
            break
    if not out:
        out = [max(1, seg_end - seg_start)]
    return out

def merge_to_4s_segments(audio, f0_curve, phonemes, durations, lags, note_lengths):
    """论文 §4"""
    segments = []

    idx = 0
    while idx < len(phonemes):
        accum = 0
        start_idx = idx
        while idx < len(phonemes) and accum + durations[idx] <= TARGET_FRAMES * 1.3:
            accum += durations[idx]
            idx += 1
            if accum >= TARGET_FRAMES * 0.7:
                break
        if idx == start_idx:
            idx = start_idx + 1
            continue
        if accum < HOP_LENGTH // 8 * 10:  # 太小, 跳过
            continue

        end_idx = idx
        seg_dur_frames = sum(durations[start_idx:end_idx])
        seg_dur_samples = seg_dur_frames * HOP_LENGTH

        # 音频切片
        audio_start = sum(durations[:start_idx]) * HOP_LENGTH
        audio_end = audio_start + seg_dur_samples
        audio_start = min(audio_start, len(audio))
        audio_end = min(audio_end, len(audio))
        seg_audio = audio[audio_start:audio_end]

        # F0 切片
        f0_start = sum(durations[:start_idx])
        f0_end = f0_start + seg_dur_frames
        f0_start = min(f0_start, len(f0_curve))
        f0_end = min(f0_end, len(f0_curve))
        seg_f0 = f0_curve[f0_start:f0_end]

        # Lags 切片
        lag_start = sum(durations[:start_idx])
        lag_end = lag_start + seg_dur_frames
        lag_start = min(lag_start, len(lags))
        lag_end = min(lag_end, len(lags))
        seg_lags = lags[lag_start:lag_end]

        # note_lengths：与片段相交的真实音符时长
        seg_note_lengths = _note_lengths_in_range(note_lengths, f0_start, f0_end)

        segments.append({
            'audio': seg_audio,
            'phoneme_ids': phonemes[start_idx:end_idx],
            'durations': durations[start_idx:end_idx],
            'f0_curve': np.array(seg_f0, dtype=np.float32),
            'lags': np.array(seg_lags, dtype=np.float32),
            'note_lengths': np.array(seg_note_lengths, dtype=np.int32),
        })

    if len(segments) == 0:
        # 整段作为一段
        segments.append({
            'audio': audio,
            'phoneme_ids': phonemes,
            'durations': durations,
            'f0_curve': np.array(f0_curve, dtype=np.float32),
            'lags': np.array(lags, dtype=np.float32),
            'note_lengths': np.array(note_lengths, dtype=np.int32),
        })

    return segments

def build_phoneme_map(all_phonemes):
    """构建音素到 ID 的映射表"""
    unique = sorted(set(all_phonemes))
    return {p: i for i, p in enumerate(unique)}, {i: p for i, p in enumerate(unique)}

def process_segments_file(trans_file, wav_dir, output_dir, split_name, all_phonemes_collector):
    """处理一个转录文件 (train"""
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0

    with open(trans_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  [{split_name}] 共 {len(lines)} 段")

    for line in lines:
        if not line.strip():
            continue
        parsed = parse_transcription(line)
        if parsed is None:
            continue
        seg_id, phonemes, note_names, note_durs, ph_durs = parsed

        wav_path = os.path.join(wav_dir, f"{seg_id}.wav")
        if not os.path.exists(wav_path):
            print(f"    WAV 缺失: {wav_path}")
            continue

        audio = load_audio(wav_path)

        # 跳过过短片段
        if len(audio) < HOP_LENGTH * 10:
            print(f"    跳过过短段: {seg_id} ({len(audio)} samples)")
            continue

        # F0
        f0_curve = extract_crepe_f0(audio)

        # 帧级特征
        frame_phones, durations, f0, lags, note_lengths = compute_frame_features(
            parsed, f0_curve, audio
        )

        # 裁剪对齐
        effective_samples = min(len(audio), len(f0) * HOP_LENGTH)
        audio = audio[:effective_samples]

        # 音素
        all_phonemes_collector.extend(frame_phones)

        # 合并为 ~4s 片段
        segments = merge_to_4s_segments(audio, f0, frame_phones, durations, lags, note_lengths)

        for i, seg in enumerate(segments):
            if len(seg['audio']) < HOP_LENGTH * 10:
                continue
            out_path = os.path.join(output_dir, f"{seg_id}_{i:04d}.npz")
            np.savez(out_path,
                     audio=seg['audio'].astype(np.float32),
                     phonemes=np.array(seg['phoneme_ids'], dtype=object),
                     durations=np.array(seg['durations'], dtype=np.int32),
                     f0=np.array(seg['f0_curve'], dtype=np.float32),
                     lags=np.array(seg['lags'], dtype=np.float32),
                     note_lengths=np.array(seg['note_lengths'], dtype=np.int32),
                     song_id=seg_id[:4])
            saved_count += 1

    return saved_count

def main():
    parser = argparse.ArgumentParser(description='OpenCPOP 预处理 V2 (标准格式)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='OpenCPOP 数据集根目录 (含 segments/)')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed',
                        help='预处理输出目录')
    parser.add_argument('--test_songs', type=int, default=3,
                        help='测试集歌曲数 (论文: 3)')
    parser.add_argument('--val_songs', type=int, default=3,
                        help='验证集歌曲数 (论文: 3)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    wav_dir = data_dir / 'segments' / 'wavs'
    segments_dir = data_dir / 'segments'

    # 转录数据
    trans_path = segments_dir / 'transcriptions.txt'
    train_path = segments_dir / 'train.txt'
    test_path = segments_dir / 'test.txt'

    # 音素表
    all_phonemes = []

    # test.txt
    if test_path.exists():
        print(f"处理测试集: {test_path}")
        test_count = process_segments_file(
            str(test_path), str(wav_dir),
            str(Path(args.output_dir) / 'test'), 'test', all_phonemes
        )
        print(f"  测试集: {test_count} 段")
    else:
        print("test.txt 未找到, 跳过测试集")
        test_count = 0

    # train.txt（含验证集划分）
    if train_path.exists():
        print(f"处理训练集: {train_path}")
        with open(train_path, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()

        song_groups = defaultdict(list)
        for line in train_lines:
            if not line.strip():
                continue
            parts = line.strip().split('|')
            if len(parts) < 6:
                continue
            song_id = parts[0][:4]
            song_groups[song_id].append(line)

        song_ids = sorted(song_groups.keys())
        val_ids = set(song_ids[:args.val_songs])
        train_ids = set(song_ids[args.val_songs:])

        print(f"  歌曲总数: {len(song_ids)}, 训练: {len(train_ids)}, 验证: {len(val_ids)}")

        train_tmp = Path(args.output_dir) / '_train_tmp.txt'
        val_tmp = Path(args.output_dir) / '_val_tmp.txt'
        os.makedirs(args.output_dir, exist_ok=True)

        with open(train_tmp, 'w', encoding='utf-8') as f_tr:
            for sid in sorted(train_ids):
                for line in song_groups[sid]:
                    f_tr.write(line)

        with open(val_tmp, 'w', encoding='utf-8') as f_val:
            for sid in sorted(val_ids):
                for line in song_groups[sid]:
                    f_val.write(line)

        print("  处理训练集...")
        train_count = process_segments_file(
            str(train_tmp), str(wav_dir),
            str(Path(args.output_dir) / 'train'), 'train', all_phonemes
        )

        print("  处理验证集...")
        val_count = process_segments_file(
            str(val_tmp), str(wav_dir),
            str(Path(args.output_dir) / 'val'), 'val', all_phonemes
        )

        train_tmp.unlink(missing_ok=True)
        val_tmp.unlink(missing_ok=True)

        print(f"  训练集: {train_count} 段, 验证集: {val_count} 段")
    else:
        print("train.txt 未找到")
        train_count, val_count = 0, 0

    # 音素映射
    p2i, i2p = build_phoneme_map(all_phonemes)
    map_path = Path(args.output_dir) / 'phoneme_map.json'
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump({'p2i': p2i, 'i2p': i2p, 'num_phonemes': len(p2i)}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n预处理完成:")
    print(f"  训练: {train_count} 段, 验证: {val_count} 段, 测试: {test_count} 段")
    print(f"  音素数: {len(p2i)}, 音素表: {map_path}")

if __name__ == '__main__':
    main()
