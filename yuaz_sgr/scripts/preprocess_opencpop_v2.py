# YUAZ SGR
"""
scripts/preprocess_opencpop_v2.py
OpenCPOP 标准格式预处理 — 论文 §4.1 流水线

论文要求:
  - 24kHz 降采样
  - -26dB 归一化
  - 4秒分段, 词级边界对齐
  - CREPE F0 提取
  - 数据集划分: 前3首=test, 随后3首=val, 其余=train

输入: opencpop 目录 (含 segments/wavs/ + segments/transcriptions.txt)
输出: data/preprocessed/{train,val,test}/*.npz
"""
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


def extract_crepe_f0(audio):
    """用 CREPE 从音频提取真实 F0 曲线 (4秒分片防 OOM)"""
    audio_t = torch.FloatTensor(audio).unsqueeze(0)
    n_frames_full = len(audio) // HOP_LENGTH + 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f0_full = torch.zeros(n_frames_full, device=device)

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
                                          model='full', device=device, pad=True)
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


def parse_transcription(line):
    """
    解析转录行
    格式: id|lyrics|phonemes|notes|note_durations|phoneme_durations|slur_flags
    返回: (seg_id, phonemes, note_names, note_durs_sec, phoneme_durs_sec)
    """
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
    # 处理如 'G#4/Ab4' 格式 — 取第一个
    name = note_name.split('/')[0]
    # 解析: 音名 + 可选升降号 + 八度
    # 例如: 'C4', 'G#4', 'A#4', 'F#4'
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
    """
    从转录数据计算帧级特征
    返回: frame_phonemes, frame_durations, frame_f0, frame_lags, note_lengths
    """
    seg_id, phonemes, note_names, note_durs_sec, ph_durs_sec = transcription
    audio_dur_sec = len(audio) / SAMPLE_RATE
    frame_phonemes = []
    frame_durations = []
    frame_f0 = []       # MIDI note F0
    frame_lags = []
    note_lengths = []

    pos_sec = 0.0
    for note_name, note_dur in zip(note_names, note_durs_sec):
        note_len_frames = max(1, int(note_dur * FRAME_RATE))
        # 找出此音符内的音素
        note_phones = []
        note_ph_start = pos_sec
        note_ph_end = pos_sec + note_dur
        pos_sec = note_ph_end

        # lag: 辅音前置导致的偏移 (论文 §3.3.1)
        # 简化: 将 note_dur 减去 note_name!='rest' 时的 lag
        # 实际上 lag 已在 phoneme_durs 中隐式编码
        # 我们用 note_len_frames 直接作为有效时长
        lag_frames = 0  # 预处理阶段 lag=0, 训练时由 temporal predictor 学习

        # 查找此 note 范围内的音素
        # 音素时长和 = note_dur
        # 按顺序分配音素
        # 简化: 直接使用所有音素 + 对应的 ph_durs

        note_f0 = note_to_f0(note_name) if note_name.lower() != 'rest' else 0.0
        frame_f0.extend([note_f0] * note_len_frames)
        frame_lags.extend([lag_frames] * note_len_frames)
        note_lengths.append(note_len_frames)

    # 音素列表: 直接使用 transcription 中的音素, 时长转为帧
    for i, ph in enumerate(phonemes):
        if i < len(ph_durs_sec):
            dur_frames = max(1, int(ph_durs_sec[i] * FRAME_RATE))
        else:
            dur_frames = 1
        frame_phonemes.append(ph)
        frame_durations.append(dur_frames)

    # 确保 frame_f0 长度不超音频
    max_frames = min(len(f0_curve), audio.shape[-1] // HOP_LENGTH + 1)
    total_note_frames = sum(note_lengths)
    frame_f0 = np.array(frame_f0[:max_frames], dtype=np.float32)
    frame_lags = np.array(frame_lags[:max_frames], dtype=np.float32)
    if len(frame_f0) < max_frames:
        frame_f0 = np.pad(frame_f0, (0, max_frames - len(frame_f0)))
        frame_lags = np.pad(frame_lags, (0, max_frames - len(frame_lags)))

    # F0 curve: 用 CREPE 替代 MIDI F0
    return (frame_phonemes, frame_durations,
            f0_curve[:max_frames] if len(f0_curve) > 0 else frame_f0,
            frame_lags, np.array(note_lengths, dtype=np.int32))


def merge_to_4s_segments(audio, f0_curve, phonemes, durations, lags, note_lengths):
    """
    论文 §4.1: 将音素序列合并为 ~4秒片段, 词级边界对齐
    策略: 累积音素帧数直到接近 TARGET_FRAMES, 在最接近的音素边界切割
    """
    total_frames = sum(durations)
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

        # Note lengths: 在 seg 范围内的 note
        # 简化: 用原始 note_lengths 按比例
        seg_note_lengths = [max(1, int(nl * seg_dur_frames / max(1, total_frames)))
                           for nl in note_lengths[:5]] if len(note_lengths) > 0 else [seg_dur_frames]

        segments.append({
            'audio': seg_audio,
            'phoneme_ids': phonemes[start_idx:end_idx],
            'durations': durations[start_idx:end_idx],
            'f0_curve': np.array(seg_f0, dtype=np.float32),
            'lags': np.array(seg_lags, dtype=np.float32),
            'note_lengths': np.array(seg_note_lengths, dtype=np.int32),
        })

    if len(segments) == 0:
        # 退路: 整个区域作为一个段
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
    """
    处理一个转录文件 (train.txt 或 test.txt) 中的所有段
    """
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

        # 加载 WAV
        wav_path = os.path.join(wav_dir, f"{seg_id}.wav")
        if not os.path.exists(wav_path):
            print(f"    WAV 缺失: {wav_path}")
            continue

        audio = load_audio(wav_path)

        # 跳过过短的
        if len(audio) < HOP_LENGTH * 10:
            print(f"    跳过过短段: {seg_id} ({len(audio)} samples)")
            continue

        # CREPE F0
        f0_curve = extract_crepe_f0(audio)

        # 帧级特征
        frame_phones, durations, f0, lags, note_lengths = compute_frame_features(
            parsed, f0_curve, audio
        )

        # 音频裁剪到与 F0 对齐
        effective_samples = min(len(audio), len(f0) * HOP_LENGTH)
        audio = audio[:effective_samples]

        # 收集音素
        all_phonemes_collector.extend(frame_phones)

        # 合并为 ~4s 片段
        segments = merge_to_4s_segments(audio, f0, frame_phones, durations, lags, note_lengths)

        # 保存
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

    # 加载转录数据
    trans_path = segments_dir / 'transcriptions.txt'
    train_path = segments_dir / 'train.txt'
    test_path = segments_dir / 'test.txt'

    # 收集所有音素
    all_phonemes = []

    # 先处理 test.txt (测试集)
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

    # 处理 train.txt (训练集 + 验证集划分)
    if train_path.exists():
        print(f"处理训练集: {train_path}")
        # 先解析所有训练段
        with open(train_path, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()

        # 按歌曲 ID 分组
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

        # 写入临时文件
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

        # 处理训练集
        print("  处理训练集...")
        train_count = process_segments_file(
            str(train_tmp), str(wav_dir),
            str(Path(args.output_dir) / 'train'), 'train', all_phonemes
        )

        # 处理验证集
        print("  处理验证集...")
        val_count = process_segments_file(
            str(val_tmp), str(wav_dir),
            str(Path(args.output_dir) / 'val'), 'val', all_phonemes
        )

        # 清理临时文件
        train_tmp.unlink(missing_ok=True)
        val_tmp.unlink(missing_ok=True)

        print(f"  训练集: {train_count} 段, 验证集: {val_count} 段")
    else:
        print("train.txt 未找到")
        train_count, val_count = 0, 0

    # 构建音素映射
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
