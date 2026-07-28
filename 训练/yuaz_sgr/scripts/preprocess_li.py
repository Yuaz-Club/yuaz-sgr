"""
preprocess_li.py — 李老师数据集预处理（WAV + TextGrid，无 MIDI）

适用场景: 已切片的 WAV + TextGrid 对，TextGrid 含 words 和 phones 两层
- phones tier: 音素边界 → 用于提取音素序列和时长
- words tier: 词边界 → 替代 MIDI note 用于计算 note_lengths 和 time-lag
- F0: 从音频用 CREPE 提取

用法:
    python preprocess_li.py --data_dir /path/to/李老师 --output_dir data/preprocessed
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import librosa
import textgrid
import torch
import torchcrepe

SAMPLE_RATE = 24000
HOP_LENGTH = 256
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH
NORM_DB = -26
SEGMENT_SEC = 7  # 匹配数据集实际片段时长（~6.4秒）


def extract_crepe_f0(audio, device='cuda'):
    """用 CREPE 从音频提取帧级真实 F0"""
    audio_t = torch.FloatTensor(audio).unsqueeze(0).to(device)
    n_frames_full = len(audio) // HOP_LENGTH + 1
    f0_full = torch.zeros(n_frames_full, device=device)

    chunk_sec = 4
    chunk_samples = chunk_sec * SAMPLE_RATE
    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, len(audio))
            chunk = audio_t[:, start:end]
            f0_chunk = torchcrepe.predict(
                chunk, SAMPLE_RATE, hop_length=HOP_LENGTH,
                fmin=50.0, fmax=2006.0, model='full',
                device=device, pad=True,
            )
            f0_chunk = f0_chunk[0]
            c_start = start // HOP_LENGTH
            c_end = c_start + f0_chunk.shape[0]
            if c_end > n_frames_full:
                c_end = n_frames_full
                f0_chunk = f0_chunk[:c_end - c_start]
            f0_full[c_start:c_end] = f0_chunk

    return f0_full.cpu().numpy().astype(np.float32)


def load_audio(wav_path):
    """加载音频，降采样到 24kHz，-26dBFS 归一化"""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    rms = np.sqrt(np.mean(audio ** 2))
    audio *= (10 ** (NORM_DB / 20)) / (rms + 1e-8)
    return audio


def process_segment(wav_path, tg_path, device='cuda'):
    """处理单个 WAV + TextGrid 对 -> 返回 npz 数据字典"""
    # 1. 加载音频 + 提取 F0
    audio = load_audio(wav_path)
    f0_curve = extract_crepe_f0(audio, device=device)

    # 2. 解析 TextGrid
    tg = textgrid.TextGrid.fromFile(tg_path)
    phones_tier = tg.getFirst('phones')
    words_tier = tg.getFirst('words')

    # 提取有效音素（跳过 SP/AP 静音标记）
    phones = []
    phone_intervals = []
    for iv in phones_tier:
        mark = iv.mark.strip()
        if mark and mark not in ('SP', 'AP', ''):
            phones.append(mark)
            phone_intervals.append((iv.minTime, iv.maxTime))

    # 提取有效词（跳过 SP/AP）
    words = []
    word_intervals = []
    for iv in words_tier:
        mark = iv.mark.strip()
        if mark and mark not in ('SP', 'AP', ''):
            words.append(mark)
            word_intervals.append((iv.minTime, iv.maxTime))

    if not phones or not words:
        return None

    # 3. 将音素按词边界分配
    #    计算: phoneme_ids, durations(帧), f0(帧), lags(帧), note_lengths(帧)
    phone_idx = 0
    frame_phonemes = []
    frame_durations = []
    frame_lags = []
    note_lengths = []

    for w_idx, (w_start, w_end) in enumerate(word_intervals):
        note_dur_sec = w_end - w_start
        note_len_frames = max(1, int(note_dur_sec * FRAME_RATE))

        # 收集这个词涵盖的音素
        word_phones = []
        word_ph_intervals = []
        while phone_idx < len(phones):
            ph_start, ph_end = phone_intervals[phone_idx]
            if ph_end <= w_start:
                phone_idx += 1
                continue
            if ph_start >= w_end:
                break
            word_phones.append(phones[phone_idx])
            word_ph_intervals.append(phone_intervals[phone_idx])
            phone_idx += 1

        if not word_phones:
            continue

        # time-lag: 第一个音素的起始相对词边界
        first_ph_start = word_ph_intervals[0][0]
        lag = first_ph_start - w_start
        lag_frames = int(lag * FRAME_RATE)
        l_hat_frames = max(1, note_len_frames - lag_frames)

        # 按音素时间比例分配帧
        ph_secs = [pe - ps for ps, pe in word_ph_intervals]
        total_ph_sec = sum(ph_secs)

        for j, ph in enumerate(word_phones):
            frame_phonemes.append(ph)
            if j == len(word_phones) - 1:
                remaining = l_hat_frames - sum(frame_durations)
                frame_durations.append(max(1, remaining))
            else:
                frac = ph_secs[j] / max(total_ph_sec, 1e-8)
                dur_frames = max(1, int(frac * l_hat_frames))
                frame_durations.append(dur_frames)

        frame_lags.extend([lag_frames] * l_hat_frames)
        note_lengths.append(note_len_frames)

    if not frame_phonemes:
        return None

    total_frames = sum(frame_durations)
    total_f0_frames = len(f0_curve)

    # 截取对应帧的 F0
    seg_f0 = f0_curve[:min(total_frames, total_f0_frames)]
    if len(seg_f0) < total_frames:
        seg_f0 = np.pad(seg_f0, (0, total_frames - len(seg_f0)))

    seg_lags = np.array(frame_lags[:total_frames], dtype=np.float32)
    if len(seg_lags) < total_frames:
        seg_lags = np.pad(seg_lags, (0, total_frames - len(seg_lags)))

    # 截取音频
    target_samples = total_frames * HOP_LENGTH
    seg_audio = audio[:min(target_samples, len(audio))]
    if len(seg_audio) < target_samples:
        seg_audio = np.pad(seg_audio, (0, target_samples - len(seg_audio)))

    return {
        'audio': seg_audio.astype(np.float32),
        'phonemes': np.array(frame_phonemes, dtype=object),
        'durations': np.array(frame_durations, dtype=np.int32),
        'f0': seg_f0.astype(np.float32),
        'lags': seg_lags.astype(np.float32),
        'note_lengths': np.array(note_lengths, dtype=np.int32),
    }


def build_merged_phoneme_map(base_map_path, li_phonemes):
    """用基座音素映射为基础，合并李老师数据集的新音素"""
    with open(base_map_path, 'r', encoding='utf-8') as f:
        base = json.load(f)
    base_p2i = base['p2i']
    base_num = base['num_phonemes']

    # 检查哪些 li 音素不在基座中
    li_phonemes_set = set(li_phonemes)
    base_phonemes_set = set(base_p2i.keys())

    new_phonemes = sorted(li_phonemes_set - base_phonemes_set)
    missing_in_li = sorted(base_phonemes_set - li_phonemes_set)

    print(f'  基座音素数: {base_num}')
    print(f'  Li 独有音素: {new_phonemes}')
    print(f'  基座独有(li未出现): {missing_in_li}')

    # 合并映射：基座音素保持原索引，新音素追加
    merged_p2i = dict(base_p2i)
    for p in new_phonemes:
        merged_p2i[p] = len(merged_p2i)

    merged_i2p = {i: p for p, i in merged_p2i.items()}
    return merged_p2i, merged_i2p, len(merged_p2i)


def main():
    parser = argparse.ArgumentParser(description='李老师数据集预处理')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录 (含 切片/ 和 TG/)')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed', help='输出目录')
    parser.add_argument('--val_songs', type=int, default=1, help='验证集歌曲数')
    parser.add_argument('--test_songs', type=int, default=1, help='测试集歌曲数')
    parser.add_argument('--device', type=str, default='cuda', help='CREPE 设备')
    parser.add_argument('--base_phoneme_map', type=str, default=None,
                        help='基座模型音素映射表路径，不指定则独立构建')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    wav_dir = data_dir / '切片'
    tg_dir = data_dir / 'TG'
    output_dir = Path(args.output_dir)

    # 匹配 WAV ↔ TextGrid
    wav_files = sorted(wav_dir.glob('*.wav'))
    pairs = []
    for wav_path in wav_files:
        stem = wav_path.stem  # e.g. "DIyitian_wav_000"
        tg_path = tg_dir / f'{stem}.TextGrid'
        if not tg_path.exists():
            print(f'  警告: 找不到 TextGrid for {wav_path.name}, 跳过')
            continue
        pairs.append((wav_path, tg_path))

    print(f'匹配到 {len(pairs)} 个 WAV+TG 对')

    # 提取所有歌曲名（用于划分）
    song_ids = sorted(set(p.stem.rsplit('_wav_', 1)[0] for p, _ in pairs))
    print(f'歌曲: {song_ids}')

    # 划分
    test_ids = set(song_ids[:args.test_songs])
    val_ids = set(song_ids[args.test_songs:args.test_songs + args.val_songs])
    print(f'测试: {sorted(test_ids)}, 验证: {sorted(val_ids)}, 训练: 其余')

    all_phonemes = []
    stats = {'train': 0, 'val': 0, 'test': 0}

    for wav_path, tg_path in pairs:
        song_id = wav_path.stem.rsplit('_wav_', 1)[0]
        if song_id in test_ids:
            split = 'test'
        elif song_id in val_ids:
            split = 'val'
        else:
            split = 'train'

        out_subdir = output_dir / split
        out_subdir.mkdir(parents=True, exist_ok=True)

        try:
            data = process_segment(str(wav_path), str(tg_path), device=args.device)
            if data is None:
                continue

            out_name = wav_path.stem
            out_path = out_subdir / f'{out_name}.npz'
            np.savez(out_path, **data, song_id=song_id)
            stats[split] += 1
            all_phonemes.extend(data['phonemes'].tolist())
            print(f'  [{split}] {out_name}: {len(data["phonemes"])} 音素, {data["audio"].shape[0]/24000:.1f}s')
        except Exception as e:
            print(f'  [ERROR] {wav_path.name}: {e}')
            import traceback
            traceback.print_exc()

    # 构建音素映射表
    if args.base_phoneme_map:
        p2i, i2p, num_ph = build_merged_phoneme_map(args.base_phoneme_map, all_phonemes)
        source = 'merged_with_base'
    else:
        unique_phonemes = sorted(set(all_phonemes))
        p2i = {p: i for i, p in enumerate(unique_phonemes)}
        i2p = {i: p for i, p in enumerate(unique_phonemes)}
        num_ph = len(p2i)
        source = 'li_dataset'

    map_path = output_dir / 'phoneme_map.json'
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump({
            'p2i': p2i, 'i2p': i2p,
            'num_phonemes': num_ph,
            'source': source,
        }, f, ensure_ascii=False, indent=2)

    print(f'\n预处理完成:')
    print(f'  音素数: {num_ph}')
    print(f'  训练: {stats["train"]} 段, 验证: {stats["val"]} 段, 测试: {stats["test"]} 段')
    print(f'  音素表: {map_path}')


if __name__ == '__main__':
    main()
