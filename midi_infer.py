"""
MIDI → SVS 推理脚本
解析 MIDI 文件，提取 F0 和歌词，通过模型合成歌声
"""
import os, sys, json, argparse, warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.svs_model import SVSModel


def load_config(config_dir):
    import yaml
    config = {}
    for name in ['base', 'model', 'train']:
        path = os.path.join(config_dir, f'{name}.yaml')
        with open(path, 'r') as f:
            config.update(yaml.safe_load(f))
    return config


def midi_to_f0_phonemes(midi_path, phoneme_map, sample_rate=24000, hop_length=256):
    """
    解析 MIDI → F0 曲线 + 音素 ID + 音素时长
    返回: (f0, phoneme_ids, phoneme_durations)
    """
    import mido
    mid = mido.MidiFile(midi_path)

    # 解析 tempo
    tempo = 500000  # 默认 120 BPM
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo

    ticks_per_beat = mid.ticks_per_beat
    sec_per_tick = tempo / 1e6 / ticks_per_beat

    # 提取节拍信息
    beats_per_bar = 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                beats_per_bar = msg.numerator

    # 提取所有音符和歌词
    notes = []  # (pitch, start_tick, end_tick)
    all_lyrics_tick = []  # (tick, lyric_text)

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append([msg.note, abs_tick, abs_tick + 1000])
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for i, (n, s, e) in enumerate(notes):
                    if n == msg.note and e == s + 1000:
                        notes[i][2] = abs_tick
                        break
            elif msg.type == 'lyrics':
                all_lyrics_tick.append((abs_tick, msg.text))
            elif msg.type == 'text':
                all_lyrics_tick.append((abs_tick, msg.text))
            elif msg.type == 'cue_marker':
                all_lyrics_tick.append((abs_tick, msg.text))

    # 排序
    notes.sort(key=lambda x: x[1])
    all_lyrics_tick.sort(key=lambda x: x[1])

    # 将 "+" 替换为 "a" (啊)
    lyrics_map = {}
    for tick, text in all_lyrics_tick:
        lrc = text.strip().lower()
        if lrc == '+':
            lrc = 'a'
        lyrics_map[tick] = lrc

    # 为每个音符分配歌词（根据最近的前一个歌词事件）
    note_lyrics = []
    lyric_items = sorted(lyrics_map.items())
    for note_pitch, note_start, note_end in notes:
        assigned_lyric = 'a'
        for lrc_tick, lrc_text in lyric_items:
            if lrc_tick <= note_start + 5:
                assigned_lyric = lrc_text
            else:
                break
        note_lyrics.append(assigned_lyric)

    # 总时长
    total_ticks = max(n[2] for n in notes) + 240
    total_sec = total_ticks * sec_per_tick + 0.5
    n_frames = int(total_sec * sample_rate / hop_length) + 1

    f0 = np.zeros(n_frames, dtype=np.float32)
    f0_silence = np.ones(n_frames, dtype=bool)
    f0_silence[:] = True

    # 构建 F0 曲线 + 记录音符边界用于 temporal predictor
    note_boundaries = []  # (start_frame, end_frame)
    for note_pitch, note_start, note_end in notes:
        start_sec = note_start * sec_per_tick
        end_sec = note_end * sec_per_tick
        start_frame = int(start_sec * sample_rate / hop_length)
        end_frame = int(end_sec * sample_rate / hop_length)
        end_frame = min(end_frame, n_frames)
        note_boundaries.append((start_frame, end_frame))
        hz = 440.0 * 2.0 ** ((note_pitch - 69) / 12.0)
        f0[start_frame:end_frame] = hz
        f0_silence[start_frame:end_frame] = False

    # F0 边界插值（消除方块凸起）
    for i in range(1, n_frames - 1):
        if f0[i] == 0 and f0[i-1] > 0 and f0[i+1] > 0:
            f0[i] = (f0[i-1] + f0[i+1]) / 2.0

    # 添加自然颤音 + 微抖动（匹配 pyworld 训练分布）
    voiced_mask = f0 > 0
    if voiced_mask.any():
        # 颤音: 5-7 Hz 正弦调制, ±15 cents (~0.87% 半音)
        t = np.arange(n_frames) * hop_length / sample_rate
        vibrato_rate = 6.0  # Hz
        vibrato_depth = 0.0087 * 1.5  # cents → 相对比 (15 cents)
        vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        # 微自然抖动 (random walk, ±2 Hz)
        jitter = np.cumsum(np.random.normal(0, 0.3, size=n_frames))
        jitter = np.clip(jitter, -3.0, 3.0)
        f0[voiced_mask] = f0[voiced_mask] * vibrato[voiced_mask] + jitter[voiced_mask]
        f0[voiced_mask] = np.maximum(f0[voiced_mask], 50.0)

    # 构建音素序列
    # 简单映射：中文拼音 → 模型音素
    # 使用 p2i 查找，如果找不到则使用相近的音素
    p2i = phoneme_map['p2i']
    i2p = phoneme_map['i2p']

    # 拼音 → 音素映射
    PINYIN_TO_PHONEMES = {
        'yue': ['y', 've'],
        'sui': ['s', 'ui'],
        'zai': ['z', 'ai'],
        'le': ['l', 'e'],
        'mei': ['m', 'ei'],
        'jian': ['j', 'ian'],
        'meng': ['m', 'eng'],
        'ye': ['y', 'e'],
        'zui': ['z', 'ui'],
        'qu': ['q', 'v'],
        'zun': ['z', 'un'],
        'qian': ['q', 'ian'],
        'xiang': ['x', 'iang'],
        'ming': ['m', 'ing'],
        'he': ['h', 'e'],
        'zhong': ['zh', 'ong'],
        'kan': ['k', 'an'],
        'yi': ['y', 'i'],
        'yan': ['y', 'an'],
        'dao': ['d', 'ao'],
        'mi': ['m', 'i'],
        'li': ['l', 'i'],
        'ren': ['r', 'en'],
        'a': ['a'],  # 啊
    }

    # 按音符构建 phoneme_ids 和 durations
    all_phoneme_ids = []
    all_phoneme_durations = []

    has_prev_silence = False

    for note_idx in range(len(notes)):
        note_pitch, note_start, note_end = notes[note_idx]
        start_sec = note_start * sec_per_tick
        end_sec = note_end * sec_per_tick
        start_frame = int(start_sec * sample_rate / hop_length)
        end_frame = int(end_sec * sample_rate / hop_length)

        # 如果前一个音符结束和当前之间有空隙，加 SP
        if note_idx > 0:
            prev_end_sec = notes[note_idx - 1][2] * sec_per_tick
            prev_end_frame = int(prev_end_sec * sample_rate / hop_length)
            gap = start_frame - prev_end_frame
            if gap > 2:
                all_phoneme_ids.append(p2i.get('SP', 1))
                all_phoneme_durations.append(gap)
                has_prev_silence = True

        note_frames = max(1, end_frame - start_frame)

        # 歌词
        lyric = note_lyrics[note_idx]

        # 查找对应音素
        phonemes = PINYIN_TO_PHONEMES.get(lyric, ['a'])

        if note_frames < len(phonemes):
            phonemes = phonemes[:note_frames]

        dur_per_ph = note_frames // len(phonemes)
        remainder = note_frames % len(phonemes)

        for j, ph in enumerate(phonemes):
            ph_id = p2i.get(ph, p2i.get('a', 2))
            dur = dur_per_ph + (1 if j < remainder else 0)
            all_phoneme_ids.append(ph_id)
            all_phoneme_durations.append(dur)

    # 尾部静音
    last_frame = int(notes[-1][2] * sec_per_tick * sample_rate / hop_length)
    tail_silence = n_frames - last_frame
    if tail_silence > 3:
        all_phoneme_ids.append(p2i.get('SP', 1))
        all_phoneme_durations.append(tail_silence)

    return f0, all_phoneme_ids, all_phoneme_durations, n_frames, note_boundaries


def smooth_f0(f0, kernel_size=7):
    """对 F0 做高斯滑音平滑 (portamento), 消除方块状"""
    import scipy.ndimage as ndimage
    smoothed = f0.copy()
    mask = f0 > 0
    if mask.sum() > 0:
        # 只对有声区域做高斯平滑
        f0_valid = np.where(mask, f0, 0)
        smoothed = ndimage.gaussian_filter1d(f0_valid, sigma=kernel_size / 2.0, mode='constant')
        # 把平滑后无声区域拉回 0
        smoothed[~mask] = 0.0
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config_dir', required=True)
    parser.add_argument('--midi', required=True)
    parser.add_argument('--phoneme_map', required=True)
    parser.add_argument('--output', default='inference_output')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config_dir)
    sr = config.get('sample_rate', 24000)
    hop_length = config.get('hop_length', 256)

    with open(args.phoneme_map, 'r') as f:
        phoneme_map = json.load(f)
    # 确保 config 包含 conditioning 的完整配置
    cond_cfg = config.get('conditioning', {})
    cond_cfg['num_phonemes'] = phoneme_map['num_phonemes']
    config['conditioning'] = cond_cfg

    # 解析 MIDI
    print(f"解析 MIDI: {args.midi}")
    f0, ph_ids, ph_durations, n_frames, note_boundaries = midi_to_f0_phonemes(
        args.midi, phoneme_map, sr, hop_length
    )

    # F0 滑音平滑 (portamento)
    f0 = smooth_f0(f0, kernel_size=5)
    # 构建 note_boundaries 和 note_lengths (过滤零长度)
    note_boundaries_clean = [(s, e) for s, e in note_boundaries if e > s]
    note_lengths = [e - s for s, e in note_boundaries_clean]
    print(f"  F0 帧数: {n_frames}")
    print(f"  音素数: {len(ph_ids)}")
    print(f"  音素: {' '.join([phoneme_map['i2p'].get(str(i), '?') for i in ph_ids[:20]])}...")

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model = SVSModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_sd = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in ckpt['model'].items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_sd, strict=False)
    model.eval()
    print(f"  加载 {loaded} 参数, 跳过 {skipped}")
    print(f"  步数: {ckpt.get('step', '?')}")

    # 合成
    ph_tensor = torch.LongTensor(ph_ids).unsqueeze(0).to(device)
    dur_tensor = torch.LongTensor(ph_durations).unsqueeze(0).to(device)
    f0_tensor = torch.FloatTensor(f0).unsqueeze(0).to(device)

    print(f"\n合成中... (F0 max={f0.max():.1f}Hz, min={f0[f0>0].min() if (f0>0).any() else 0:.1f}Hz, frames={n_frames})")
    print(f"  有声音区域: {(f0>0).sum()} 帧 / {n_frames} 帧")
    with torch.no_grad():
        # GAN 阶段 (0-68000): flow matching 未训练, 不使用
        # 直接用零潜变量 + F0 → DDSP 解码器
        # z 的 shape: (1, 128, T), 全零
        # 配合训练好的 env_net/ap_net/weight_net 输出稳定频谱
        T = f0_tensor.shape[-1]
        z_hat = torch.zeros(1, 128, T, device=device)
        waveform = model.ddsp_decoder(f0_tensor, z_hat)

    wav = waveform.squeeze().cpu().numpy()

    # 保存
    midi_name = Path(args.midi).stem
    out_path = os.path.join(args.output, f'{midi_name}_svs.wav')
    import soundfile as sf
    sf.write(out_path, wav, sr)
    print(f"\n完成! 保存: {out_path}")
    print(f"  时长: {len(wav) / sr:.2f}s")


if __name__ == '__main__':
    main()
