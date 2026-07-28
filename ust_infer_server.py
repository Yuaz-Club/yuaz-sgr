"""
UST → SVS 推理脚本
解析 UST 文件（含歌词+音高+弯音），通过模型合成歌声
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


MIDI_NOTE_A4 = 69
A4_FREQ = 440.0
TICKS_PER_BEAT = 480

PINYIN_TO_PHONEMES = {
    'sheng': ['sh', 'eng'], 'er': ['e', 'r'], 'wei': ['w', 'ei'],
    'ren': ['r', 'en'], 'wu': ['w', 'u'], 'zui': ['z', 'ui'],
    'ni': ['n', 'i'], 'bu': ['b', 'u'], 'xu': ['x', 'v'],
    'yao': ['y', 'ao'], 'bao': ['b', 'ao'], 'qian': ['q', 'ian'],
    'wan': ['w', 'an'], 'dei': ['d', 'ei'], 'ai': ['a', 'i'],
    'e': ['e'], 'ou': ['o', 'u'], 'bi': ['b', 'i'], 'bo': ['b', 'o'],
    'an': ['a', 'n'], 'yu': ['y', 'v'], 'u': ['u'], 'gong': ['g', 'ong'],
    'te': ['t', 'e'], 'mi': ['m', 'i'], 'xuan': ['x', 'van'],
    'hua': ['h', 'ua'], 'ru': ['r', 'u'], 'guo': ['g', 'uo'],
    'bu': ['b', 'u'], 'ting': ['t', 'ing'], 'rang': ['r', 'ang'],
    'wo': ['w', 'o'], 'pei': ['p', 'ei'], 'jing': ['j', 'ing'],
    'ai': ['a', 'i'], 'ken': ['k', 'en'], 'hai': ['h', 'ai'],
    'pu': ['p', 'u'], 'de': ['d', 'e'], 'te': ['t', 'e'],
    'ei': ['e', 'i'], 'ao': ['a', 'o'], 're': ['r', 'e'],
    'li': ['l', 'i'], 'fu': ['f', 'u'], 'rui': ['r', 'ui'],
    'na': ['n', 'a'], 'duo': ['d', 'uo'], 'mei': ['m', 'ei'],
    'gui': ['g', 'ui'], 'you': ['y', 'ou'], 'ji': ['j', 'i'],
    'yi': ['y', 'i'], 'hao': ['h', 'ao'], 'shi': ['sh', 'i'],
    'kai': ['k', 'ai'], 'fan': ['f', 'an'],
    'a': ['a'], 'o': ['o'], 'y': ['y'], 'v': ['v'],
}
# 复合音节映射
COMPOUND_PINYIN = {
    'sheng': ['sh', 'eng'], 'qian': ['q', 'ian'], 'xuan': ['x', 'van'],
    'zhong': ['zh', 'ong'], 'jiang': ['j', 'iang'], 'xiang': ['x', 'iang'],
    'hua': ['h', 'ua'], 'guo': ['g', 'uo'], 'dei': ['d', 'ei'],
}


def parse_ust(ust_path):
    """解析 UST 文件，返回所有音符（含 R）和 tempo"""
    notes = []
    tempo = 120
    current_note = None

    with open(ust_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line.startswith('[#SETTING]'):
                current_note = None
            elif line.startswith('[#'):
                if current_note is not None:
                    notes.append(current_note)
                current_note = {}
            elif line.startswith('Tempo='):
                tempo = float(line.split('=')[1])
            elif current_note is not None and '=' in line:
                k, v = line.split('=', 1)
                current_note[k] = v

    if current_note is not None:
        notes.append(current_note)

    # 转为结构化列表，保留 R
    all_notes = []
    abs_tick = 0
    for note in notes:
        length = int(note.get('Length', 0))
        lyric = note.get('Lyric', 'R')
        note_num = int(note.get('NoteNum', 60))

        all_notes.append({
            'lyric': lyric,
            'note_num': note_num,
            'start_tick': abs_tick,
            'end_tick': abs_tick + length,
            'length': length,
        })
        abs_tick += length

    return all_notes, tempo


def note_to_hz(note_num, cents=0):
    return A4_FREQ * (2 ** ((note_num - 69 + cents / 100) / 12.0))


def build_f0_curve(all_notes, tempo, frame_rate=24000/256):
    """用音符数据生成 F0 曲线。
    R = 静音 (F0=0), 音符 = 固定音高 (F0=NoteNum)
    """
    sec_per_tick = 60.0 / tempo / TICKS_PER_BEAT
    total_sec = max(n['end_tick'] for n in all_notes) * sec_per_tick + 0.5
    n_frames = int(total_sec * frame_rate) + 1
    f0 = np.zeros(n_frames, dtype=np.float32)

    for note in all_notes:
        lyric = note['lyric'].strip().lower()
        if lyric == 'r':  # 休止符 → 静音
            continue

        start_sec = note['start_tick'] * sec_per_tick
        end_sec = note['end_tick'] * sec_per_tick
        s = int(start_sec * frame_rate)
        e = min(int(end_sec * frame_rate), n_frames)
        hz = note_to_hz(note['note_num'], 0)
        f0[s:e] = hz

    # 边界 1 帧平滑（消除方块凸起）
    for i in range(1, n_frames - 1):
        if f0[i] == 0 and f0[i-1] > 0 and f0[i+1] > 0:
            f0[i] = (f0[i-1] + f0[i+1]) / 2.0

    return f0, n_frames


def build_phoneme_sequence(sung_notes, phoneme_map, tempo, frame_rate=24000/256):
    """构建音素 ID 序列和时长"""
    p2i = phoneme_map['p2i']
    sec_per_tick = 60.0 / tempo / TICKS_PER_BEAT

    ph_ids = []
    ph_durations = []

    sorted_notes = sorted(sung_notes, key=lambda x: x['start_tick'])

    # 总帧数
    total_ticks = 0
    for note in sorted_notes:
        end = note['end_tick']
        if end > total_ticks:
            total_ticks = end
    total_sec = total_ticks * sec_per_tick + 0.5
    n_frames_target = int(total_sec * frame_rate) + 1

    for i, note in enumerate(sorted_notes):
        lyric = note['lyric'].strip().lower()

        # 静音
        if lyric == 'r' or lyric == '':
            continue

        # 处理 "+" → "啊"
        if lyric == '+':
            lyric = 'a'

        # 计算帧数
        start_sec = note['start_tick'] * sec_per_tick
        end_sec = note['end_tick'] * sec_per_tick
        note_frames = max(1, int((end_sec - start_sec) * frame_rate))

        # 获取音素分解
        if lyric in COMPOUND_PINYIN:
            phonemes = COMPOUND_PINYIN[lyric]
        elif lyric in PINYIN_TO_PHONEMES:
            phonemes = PINYIN_TO_PHONEMES[lyric]
        else:
            phonemes = [lyric]

        if note_frames < len(phonemes):
            phonemes = phonemes[:note_frames]

        dur_per_ph = note_frames // len(phonemes)
        remainder = note_frames % len(phonemes)

        for j, ph in enumerate(phonemes):
            ph_id = p2i.get(ph, p2i.get('a', 2))
            dur = dur_per_ph + (1 if j < remainder else 0)
            ph_ids.append(ph_id)
            ph_durations.append(dur)

    # 补尾部静音使总时长对齐 F0 帧数
    total_ph_frames = sum(ph_durations)
    if total_ph_frames < n_frames_target:
        ph_ids.append(p2i.get('SP', 1))
        ph_durations.append(n_frames_target - total_ph_frames)

    return ph_ids, ph_durations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config_dir', required=True)
    parser.add_argument('--ust', required=True)
    parser.add_argument('--phoneme_map', required=True)
    parser.add_argument('--output', default='inference_output')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config_dir)
    sr = config.get('sample_rate', 24000)
    hop_length = config.get('hop_length', 256)
    frame_rate = sr / hop_length

    with open(args.phoneme_map, 'r') as f:
        phoneme_map = json.load(f)
    cond_cfg = config.get('conditioning', {})
    cond_cfg['num_phonemes'] = phoneme_map['num_phonemes']
    config['conditioning'] = cond_cfg

    # 解析 UST
    print(f"解析 UST: {args.ust}")
    all_notes, tempo = parse_ust(args.ust)
    sung_notes = [n for n in all_notes if n['lyric'].strip().lower() != 'r']
    print(f"  总音符: {len(all_notes)}, 演唱: {len(sung_notes)}")
    print(f"  Tempo: {tempo} BPM")

    f0, n_frames = build_f0_curve(all_notes, tempo, frame_rate)
    ph_ids, ph_durations = build_phoneme_sequence(sung_notes, phoneme_map, tempo, frame_rate)

    print(f"  F0 帧数: {n_frames}")
    print(f"  音素数: {len(ph_ids)}")
    print(f"  F0 max={f0.max():.1f}Hz, 有声={(f0>0).sum()}/{n_frames} 帧")

    # 显示前 20 个音素
    ph_display = []
    for pid in ph_ids[:20]:
        for k, v in phoneme_map['p2i'].items():
            if v == pid:
                ph_display.append(k)
                break
        else:
            ph_display.append('?')
    print(f"  音素: {' '.join(ph_display)}...")

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

    # 推理
    ph_tensor = torch.LongTensor(ph_ids).unsqueeze(0).to(device)
    dur_tensor = torch.LongTensor(ph_durations).unsqueeze(0).to(device)
    f0_tensor = torch.FloatTensor(f0).unsqueeze(0).to(device)

    print(f"\n合成中...")
    with torch.no_grad():
        mu = model.cond_encoder(ph_tensor, dur_tensor, f0_tensor)
        print(f"  mu: std={mu.std():.4f}")

        # 论文 §5: flow_gen.sample (8步 Euler) → z_hat + f0_refined → DDSP
        z_hat, f0_refined = model.flow_gen.sample(mu, n_steps=8)
        print(f"  z_hat: std={z_hat.std():.4f}")
        waveform = model.ddsp_decoder(f0_tensor, z_hat)

    wav = waveform.squeeze().cpu().numpy()

    # 保存
    ust_name = Path(args.ust).stem
    out_path = os.path.join(args.output, f'{ust_name}_svs.wav')
    import soundfile as sf
    sf.write(out_path, wav, sr)
    print(f"\n完成! 保存: {out_path}")
    print(f"  时长: {len(wav) / sr:.2f}s")


if __name__ == '__main__':
    main()
