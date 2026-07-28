"""VSQX 解析 + 推理: 虚拟.vsqx 自带音素时长"""
import sys, json, torch, numpy as np, soundfile as sf
import xml.etree.ElementTree as ET
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel
import yaml

config = {}
for name in ['base', 'model']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']
sr = config.get('sample_rate', 24000)
hop = config.get('hop_length', 256)
FR = sr / hop

model = SVSModel(config).to('cuda')
model.eval()
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_091000.pt', map_location='cuda')
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)
print("Model loaded.")

# 解析 VSQX
NS = 'http://www.yamaha.co.jp/vocaloid/schema/vsq4/'
tree = ET.parse('/root/autodl-tmp/yuaz_sgr/xuni.vsqx')
root = tree.getroot()

# 获取 resolution 和 tempo
mt = root.find(f'{{{NS}}}masterTrack')
resolution = int(mt.find(f'{{{NS}}}resolution').text)  # 480
tempo_el = mt.find(f'{{{NS}}}tempo')
tempo_val = int(tempo_el.find(f'{{{NS}}}v').text)  # 6500
tempo_bpm = tempo_val / 100.0  # 65.00 BPM
print(f"Resolution: {resolution} tpq, Tempo: {tempo_bpm} BPM")

def ticks_to_sec(t):
    return t / resolution * 60.0 / tempo_bpm

# 提取所有 note
notes = []
vs_track = root.find(f'{{{NS}}}vsTrack')
for vs_part in vs_track.findall(f'{{{NS}}}vsPart'):
    for note_el in vs_part.findall(f'{{{NS}}}note'):
        t = int(note_el.find(f'{{{NS}}}t').text)
        dur = int(note_el.find(f'{{{NS}}}dur').text)
        n = int(note_el.find(f'{{{NS}}}n').text)
        v = int(note_el.find(f'{{{NS}}}v').text)
        y = note_el.find(f'{{{NS}}}y').text or ''
        p = note_el.find(f'{{{NS}}}p').text or ''
        notes.append({'t': t, 'dur': dur, 'note': n, 'vel': v, 'lyric': y, 'phonemes': p})

print(f"Total notes: {len(notes)}")

# VOCALOID 音素 → 我们的音素映射
# VOCALOID 格式: 空格分隔, ` 表示音高载体, _ 连接, h 送气, 大写特殊
# 我们的格式: 小写拼音 (如 'k', 'u', 'ts', 'i', 'ia', 'y', 'n', 't', 'e', 'p', 'aI', 'a', 'o')
VOCO_TO_OURS = {
    # 声母
    'k': 'k', 't': 't', 'p': 'p', 's': 's', 'x': 'x',
    'd': 'd', 'l': 'l', 'm': 'm', 'n': 'n', 'r': 'r',
    'h': 'h', 'z': 'z', 'q': 'q', 'f': 'f', 'g': 'g',
    'j': 'j', 'c': 'c', 'b': 'b',
    'ts': 'zh', 'tsh': 'ch', 'S': 'sh',  # VOCALOID 翘舌音
    # 韵母
    'a': 'a', 'o': 'o', 'e': 'e', 'i': 'i', 'u': 'u', 'v': 'v',
    'ai': 'ai', 'ei': 'ei', 'ao': 'ao', 'ou': 'ou',
    'an': 'an', 'en': 'en', 'ang': 'ang', 'eng': 'eng', 'er': 'er',
    'ia': 'ia', 'ie': 'ie', 'ua': 'ua', 'uo': 'uo', 'ue': 'ue',
    'iao': 'iao', 'iu': 'iu', 'ian': 'ian', 'in': 'in',
    'iang': 'iang', 'ing': 'ing', 'ui': 'ui', 'un': 'un',
    'ong': 'ong', 'iong': 'iong',
    'aI': 'ai', 'aU': 'ao', 'eI': 'ei', 'oU': 'ou',  # VOCALOID 大写韵母
    # VOCALOID 特殊符号
    '7': 'e',       # de 的特殊韵母
    '@': 'e',       # schwa → e
    '@U': 'ou',     # schwa+U → ou
    'i@U': 'iu',    # i+schwa+U → iu
    '@N': 'en',     # schwa+N → en
    'iun': 'un',    # i+un → un (实际是 ün)
    'ye': 'ue',     # ye → üe
    'y': 'y',       # 半元音
    'au': 'ao',     # au → ao
    'iau': 'iao',   # iau → iao
    'uei': 'ui',    # uei → ui
    'u@': 'ua',     # u+schwa → ua
    'u@N': 'uan',   # u+schwa+N → uan
    'y{': 'i',      # y{ → i (VOCALOID 特殊)
    'i\\': 'i',     # i\ → i
    's\\': 's',     # s\ → s
    'ts\\': 'zh',   # ts\ → zh
    # 送气/特殊辅音
    'p_h': 'p',
    't_h': 't',
    'k_h': 'k',
    'ts_h': 'ch',
    'tS': 'ch',
    'dz': 'z',
    # 连接器
    'y_n': 'y',
    'w': 'u',
}

def parse_voco_phonemes(p_str):
    """解析 VOCALOID 音素字符串 → 我们的音素列表"""
    if not p_str or p_str == '-':
        return ['SP']
    
    # 按空格分割
    parts = p_str.strip().split()
    result = []
    for part in parts:
        # 去掉 ` 和 \ (V1/V2 音高标记)
        part = part.replace('`', '').replace('\\', '')
        # 处理 _ (连接)
        if '_' in part:
            sub_parts = part.split('_')
            for sp in sub_parts:
                sp = sp.replace('h', '')  # 去掉送气标记
                if sp:
                    mapped = VOCO_TO_OURS.get(sp)
                    if mapped is None:
                        # 尝试小写匹配
                        mapped = VOCO_TO_OURS.get(sp.lower())
                    if mapped is None:
                        print(f"  WARNING: unknown phoneme '{sp}' (from '{part}'), using SP")
                        mapped = 'SP'
                    result.append(mapped)
        else:
            part = part.replace('h', '')
            if part:
                mapped = VOCO_TO_OURS.get(part)
                if mapped is None:
                    mapped = VOCO_TO_OURS.get(part.lower())
                if mapped is None:
                    print(f"  WARNING: unknown phoneme '{part}' (from '{p_str}'), using SP")
                    mapped = 'SP'
                result.append(mapped)
    return result

# 构建 segment 列表
segments = []
all_phonemes = []
note_lengths_frames = []

for i, note in enumerate(notes):
    start_tick = note['t']
    dur_tick = note['dur']
    midi_n = note['note']
    f0v = 440.0 * (2 ** ((midi_n - 69) / 12))
    
    start_sec = ticks_to_sec(start_tick)
    dur_sec = ticks_to_sec(dur_tick)
    end_sec = start_sec + dur_sec
    
    phs = parse_voco_phonemes(note['phonemes'])
    all_phonemes.extend(phs)
    
    n_frames = max(1, int(dur_sec * FR))
    note_lengths_frames.append(n_frames)
    segments.append(('note', dur_sec, f0v, note['lyric'], phs, n_frames))
    
    # 检查与下一个音符之间是否有间隙
    if i < len(notes) - 1:
        next_start = ticks_to_sec(notes[i+1]['t'])
        gap = next_start - end_sec
        if gap > 0.02:
            gap_frames = max(1, int(gap * FR))
            all_phonemes.append('SP')
            segments.append(('rest', gap, 0.0, None, ['SP'], gap_frames))

# 映射音素 ID
ph_ids = []
for p in all_phonemes:
    pid = pm['p2i'].get(p)
    if pid is None:
        print(f"  WARNING: unknown phoneme '{p}', using SP")
        pid = pm['p2i'].get('SP', 0)
    ph_ids.append(pid)

# 构建音素时长（帧）
ph_dur_frames = []
for seg_type, dur_sec, f0v, lyric, phs, n_frames in segments:
    n_ph = len(phs)
    if n_ph == 1:
        ph_dur_frames.append(n_frames)
    else:
        # 辅音 20%, 元音 80% 分配
        base = n_frames // n_ph
        rem = n_frames % n_ph
        for j in range(n_ph - 1):
            ph_dur_frames.append(max(1, int(n_frames * 0.2)))
        # 剩余全给最后一个音素（元音）
        remaining = n_frames - sum(ph_dur_frames[-(n_ph-1):]) if n_ph > 1 else n_frames
        ph_dur_frames.append(max(1, remaining))

# 构建 F0
total_frames = sum(ph_dur_frames)
f0_frames = []
for seg_type, dur_sec, f0v, lyric, phs, n_frames in segments:
    f0_frames.extend([f0v] * n_frames)
f0_frames = np.array(f0_frames[:total_frames], dtype=np.float32)

print(f"Segments: {len(segments)}")
print(f"Total phonemes: {len(all_phonemes)}")
print(f"Total frames: {total_frames}, {total_frames/sr*hop:.1f}s")
print(f"F0 range: {f0_frames[f0_frames>0].min():.1f}-{f0_frames.max():.1f}Hz")
print(f"Phoneme IDs: {ph_ids[:20]}...")

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_dur_frames).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor(note_lengths_frames).unsqueeze(0).cuda()

# 推理
print("\nInference (manual durations from VSQX)...")
with torch.no_grad():
    w, mu, z, f0_out, lag, dur_pred = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t, predict_durations=False
    )
wav = w.squeeze().cpu().numpy()
wav = wav / (np.abs(wav).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/infer_xuni.wav', wav, sr)

# 也做模型预测时长版本
print("Inference (model-predicted durations)...")
with torch.no_grad():
    w2, mu2, z2, f0_out2, lag2, dur_pred2 = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t, predict_durations=True
    )
wav2 = w2.squeeze().cpu().numpy()
wav2 = wav2 / (np.abs(wav2).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/infer_xuni_pred.wav', wav2, sr)

f0_np = f0_out.squeeze().cpu().numpy()
print(f"\nF0 output: {f0_np[f0_np>0].min():.1f}-{f0_np.max():.1f}Hz")
print(f"Done! infer_xuni.wav and infer_xuni_pred.wav")
