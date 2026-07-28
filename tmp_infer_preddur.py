"""推理测试: model_123000 — 虚拟.vsqx 用时长预测器"""
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
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_123000.pt', map_location='cuda')
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)
print("Model loaded from model_123000.pt")

# 解析 VSQX
NS = 'http://www.yamaha.co.jp/vocaloid/schema/vsq4/'
tree = ET.parse('/root/autodl-tmp/yuaz_sgr/xuni.vsqx')
root = tree.getroot()

mt = root.find(f'{{{NS}}}masterTrack')
resolution = int(mt.find(f'{{{NS}}}resolution').text)
tempo_el = mt.find(f'{{{NS}}}tempo')
tempo_val = int(tempo_el.find(f'{{{NS}}}v').text)
tempo_bpm = tempo_val / 100.0

def ticks_to_sec(t):
    return t / resolution * 60.0 / tempo_bpm

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

VOCO_TO_OURS = {
    'k': 'k', 't': 't', 'p': 'p', 's': 's', 'x': 'x',
    'd': 'd', 'l': 'l', 'm': 'm', 'n': 'n', 'r': 'r',
    'h': 'h', 'z': 'z', 'q': 'q', 'f': 'f', 'g': 'g',
    'j': 'j', 'c': 'c', 'b': 'b',
    'ts': 'zh', 'tsh': 'ch', 'S': 'sh',
    'a': 'a', 'o': 'o', 'e': 'e', 'i': 'i', 'u': 'u', 'v': 'v',
    'ai': 'ai', 'ei': 'ei', 'ao': 'ao', 'ou': 'ou',
    'an': 'an', 'en': 'en', 'ang': 'ang', 'eng': 'eng', 'er': 'er',
    'ia': 'ia', 'ie': 'ie', 'ua': 'ua', 'uo': 'uo', 'ue': 'ue',
    'iao': 'iao', 'iu': 'iu', 'ian': 'ian', 'in': 'in',
    'iang': 'iang', 'ing': 'ing', 'ui': 'ui', 'un': 'un',
    'ong': 'ong', 'iong': 'iong',
    'aI': 'ai', 'aU': 'ao', 'eI': 'ei', 'oU': 'ou',
    '7': 'e', '@': 'e', '@U': 'ou', 'i@U': 'iu',
    '@N': 'en', 'iun': 'un', 'ye': 'ue', 'y': 'y',
    'au': 'ao', 'iau': 'iao', 'uei': 'ui',
    'u@': 'ua', 'u@N': 'uan',
    'y{': 'i', 'i\\': 'i', 's\\': 's', 'ts\\': 'zh',
    'p_h': 'p', 't_h': 't', 'k_h': 'k',
    'ts_h': 'ch', 'tS': 'ch', 'dz': 'z',
    'y_n': 'y', 'w': 'u', 'uAN': 'uan',
}

def parse_voco_phonemes(p_str):
    if not p_str or p_str == '-':
        return ['SP']
    parts = p_str.strip().split()
    result = []
    for part in parts:
        part = part.replace('`', '').replace('\\', '')
        if '_' in part:
            sub_parts = part.split('_')
            for sp in sub_parts:
                sp = sp.replace('h', '')
                if sp:
                    mapped = VOCO_TO_OURS.get(sp) or VOCO_TO_OURS.get(sp.lower()) or 'SP'
                    result.append(mapped)
        else:
            part = part.replace('h', '')
            if part:
                mapped = VOCO_TO_OURS.get(part) or VOCO_TO_OURS.get(part.lower()) or 'SP'
                result.append(mapped)
    return result

segments = []
all_phonemes = []
note_lengths_frames = []

for i, note in enumerate(notes):
    start_tick = note['t']
    dur_tick = note['dur']
    midi_n = note['note']
    f0v = 440.0 * (2 ** ((midi_n - 69) / 12))
    dur_sec = ticks_to_sec(dur_tick)
    phs = parse_voco_phonemes(note['phonemes'])
    all_phonemes.extend(phs)
    n_frames = max(1, int(dur_sec * FR))
    note_lengths_frames.append(n_frames)
    segments.append(('note', dur_sec, f0v, note['lyric'], phs, n_frames))
    if i < len(notes) - 1:
        next_start = ticks_to_sec(notes[i+1]['t'])
        gap = next_start - (ticks_to_sec(start_tick) + dur_sec)
        if gap > 0.02:
            gap_frames = max(1, int(gap * FR))
            all_phonemes.append('SP')
            segments.append(('rest', gap, 0.0, None, ['SP'], gap_frames))

ph_ids = []
for p in all_phonemes:
    pid = pm['p2i'].get(p)
    if pid is None: pid = pm['p2i'].get('SP', 0)
    ph_ids.append(pid)

ph_dur_frames = []
for seg_type, dur_sec, f0v, lyric, phs, n_frames in segments:
    n_ph = len(phs)
    if n_ph == 1:
        ph_dur_frames.append(n_frames)
    else:
        for j in range(n_ph - 1):
            ph_dur_frames.append(max(1, int(n_frames * 0.2)))
        remaining = n_frames - sum(ph_dur_frames[-(n_ph-1):]) if n_ph > 1 else n_frames
        ph_dur_frames.append(max(1, remaining))

total_frames = sum(ph_dur_frames)
f0_frames = []
for seg_type, dur_sec, f0v, lyric, phs, n_frames in segments:
    f0_frames.extend([f0v] * n_frames)
f0_frames = np.array(f0_frames[:total_frames], dtype=np.float32)

print(f"Notes: {len(notes)}, Phonemes: {len(all_phonemes)}, Duration: {total_frames/sr*hop:.1f}s")

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_dur_frames).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor(note_lengths_frames).unsqueeze(0).cuda()

# 只做预测时长推理
print("\n[1] Infer with predicted durations (model_123000)...")
with torch.no_grad():
    w, mu, z, f0_out, lag, dur_pred = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t, predict_durations=True
    )
wav = w.squeeze().cpu().numpy()
wav = wav / (np.abs(wav).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/infer_xuni_preddur.wav', wav, sr)

# 分析预测时长分布
dur_np = dur_pred.squeeze().cpu().numpy()
gt_dur_np = np.array(ph_dur_frames)
print(f"  Predicted dur: mean={dur_np.mean():.1f}, min={dur_np.min():.1f}, max={dur_np.max():.1f}")
print(f"  GT dur:        mean={gt_dur_np.mean():.1f}, min={gt_dur_np.min():.1f}, max={gt_dur_np.max():.1f}")
print(f"  Output: {len(wav)/sr:.1f}s")

# 打印前 20 个音素的 GT vs 预测时长对比
print("\n  First 20 phoneme durations (GT vs Pred):")
for i in range(min(20, len(ph_ids))):
    ph_name = [k for k, v in pm['p2i'].items() if v == ph_ids[i]]
    ph_name = ph_name[0] if ph_name else '?'
    print(f"    {ph_name:6s}: GT={gt_dur_np[i]:4d}  Pred={dur_np[i]:.1f}")

print("\nDone! infer_xuni_preddur.wav")
