"""诊断时长预测问题"""
import sys, json, torch, numpy as np, mido, soundfile as sf, os

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
sr = 24000; hop = 256; FR = sr / hop

ckpt_files = sorted([f for f in os.listdir('/root/autodl-tmp/yuaz_sgr/checkpoints') 
                     if f.startswith('model_') and f.endswith('.pt')])
model = SVSModel(config).to('cuda')
model.eval()
ckpt = torch.load(f'/root/autodl-tmp/yuaz_sgr/checkpoints/{ckpt_files[-1]}', map_location='cuda')
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)
print(f'Loaded {ckpt_files[-1]}: step {ckpt.get("step", 0)}')

# ── 思华年.mid ──
mid = mido.MidiFile('/root/autodl-tmp/yuaz_sgr/思华年.mid')
tempo = 500000
for track in mid.tracks:
    for msg in track:
        if msg.type == 'set_tempo': tempo = msg.tempo; break
    break
tpb = mid.ticks_per_beat

raw_notes = []; lyrics_events = []
for track in mid.tracks:
    now = 0
    for msg in track:
        now += msg.time
        if msg.type == 'lyrics' and msg.text.strip():
            lyrics_events.append((now, msg.text.strip()))
        if msg.type == 'note_on' and msg.velocity > 0:
            raw_notes.append((now, msg.note, True))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            raw_notes.append((now, msg.note, False))
raw_notes.sort(key=lambda x: x[0]); lyrics_events.sort(key=lambda x: x[0])
active = {}; note_list = []
for tick, note, is_on in raw_notes:
    if is_on: active[note] = tick
    else:
        st = active.pop(note, None)
        if st is not None: note_list.append((st, tick, note))
note_list.sort(key=lambda x: x[0])
min_l = min(len(lyrics_events), len(note_list))
lyrics = [l for _, l in lyrics_events[:min_l]]
note_list = note_list[:min_l]
def t2s(t): return mido.tick2second(t, tpb, tempo)
notes_sec = [(t2s(s), t2s(e), n) for s,e,n in note_list]

segments = []
for i, ((st, en, midi_n), lyric) in enumerate(zip(notes_sec, lyrics)):
    f0v = 440.0 * (2 ** ((midi_n - 69) / 12))
    segments.append(('note', en - st, f0v, lyric))
    if i < len(notes_sec) - 1:
        rest_d = notes_sec[i+1][0] - en
        if rest_d > 0.01: segments.append(('rest', rest_d, 0.0, None))

PY_TO_PH = {
    'yue':['y','ue'],'sui':['s','ui'],'zai':['z','ai'],
    'le':['l','e'],'mei':['m','ei'],'jian':['j','ian'],
    'meng':['m','eng'],'ye':['y','e'],'zui':['z','ui'],
    'qu':['q','v'],'zun':['z','un'],'qian':['q','ian'],
    'za':['z','a'],'xiang':['x','iang'],'ming':['m','ing'],
    'he':['h','e'],'zhong':['zh','ong'],'kan':['k','an'],
    'yi':['y','i'],'yan':['y','an'],'dao':['d','ao'],
    'mi':['m','i'],'li':['l','i'],'ren':['r','en'],
    'si':['s','i'],'shi':['sh','i'],'ni':['n','i'],
    'wo':['w','o'],'ta':['t','a'],'zhe':['zh','e'],
    'na':['n','a'],'bu':['b','u'],'hua':['h','ua'],
    'nian':['n','ian'],'you':['y','ou'],'zhu':['zh','u'],
    'shou':['sh','ou'],'xin':['x','in'],
}
VOWELS = {'ue','ui','ai','e','ei','ian','eng','v','un','iang','ing',
          'ong','an','i','ao','en','a','o','er','u','ang','ou','ua','in'}

all_phonemes = []; seg_ph_counts = []; last_vowel = None
for typ, dur, f0v, lyr in segments:
    if typ == 'note':
        phs = PY_TO_PH.get(lyr, ['SP']) if lyr != '+' else ([last_vowel] if last_vowel else ['a'])
        if lyr != '+' and len(phs) > 0: last_vowel = phs[-1]
    else: phs = ['SP']
    all_phonemes.extend(phs); seg_ph_counts.append(len(phs))
ph_ids = [pm['p2i'].get(p, 0) for p in all_phonemes]
seg_frames = [max(1, int(d * FR)) for _, d, _, _ in segments]

# 手动时长
C_W = 0.09; V_W = 0.91
ph_dur_manual = []
for s_idx, n_ph in enumerate(seg_ph_counts):
    nf = seg_frames[s_idx]
    if n_ph <= 1: ph_dur_manual.append(nf); continue
    ph_start = sum(seg_ph_counts[:s_idx])
    ph_list = all_phonemes[ph_start:ph_start+n_ph]
    n_cons = sum(1 for p in ph_list if p not in VOWELS and p != 'SP')
    n_vows = n_ph - n_cons
    if n_cons == 0 or n_vows == 0:
        base = nf // n_ph; rem = nf % n_ph
        for j in range(n_ph): ph_dur_manual.append(base + (1 if j < rem else 0))
    else:
        ctw = n_cons * C_W; vtw = n_vows * V_W; tw = ctw + vtw
        cf = max(1, int(nf * ctw / tw)); vf = nf - cf
        cp = cf // n_cons; cr = cf % n_cons
        cd = [cp + (1 if i < cr else 0) for i in range(n_cons)]
        vp = vf // n_vows; vr = vf % n_vows
        vd = [vp + (1 if i < vr else 0) for i in range(n_vows)]
        ci = 0; vi = 0
        for p in ph_list:
            if p not in VOWELS and p != 'SP': ph_dur_manual.append(cd[ci]); ci += 1
            else: ph_dur_manual.append(vd[vi]); vi += 1

total_pf = sum(ph_dur_manual)
f0_frames = []
for s_idx, (typ, dur, f0v, lyr) in enumerate(segments):
    f0_frames.extend([f0v] * seg_frames[s_idx])
f0_frames = np.array(f0_frames[:total_pf], dtype=np.float32)
note_frames = [seg_frames[i] for i, (typ, _, _, _) in enumerate(segments) if typ == 'note']

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_dur_manual).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor(note_frames).unsqueeze(0).cuda()

print(f"\n输入信息:")
print(f"  音素数: {len(ph_ids)}")
print(f"  音符数: {len(note_frames)}")
print(f"  手动时长: {ph_dur_manual[:30]}...")
print(f"  总帧数: {total_pf}")
print(f"  音符帧数: {note_frames[:30]}...")

# 测试 predict_durations=True
print(f"\n{'='*60}")
print(f"预测时长模式 (predict_durations=True)")
print(f"{'='*60}")
with torch.no_grad():
    mu_init = model.cond_encoder(phoneme_t, dur_t, f0_t.squeeze(1))
    f0_1d = f0_t.squeeze(1)
    dur_pred = model.temporal_predictor.predict_durations(mu_init, f0_1d, nl_t, phoneme_t, durations=dur_t)

print(f"预测时长: {dur_pred.squeeze().cpu().tolist()}")
print(f"预测总帧数: {dur_pred.sum().item():.0f}")
print(f"GT 总帧数: {total_pf}")

# 对比每个音符
print(f"\n{'='*60}")
print(f"音符级时长对比")
print(f"{'='*60}")
ph_idx = 0
for n in range(len(note_frames)):
    nl = note_frames[n]
    # 找这个音符对应的音素数量（用手动时长分组）
    n_ph_in_note = 0
    ph_start = ph_idx
    ph_frames = 0
    while ph_idx < len(ph_dur_manual) and ph_frames < nl:
        ph_frames += ph_dur_manual[ph_idx]
        n_ph_in_note += 1
        ph_idx += 1
    if n_ph_in_note == 0:
        continue
    pred_sum = dur_pred[0, ph_start:ph_start+n_ph_in_note].sum().item()
    print(f"  音符{n}: note_len={nl} ph_count={n_ph_in_note} GT_sum={sum(ph_dur_manual[ph_start:ph_start+n_ph_in_note])} pred_sum={pred_sum:.1f}")

# 计算GT vs pred 的偏差
dur_gt_t = torch.LongTensor(ph_dur_manual).unsqueeze(0).cuda()
dur_pred_clamped = dur_pred.clamp(min=1)
mask = dur_gt_t > 0
mse = torch.nn.functional.mse_loss(
    torch.log1p(dur_pred_clamped[mask]),
    torch.log1p(dur_gt_t[mask].float())
)
print(f"\n对数域 MSE (GT vs pred): {mse.item():.4f}")

# 听觉测试: 用 pred dur + orig f0 合成
with torch.no_grad():
    w_pred_dur, _, _, _, _, _ = model.forward_synthesis(
        phoneme_t, dur_pred_clamped.long(), f0_t, n_steps=1, note_lengths=nl_t,
        use_predicted_f0=False, predict_durations=False  # 传入预测的时长但不重新预测
    )
w_pred_dur = w_pred_dur.squeeze().cpu().numpy()
w_pred_dur = w_pred_dur / (np.abs(w_pred_dur).max() + 1e-8)
sf.write('/root/autodl-tmp/test_latest/pred_dur_diag.wav', w_pred_dur, sr)
print(f"\n诊断音频已保存: pred_dur_diag.wav ({len(w_pred_dur)/sr:.1f}s)")
