"""诊断: 对比 f0_refined vs MIDI F0 对音高的影响"""
import sys, json, torch, numpy as np, mido, soundfile as sf
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

model = SVSModel(config).to('cuda')
model.eval()
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_091000.pt', map_location='cuda')
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)

mid = mido.MidiFile('/root/autodl-tmp/yuaz_sgr/sihuanian.mid')
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
}
VOWELS = {'ue','ui','ai','e','ei','ian','eng','v','un','iang','ing',
          'ong','an','i','ao','en','a','o','er','u','ang','ou'}
all_phonemes = []; seg_ph_counts = []; last_vowel = None
for typ, dur, f0v, lyr in segments:
    if typ == 'note':
        phs = PY_TO_PH.get(lyr, ['SP']) if lyr != '+' else ([last_vowel] if last_vowel else ['a'])
        if lyr != '+' and len(phs) > 0: last_vowel = phs[-1]
    else: phs = ['SP']
    all_phonemes.extend(phs); seg_ph_counts.append(len(phs))
ph_ids = [pm['p2i'].get(p, 0) for p in all_phonemes]
FR = sr / hop
seg_frames_list = [max(1, int(d * FR)) for _, d, _, _ in segments]
C_W = 0.2; V_W = 0.8
ph_dur_frames = []
for s_idx, n_ph in enumerate(seg_ph_counts):
    nf = seg_frames_list[s_idx]
    if n_ph <= 1: ph_dur_frames.append(nf); continue
    ph_start = sum(seg_ph_counts[:s_idx])
    ph_list = all_phonemes[ph_start:ph_start+n_ph]
    n_cons = sum(1 for p in ph_list if p not in VOWELS and p != 'SP')
    n_vows = n_ph - n_cons
    if n_cons == 0 or n_vows == 0:
        base = nf // n_ph; rem = nf % n_ph
        for j in range(n_ph): ph_dur_frames.append(base + (1 if j < rem else 0))
    else:
        ctw = n_cons * C_W; vtw = n_vows * V_W; tw = ctw + vtw
        cf = max(1, int(nf * ctw / tw)); vf = nf - cf
        cp = cf // n_cons; cr = cf % n_cons
        cd = [cp + (1 if i < cr else 0) for i in range(n_cons)]
        vp = vf // n_vows; vr = vf % n_vows
        vd = [vp + (1 if i < vr else 0) for i in range(n_vows)]
        ci = 0; vi = 0
        for p in ph_list:
            if p not in VOWELS and p != 'SP': ph_dur_frames.append(cd[ci]); ci += 1
            else: ph_dur_frames.append(vd[vi]); vi += 1
total_pf = sum(ph_dur_frames)
f0_frames_full = []
for s_idx, (typ, dur, f0v, lyr) in enumerate(segments):
    f0_frames_full.extend([f0v] * seg_frames_list[s_idx])
f0_frames_full = np.array(f0_frames_full[:total_pf], dtype=np.float32)

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_dur_frames).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames_full).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor([seg_frames_list[i] for i, (typ, _, _, _) in enumerate(segments) if typ == 'note']).unsqueeze(0).cuda()

ddsp = model.ddsp_decoder

# 获取 mu, z_q, f0_refined
with torch.no_grad():
    mu = model.cond_encoder(phoneme_t, dur_t, f0_t)
    z, f0_refined = model.flow_gen.sample(mu, n_steps=1)
    z_q, _, _ = model.rvq(z)
    
    # 检查 f0_refined
    f0r = f0_refined.squeeze().cpu().numpy()
    f0m = f0_frames_full
    print(f"=== F0 Comparison ===")
    print(f"MIDI F0:   range {f0m[f0m>0].min():.1f}-{f0m.max():.1f}Hz, mean={f0m.mean():.1f}")
    print(f"refined F0: range {f0r.min():.1f}-{f0r.max():.1f}Hz, mean={f0r.mean():.1f}, std={f0r.std():.1f}")
    # 检查相关性
    min_len = min(len(f0r), len(f0m))
    corr = np.corrcoef(f0m[:min_len], f0r[:min_len])[0,1]
    print(f"correlation (MIDI vs refined): {corr:.4f}")
    # 检查是否 refined 是常数
    unique_vals = len(np.unique(np.round(f0r)))
    print(f"unique rounded values in refined: {unique_vals} / {len(f0r)}")
    
    # 检查 validity check
    f0r_abs = np.abs(f0r)
    f0r_mean = f0r_abs.mean()
    f0r_std = f0r_abs.std()
    f0in_mean = np.abs(f0m).mean()
    ratio = f0r_mean / f0in_mean
    print(f"Validity: mean={f0r_mean:.1f}, std={f0r_std:.1f}, ratio={ratio:.4f} (need 0.85-1.15)")

print(f"\n=== Generating test files ===")

# Test A: forward_synthesis 但强制用 MIDI F0
print("\n[A] forward_synthesis with MIDI F0 forced...")
with torch.no_grad():
    # Monkey-patch the validity check to always fail → uses MIDI F0
    w_a, _, _, f0_a, _, _ = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t, predict_durations=False
    )
    # But f0_out is still f0_refined (validity passed). Force override:
    # Actually let's redo by manually calling the parts.
    
    mu_a = model.cond_encoder(phoneme_t, dur_t, f0_t)
    z_a, f0r_a = model.flow_gen.sample(mu_a, n_steps=1)
    zq_a, _, _ = model.rvq(z_a)
    
    # Force MIDI F0
    f0_midi_aligned = f0_t
    if f0_midi_aligned.dim() == 2:
        f0_midi_aligned = f0_midi_aligned.unsqueeze(1)
    if f0_midi_aligned.shape[-1] != zq_a.shape[-1]:
        f0_midi_aligned = torch.nn.functional.interpolate(f0_midi_aligned, size=zq_a.shape[-1], mode='linear')
    
    w_midi = ddsp(f0_midi_aligned, zq_a, cond=mu_a)

wav_a = w_midi.squeeze().cpu().numpy()
wav_a = wav_a / (np.abs(wav_a).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/infer_MIDI_f0.wav', wav_a, sr)
f0a = f0_midi_aligned.squeeze().cpu().numpy()
print(f"  F0 range: {f0a.min():.1f}-{f0a.max():.1f}Hz")

# Test B: forward_synthesis 原始 (用 f0_refined)
print("\n[B] forward_synthesis original (f0_refined)...")
with torch.no_grad():
    mu_b = model.cond_encoder(phoneme_t, dur_t, f0_t)
    z_b, f0r_b = model.flow_gen.sample(mu_b, n_steps=1)
    zq_b, _, _ = model.rvq(z_b)
    
    f0_ref_aligned = f0r_b
    if f0_ref_aligned.dim() == 2:
        f0_ref_aligned = f0_ref_aligned.unsqueeze(1)
    if f0_ref_aligned.shape[-1] != zq_b.shape[-1]:
        f0_ref_aligned = torch.nn.functional.interpolate(f0_ref_aligned, size=zq_b.shape[-1], mode='linear')
    f0_ref_aligned = torch.clamp(f0_ref_aligned, min=50.0)
    
    w_ref = ddsp(f0_ref_aligned, zq_b, cond=mu_b)

wav_b = w_ref.squeeze().cpu().numpy()
wav_b = wav_b / (np.abs(wav_b).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/infer_REF_f0.wav', wav_b, sr)
f0b = f0_ref_aligned.squeeze().cpu().numpy()
print(f"  F0 range: {f0b.min():.1f}-{f0b.max():.1f}Hz")

# Test C: DDSP with MIDI F0 + z_q from mu (no flow matching) - same as diagH
print("\n[C] DDSP MIDI F0 + mu→z_q (no flow matching, baseline)...")
with torch.no_grad():
    mu_c = model.cond_encoder(phoneme_t, dur_t, f0_t)
    zq_c, _, _ = model.rvq(mu_c)
    f0_c = f0_t
    if f0_c.dim() == 2: f0_c = f0_c.unsqueeze(1)
    if f0_c.shape[-1] != zq_c.shape[-1]:
        f0_c = torch.nn.functional.interpolate(f0_c, size=zq_c.shape[-1], mode='linear')
    w_c = ddsp(f0_c, zq_c)  # no cond
wav_c = w_c.squeeze().cpu().numpy()
wav_c = wav_c / (np.abs(wav_c).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/infer_mu_zq.wav', wav_c, sr)

# Test D: DDSP with refined F0 + z_q from mu (NO flow matching z, just refined F0)
print("\n[D] DDSP refined F0 + mu→z_q (no flow z)...")
with torch.no_grad():
    w_d = ddsp(f0_ref_aligned, zq_c, cond=mu_c)
wav_d = w_d.squeeze().cpu().numpy()
wav_d = wav_d / (np.abs(wav_d).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/infer_refF0_muZ.wav', wav_d, sr)

# Test E: DDSP with MIDI F0 + flow_matching z_q (and cond)
print("\n[E] DDSP MIDI F0 + flow z_q + cond...")
with torch.no_grad():
    w_e = ddsp(f0_midi_aligned, zq_a, cond=mu_a)
wav_e = w_e.squeeze().cpu().numpy()
wav_e = wav_e / (np.abs(wav_e).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/infer_midiF0_flowZ.wav', wav_e, sr)

print("\n=== Summary ===")
print("infer_MIDI_f0.wav     — MIDI F0 + flow z_q + cond (Test A)")
print("infer_REF_f0.wav      — refined F0 + flow z_q + cond (Test B, = original)")
print("infer_mu_zq.wav       — MIDI F0 + mu→z_q, no cond (Test C, = diagH baseline)")
print("infer_refF0_muZ.wav   — refined F0 + mu→z_q, cond (Test D)")
print("infer_midiF0_flowZ.wav— MIDI F0 + flow z_q, cond (Test E)")
print("\nLogic:")
print("  A has pitch, B doesn't → f0_refined is the killer")
print("  C has pitch, A doesn't → flow matching z is the killer")
print("  E has pitch, A doesn't → cond=mu is the killer (unlikely)")
