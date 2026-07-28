import sys, json, torch, numpy as np, mido, soundfile as sf
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel
import yaml, torch.nn.functional as F

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
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_081000.pt', map_location='cuda')
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)

VOWELS = {'ue','ui','ai','e','ei','ian','eng','v','un','iang','ing',
          'ong','an','i','ao','en','a','o','er','u','ang','ou'}

mid = mido.MidiFile('/root/autodl-tmp/yuaz_sgr/sihuanian.mid')
tempo = 500000
for track in mid.tracks:
    for msg in track:
        if msg.type == 'set_tempo':
            tempo = msg.tempo; break
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

# Take first 5 notes for diagnostic
test_notes = notes_sec[:5]
test_lyrics = lyrics[:5]
PY_TO_PH = {
    'yue':['y','ue'],'sui':['s','ui'],'zai':['z','ai'],
    'le':['l','e'],'mei':['m','ei'],'jian':['j','ian'],
}

segments = []
for i, ((st, en, midi_n), lyric) in enumerate(zip(test_notes, test_lyrics)):
    f0v = 440.0 * (2 ** ((midi_n - 69) / 12))
    segments.append(('note', en - st, f0v, lyric))
    if i < len(test_notes) - 1:
        rest_d = test_notes[i+1][0] - en
        if rest_d > 0.01:
            segments.append(('rest', rest_d, 0.0, None))

all_phonemes = []; seg_ph_counts = []; last_vowel = None
for typ, dur, f0v, lyr in segments:
    if typ == 'note':
        phs = PY_TO_PH.get(lyr, ['SP']) if lyr != '+' else ([last_vowel] if last_vowel else ['a'])
        if lyr != '+' and len(phs) > 0: last_vowel = phs[-1]
    else:
        phs = ['SP']
    all_phonemes.extend(phs); seg_ph_counts.append(len(phs))
ph_ids = [pm['p2i'].get(p, 0) for p in all_phonemes]
seg_frames = [max(1, int(d * FR)) for _, d, _, _ in segments]

C_W = 0.2; V_W = 0.8
ph_dur_frames = []
for s_idx, n_ph in enumerate(seg_ph_counts):
    nf = seg_frames[s_idx]
    if n_ph <= 1:
        ph_dur_frames.append(nf); continue
    ph_start = sum(seg_ph_counts[:s_idx])
    ph_list = all_phonemes[ph_start:ph_start+n_ph]
    n_cons = sum(1 for p in ph_list if p not in VOWELS and p != 'SP')
    n_vows = n_ph - n_cons
    if n_cons == 0 or n_vows == 0:
        base = nf // n_ph; rem = nf % n_ph
        for j in range(n_ph):
            ph_dur_frames.append(base + (1 if j < rem else 0))
    else:
        ctw = n_cons * C_W; vtw = n_vows * V_W; tw = ctw + vtw
        cf = max(1, int(nf * ctw / tw)); vf = nf - cf
        cp = cf // n_cons; cr = cf % n_cons
        cd = [cp + (1 if i < cr else 0) for i in range(n_cons)]
        vp = vf // n_vows; vr = vf % n_vows
        vd = [vp + (1 if i < vr else 0) for i in range(n_vows)]
        ci = 0; vi = 0
        for p in ph_list:
            if p not in VOWELS and p != 'SP':
                ph_dur_frames.append(cd[ci]); ci += 1
            else:
                ph_dur_frames.append(vd[vi]); vi += 1
total_pf = sum(ph_dur_frames)
f0_frames = []
for s_idx, (typ, dur, f0v, lyr) in enumerate(segments):
    f0_frames.extend([f0v] * seg_frames[s_idx])
f0_frames = np.array(f0_frames[:total_pf], dtype=np.float32)
note_frames = [seg_frames[i] for i, (typ, _, _, _) in enumerate(segments) if typ == 'note']

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_dur_frames).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor(note_frames).unsqueeze(0).cuda()

# Hook DDSP decoder to capture gate/AP
captured_gate = []
captured_ap = []
orig_forward = model.ddsp_decoder.forward

def hooked_forward(f0_in, z_in, target_samples=None):
    if f0_in.dim() == 2:
        f0_in = f0_in.unsqueeze(1)
    B, _, T = f0_in.shape
    feat = torch.cat([z_in, f0_in], dim=1)
    feat = feat.transpose(1, 2)
    feat = model.ddsp_decoder.emformer_input_proj(feat)
    h, _ = model.ddsp_decoder.emformer(feat)
    h = model.ddsp_decoder.emformer_proj(h)
    h = h.transpose(1, 2)
    gate = model.ddsp_decoder.weight_net(h.transpose(1, 2)).transpose(1, 2)
    ap = model.ddsp_decoder.ap_net(h.transpose(1, 2)).transpose(1, 2)
    captured_gate.append(gate.detach().cpu())
    captured_ap.append(ap.detach().cpu())
    return orig_forward(f0_in, z_in, target_samples)

model.ddsp_decoder.forward = hooked_forward

with torch.no_grad():
    wav, mu, z, f0_refined, lag_pred, dur_pred = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t,
        predict_durations=False
    )

# mu is (B, 128, T_enc), z is (B, 128, T_enc)
# gate is (B, 1, T_enc) - same T as mu/z
# Align: phoneme durations in encoder frames
gate_t = captured_gate[0][0, 0]  # (T_enc,)
ap_t = captured_ap[0][0].mean(dim=0)  # (T_enc,) - mean over 16 bands

print('=== DDSP Gate/AP Diagnostic (per phoneme) ===')
print(f'Phoneme   Frames  Gate_mean  Gate_std  AP_mean   AP_std   Type')
pos = 0
for p_idx, p_name in enumerate(all_phonemes):
    p_frames = ph_dur_frames[p_idx]
    if p_frames <= 0 or pos >= len(gate_t):
        continue
    end = min(pos + p_frames, len(gate_t))
    if end <= pos:
        continue
    g = gate_t[pos:end]
    a = ap_t[pos:end]
    ptype = 'VOWEL' if p_name in VOWELS else ('REST' if p_name == 'SP' else 'CONS')
    print(f'{p_name:8s} {p_frames:6d}  {g.mean():.4f}     {g.std():.4f}    {a.mean():.4f}    {a.std():.4f}   {ptype}')
    pos = end

print(f'\n--- Overall ---')
print(f'Gate: mean={gate_t.mean():.4f} std={gate_t.std():.4f}')
print(f'AP:   mean={ap_t.mean():.4f} std={ap_t.std():.4f}')

for ptype in ['CONS', 'VOWEL']:
    indices = []
    pos = 0
    for p_idx, p_name in enumerate(all_phonemes):
        pf = ph_dur_frames[p_idx]
        is_type = (ptype == 'VOWEL' and p_name in VOWELS) or (ptype == 'CONS' and p_name not in VOWELS and p_name != 'SP')
        if is_type and pos < len(gate_t):
            end = min(pos + pf, len(gate_t))
            if end > pos:
                indices.extend(range(pos, end))
        pos += pf
    if indices:
        g_t = gate_t[indices]
        a_t = ap_t[indices]
        n = len(indices)
        print(f'{ptype}(n={n}): Gate={g_t.mean():.4f}+-{g_t.std():.3f}  AP={a_t.mean():.4f}+-{a_t.std():.3f}')
