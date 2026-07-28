import sys, json, torch, numpy as np, mido
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
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_089000.pt', map_location='cuda')
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

test_notes = notes_sec[:3]; test_lyrics = lyrics[:3]
PY_TO_PH = {'yue':['y','ue'],'sui':['s','ui'],'zai':['z','ai'],'le':['l','e'],'mei':['m','ei']}
VOWELS = {'ue','ui','ai','e','ei','ian','eng','v','un','iang','ing','ong','an','i','ao','en','a','o','er','u','ang','ou'}

segments = []
for i, ((st, en, midi_n), lyric) in enumerate(zip(test_notes, test_lyrics)):
    f0v = 440.0 * (2 ** ((midi_n - 69) / 12))
    segments.append(('note', en - st, f0v, lyric))
    if i < len(test_notes) - 1:
        rest_d = test_notes[i+1][0] - en
        if rest_d > 0.01: segments.append(('rest', rest_d, 0.0, None))

all_phonemes = []; seg_ph_counts = []; last_vowel = None
for typ, dur, f0v, lyr in segments:
    if typ == 'note':
        phs = PY_TO_PH.get(lyr, ['SP']) if lyr != '+' else ([last_vowel] if last_vowel else ['a'])
        if lyr != '+' and len(phs) > 0: last_vowel = phs[-1]
    else: phs = ['SP']
    all_phonemes.extend(phs); seg_ph_counts.append(len(phs))
ph_ids = [pm['p2i'].get(p, 0) for p in all_phonemes]
seg_frames = [max(1, int(d * FR)) for _, d, _, _ in segments]

C_W = 0.2; V_W = 0.8
ph_dur_frames = []
for s_idx, n_ph in enumerate(seg_ph_counts):
    nf = seg_frames[s_idx]
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
f0_frames = []
for s_idx, (typ, dur, f0v, lyr) in enumerate(segments):
    f0_frames.extend([f0v] * seg_frames[s_idx])
f0_frames = np.array(f0_frames[:total_pf], dtype=np.float32)
note_frames = [seg_frames[i] for i, (typ, _, _, _) in enumerate(segments) if typ == 'note']

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_dur_frames).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor(note_frames).unsqueeze(0).cuda()

# Hook to capture f0 values at DDSP input
captured_f0_in = []
captured_f0_out = []
orig_fwd = model.ddsp_decoder.forward
def hooked_fwd(f0_in, z_in, target_samples=None, cond=None):
    captured_f0_in.append(f0_in.detach().cpu())
    result = orig_fwd(f0_in, z_in, target_samples, cond)
    return result
model.ddsp_decoder.forward = hooked_fwd

# Test predict_durations=False (manual durations)
print("=== Manual durations (predict_durations=False) ===")
with torch.no_grad():
    wav, mu, z, f0_out, lag_pred, dur_pred = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t,
        predict_durations=False
    )

f0_in = captured_f0_in[0][0,0]
print(f"Input F0: mean={f0_t.mean():.1f} std={f0_t.std():.1f} min={f0_t.min():.1f} max={f0_t.max():.1f}")
print(f"F0 to DDSP: mean={f0_in.mean():.1f} std={f0_in.std():.1f} min={f0_in.min():.1f} max={f0_in.max():.1f}")
print(f"F0_out from synthesis: mean={f0_out.mean():.1f} std={f0_out.std():.1f}")

# Check f0_refined
print(f"\nf0_refined from flow:")
import torch.nn.functional as F
with torch.no_grad():
    z_flow, f0_refined = model.flow_gen.sample(mu, n_steps=1)
    fr = f0_refined.abs()
    print(f"  f0_refined shape: {f0_refined.shape}")
    print(f"  f0_refined abs: mean={fr.mean():.1f} std={fr.std():.1f} min={fr.min():.1f} max={fr.max():.1f}")
    print(f"  Input F0 at encoder frames: mean={f0_t.mean():.1f}")

# Check the use_refined logic
print(f"\n=== F0_refined use check ===")
f0_in_1d = f0_t.squeeze(1)
f0r = f0_refined.abs()
f0r_mean = f0r.mean().item()
f0r_std = f0r.std().item()
f0in_mean = f0_t.abs().mean().item()
print(f"f0r_mean={f0r_mean:.1f} (need 50-2000)")
print(f"f0r_std={f0r_std:.1f} (need > 10.0)")
print(f"f0in_mean={f0in_mean:.1f} (need > 0)")
if 50 < f0r_mean < 2000 and f0r_std > 10.0 and f0in_mean > 0:
    ratio = f0r_mean / f0in_mean
    print(f"ratio={ratio:.2f} (need 0.7-1.3)")
else:
    print("=> FALLBACK: using original MIDI F0")
