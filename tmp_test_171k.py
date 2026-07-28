"""测试 171K 检查点三种推理模式"""
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
sr = config.get('sample_rate', 24000)
hop = config.get('hop_length', 256)
FR = sr / hop

model = SVSModel(config).to('cuda')
model.eval()
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_171000.pt', map_location='cuda')
msd = model.state_dict()
loaded = 0; skipped = 0
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v; loaded += 1
    else:
        skipped += 1
model.load_state_dict(msd, strict=False)
print(f'Loaded step {ckpt.get("step", 0)}: {loaded} params, {skipped} skipped')

mid = mido.MidiFile('/root/autodl-tmp/yuaz_sgr/思华年.mid')
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

segments = []
for i, ((st, en, midi_n), lyric) in enumerate(zip(notes_sec, lyrics)):
    f0v = 440.0 * (2 ** ((midi_n - 69) / 12))
    segments.append(('note', en - st, f0v, lyric))
    if i < len(notes_sec) - 1:
        rest_d = notes_sec[i+1][0] - en
        if rest_d > 0.01:
            segments.append(('rest', rest_d, 0.0, None))

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
    else:
        phs = ['SP']
    all_phonemes.extend(phs); seg_ph_counts.append(len(phs))
ph_ids = [pm['p2i'].get(p, 0) for p in all_phonemes]
seg_frames = [max(1, int(d * FR)) for _, d, _, _ in segments]

# Manual durations (辅音~9%, 元音~91%)
C_W = 0.09; V_W = 0.91
ph_dur_manual = []
for s_idx, n_ph in enumerate(seg_ph_counts):
    nf = seg_frames[s_idx]
    if n_ph <= 1:
        ph_dur_manual.append(nf); continue
    ph_start = sum(seg_ph_counts[:s_idx])
    ph_list = all_phonemes[ph_start:ph_start+n_ph]
    n_cons = sum(1 for p in ph_list if p not in VOWELS and p != 'SP')
    n_vows = n_ph - n_cons
    if n_cons == 0 or n_vows == 0:
        base = nf // n_ph; rem = nf % n_ph
        for j in range(n_ph):
            ph_dur_manual.append(base + (1 if j < rem else 0))
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
                ph_dur_manual.append(cd[ci]); ci += 1
            else:
                ph_dur_manual.append(vd[vi]); vi += 1

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

OUT = '/root/autodl-tmp/test_171k'
os.makedirs(OUT, exist_ok=True)

print(f'\nPhonemes: {len(ph_ids)}, Total frames: {total_pf}')
print(f'Note frames: {note_frames}')
print(f'F0 range: {f0_frames.min():.0f} - {f0_frames.max():.0f} Hz')

# === Mode 1: 原版 F0 (use_predicted_f0=False) ===
print('\n[1/3] 原版 F0...')
with torch.no_grad():
    w1, _, _, _, _, _ = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t,
        use_predicted_f0=False, predict_durations=False
    )
w1 = w1.squeeze().cpu().numpy()
w1 = w1 / (np.abs(w1).max() + 1e-8)
sf.write(f'{OUT}/orig_f0.wav', w1, sr)
print(f'  Done: {len(w1)/sr:.1f}s')

# === Mode 2: 预测 F0 (use_predicted_f0=True) ===
print('\n[2/3] 预测 F0...')
with torch.no_grad():
    w2, _, _, _, _, _ = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t,
        use_predicted_f0=True, predict_durations=False
    )
w2 = w2.squeeze().cpu().numpy()
w2 = w2 / (np.abs(w2).max() + 1e-8)
sf.write(f'{OUT}/pred_f0.wav', w2, sr)
print(f'  Done: {len(w2)/sr:.1f}s')

# === Mode 3: 预测音素时长 ===
print('\n[3/3] 预测音素时长...')
with torch.no_grad():
    w3, _, _, _, _, _ = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t,
        use_predicted_f0=False, predict_durations=True
    )
w3 = w3.squeeze().cpu().numpy()
w3 = w3 / (np.abs(w3).max() + 1e-8)
sf.write(f'{OUT}/pred_dur.wav', w3, sr)
print(f'  Done: {len(w3)/sr:.1f}s')

# === 诊断 AE 重建 (用训练好的编码器直接重建) ===
print('\n[诊断] AE 重建 (encoder F0 + decoder)...')
from yuaz_sgr.data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json',
    bs=1, device='cuda',
)
batch = next(train_iter())
audio = batch['audio']; f0_audio = batch['f0_audio']
with torch.no_grad():
    z1, f0_e = model.encoder(audio, f0_override=f0_audio)
    z_q, _, _ = model.rvq(z1)
    ae_wav = model.ddsp_decoder(f0_e, z_q, target_samples=audio.shape[-1])
ae_wav = ae_wav.squeeze().cpu().numpy()
ae_wav = ae_wav / (np.abs(ae_wav).max() + 1e-8)
sf.write(f'{OUT}/ae_test.wav', ae_wav, sr)
print(f'  Done: {len(ae_wav)/sr:.1f}s')

print(f'\n完成: {OUT}/')
for f in ['orig_f0.wav', 'pred_f0.wav', 'pred_dur.wav', 'ae_test.wav']:
    fpath = os.path.join(OUT, f)
    if os.path.exists(fpath):
        print(f'  {fpath}')
