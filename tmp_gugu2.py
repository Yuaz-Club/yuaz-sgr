"""推理测试: 咕咕嘎嘎, 快语速, C4→A3"""
import sys, json, torch, numpy as np, soundfile as sf
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
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_110000.pt', map_location='cuda')
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)
print("Model loaded from model_110000.pt")

FR = sr / hop  # 93.75

# 咕咕嘎嘎 × 2组，更快语速
# 咕(g,u) 咕(g,u) 嘎(g,a) 嘎(g,a) | 咕(g,u) 咕(g,u) 嘎(g,a) 嘎(g,a)
# F0: C4(261.6) A3(220)

lyrics_f0 = [
    ('gu', ['g','u'], 261.6),
    ('gu', ['g','u'], 261.6),
    ('ga', ['g','a'], 220.0),
    ('ga', ['g','a'], 220.0),
    ('gu', ['g','u'], 261.6),
    ('gu', ['g','u'], 261.6),
    ('ga', ['g','a'], 220.0),
    ('ga', ['g','a'], 220.0),
]

note_dur_sec = 0.12  # 很快，像说话

ph_ids = []
ph_durs = []
note_lengths = []

for lyric, phs, f0 in lyrics_f0:
    nf = int(note_dur_sec * FR)
    note_lengths.append(nf)
    for p in phs:
        pid = pm['p2i'].get(p, pm['p2i']['SP'])
        ph_ids.append(pid)
    cf = max(1, int(nf * 0.2))
    vf = nf - cf
    ph_durs.append(cf)
    ph_durs.append(vf)

total_frames = sum(ph_durs)
f0_frames = np.zeros(total_frames, dtype=np.float32)
idx = 0
for lyric, phs, f0 in lyrics_f0:
    nf = int(note_dur_sec * FR)
    f0_frames[idx:idx+nf] = f0
    idx += nf

print(f"Frames: {total_frames}, Duration: {total_frames/sr*hop:.1f}s")
print(f"F0 segments: {[(l, f) for l, _, f in lyrics_f0]}")

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_durs).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor(note_lengths).unsqueeze(0).cuda()

print("\nInfer...")
with torch.no_grad():
    w, mu, z, f0_out, lag, dur = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t, predict_durations=False
    )
wav = w.squeeze().cpu().numpy()
wav = wav / (np.abs(wav).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/gugu_gaga2.wav', wav, sr)

f0_np = f0_out.squeeze().cpu().numpy()
print(f"Output: {len(wav)/sr:.1f}s")
print(f"F0: mean={f0_np.mean():.1f}Hz, min={f0_np.min():.1f}, max={f0_np.max():.1f}")

print("Done! gugu_gaga2.wav")
