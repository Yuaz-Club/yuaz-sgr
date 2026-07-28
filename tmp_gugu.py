"""推理测试: 咕咕嘎嘎 C4 音高"""
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

# 咕咕嘎嘎 → 音素
# 咕: g u, 咕: g u, 嘎: g a, 嘎: g a
# 每个音符 C4 = 261.6Hz
# 每个音节 0.5 秒, 共 4 个音节约 2 秒
FR = sr / hop  # 93.75

f0_c4 = 261.6
note_dur_sec = 0.6  # 每个音节 0.6 秒
n_notes = 4
total_frames = int(n_notes * note_dur_sec * FR)

# 音素: gu[g,u] gu[g,u] ga[g,a] ga[g,a]
lyrics = [('gu', ['g','u']), ('gu', ['g','u']), ('ga', ['g','a']), ('ga', ['g','a'])]

ph_ids = []
ph_durs = []
note_lengths = []

for i, (lyric, phs) in enumerate(lyrics):
    nf = int(note_dur_sec * FR)
    note_lengths.append(nf)
    for p in phs:
        pid = pm['p2i'].get(p, pm['p2i']['SP'])
        ph_ids.append(pid)
    # 辅音 20%, 元音 80%
    n_cons = 1  # g
    n_vows = 1  # u/a
    cf = max(1, int(nf * 0.2))
    vf = nf - cf
    ph_durs.append(cf)  # g
    ph_durs.append(vf)  # u/a

total_frames = sum(ph_durs)
f0_frames = np.full(total_frames, f0_c4, dtype=np.float32)

print(f"Phonemes: {len(ph_ids)}, Frames: {total_frames}")
print(f"  IDs: {ph_ids}")
print(f"  Durs: {ph_durs}")
print(f"  Output len: {total_frames/sr*hop:.1f}s")

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_durs).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor(note_lengths).unsqueeze(0).cuda()

print("\n[1] VSQX durations...")
with torch.no_grad():
    w, mu, z, f0_out, lag, dur = model.forward_synthesis(
        phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t, predict_durations=False
    )
wav = w.squeeze().cpu().numpy()
wav = wav / (np.abs(wav).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/gugu_gaga.wav', wav, sr)
print(f"  Output: {len(wav)/sr:.1f}s, F0: {f0_out.squeeze().cpu().numpy().mean():.1f}Hz")

print("\nDone! gugu_gaga.wav")
