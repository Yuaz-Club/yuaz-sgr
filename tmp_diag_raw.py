"""诊断: 逐层剥离 DDSP 信号链, 定位音高消失的根源"""
import sys, json, torch, numpy as np, mido, soundfile as sf
import torch.nn.functional as F
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
FR = sr / config.get('hop_length', 256)
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

phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
dur_t = torch.LongTensor(ph_dur_frames).unsqueeze(0).cuda()
f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor([seg_frames[i] for i, (typ, _, _, _) in enumerate(segments) if typ == 'note']).unsqueeze(0).cuda()

# ========== 诊断 1: 纯谐波振荡器原始输出 (绕过 env_net, 绕过 STFT) ==========
print("Diagnostic 1: Raw harmonic oscillator...")
# 取前 10 帧的 F0 值
f0_test = f0_t[:, :, :10]  # (1, 1, 10)
n_samp = 10 * 256  # 10 frames @ hop=256
with torch.no_grad():
    e_h_raw = model.ddsp_decoder.harmonic_oscillator(f0_test, n_samp)
e_h_raw = e_h_raw.squeeze().cpu().numpy()
e_h_raw = e_h_raw / (np.abs(e_h_raw).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diag1_raw_harm.wav', e_h_raw, sr)
print(f"  F0 test values: {f0_test.squeeze().cpu().numpy()}")
print(f"  Max amplitude: {np.abs(e_h_raw).max():.4f}")

# ========== 诊断 2: 谐波 + env_net (不走 emformer, 用固定 z) ==========
print("\nDiagnostic 2: Harmonic + env_net (z=random)...")
with torch.no_grad():
    z_test = torch.randn(1, 128, 10, device='cuda') * 0.1
    S_c = model.ddsp_decoder.env_net(z_test.transpose(1, 2)).transpose(1, 2)
    S_lin = model.ddsp_decoder._decompress_envelope(S_c)
    n_samp2 = 10 * 256
    e_h2 = model.ddsp_decoder.harmonic_oscillator(f0_test, n_samp2)
    spec_h = torch.stft(e_h2.squeeze(1), model.ddsp_decoder.fft_size, model.ddsp_decoder.hop_length,
                        window=model.ddsp_decoder.hann_window.to('cuda'), return_complex=True)
    spec_h = spec_h[:, :model.ddsp_decoder.fft_size // 2 + 1, :]
    n_spec = spec_h.shape[-1]
    S_spec_test = F.interpolate(S_lin, size=n_spec, mode='linear')
    spec_filt = spec_h * S_spec_test
    x_h2 = torch.istft(spec_filt, model.ddsp_decoder.fft_size, model.ddsp_decoder.hop_length,
                       window=model.ddsp_decoder.hann_window.to('cuda'), length=n_samp2)

x_h2 = x_h2.squeeze().cpu().numpy()
x_h2 = x_h2 / (np.abs(x_h2).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diag2_harm_env.wav', x_h2, sr)

# 打印 env_net 统计
S_c_np = S_c.squeeze().cpu().numpy()
S_lin_np = S_lin.squeeze().cpu().numpy()
print(f"  S_c (log10 mel, 128 channels): mean={S_c_np.mean():.4f}, std={S_c_np.std():.4f}, min={S_c_np.min():.4f}, max={S_c_np.max():.4f}")
print(f"  S_lin (linear freq, 1025 bins): mean={S_lin_np.mean():.6f}, std={S_lin_np.std():.6f}, min={S_lin_np.min():.6f}, max={S_lin_np.max():.6f}")
print(f"  Are env values too small? median={np.median(S_lin_np):.8f}")

# ========== 诊断 3: 谐波振荡器 + 单位包络 (env=1.0) ==========
print("\nDiagnostic 3: Harmonic + unity envelope...")
with torch.no_grad():
    n_samp3 = 10 * 256
    e_h3 = model.ddsp_decoder.harmonic_oscillator(f0_test, n_samp3)
    spec_h3 = torch.stft(e_h3.squeeze(1), model.ddsp_decoder.fft_size, model.ddsp_decoder.hop_length,
                         window=model.ddsp_decoder.hann_window.to('cuda'), return_complex=True)
    spec_h3 = spec_h3[:, :model.ddsp_decoder.fft_size // 2 + 1, :]
    spec_filt3 = spec_h3 * 1.0  # unity envelope
    x_h3 = torch.istft(spec_filt3, model.ddsp_decoder.fft_size, model.ddsp_decoder.hop_length,
                       window=model.ddsp_decoder.hann_window.to('cuda'), length=n_samp3)
x_h3 = x_h3.squeeze().cpu().numpy()
x_h3 = x_h3 / (np.abs(x_h3).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diag3_unity_env.wav', x_h3, sr)

# ========== 诊断 4: 用模型真实的 z (从 conditioning encoder 来) ==========
print("\nDiagnostic 4: Full model's z + harmonic oscillator (bypass emformer)...")
with torch.no_grad():
    mu = model.cond_encoder(phoneme_t, dur_t, f0_t)
    z_q, _, _ = model.rvq(mu)
    # 直接用 z_q + f0 调谐波振荡器，不经过 emformer
    z_first10 = z_q[:, :, :10]
    S_c4 = model.ddsp_decoder.env_net(z_first10.transpose(1, 2)).transpose(1, 2)
    S_lin4 = model.ddsp_decoder._decompress_envelope(S_c4)
    spec_h4 = spec_h3
    S_spec4 = F.interpolate(S_lin4, size=spec_h4.shape[-1], mode='linear')
    spec_filt4 = spec_h4 * S_spec4
    x_h4 = torch.istft(spec_filt4, model.ddsp_decoder.fft_size, model.ddsp_decoder.hop_length,
                       window=model.ddsp_decoder.hann_window.to('cuda'), length=n_samp3)
x_h4 = x_h4.squeeze().cpu().numpy()
x_h4 = x_h4 / (np.abs(x_h4).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diag4_real_z.wav', x_h4, sr)
S_c4_np = S_c4.squeeze().cpu().numpy()
S_lin4_np = S_lin4.squeeze().cpu().numpy()
print(f"  S_c (real z): mean={S_c4_np.mean():.4f}, std={S_c4_np.std():.4f}")
print(f"  S_lin (real z): mean={S_lin4_np.mean():.6f}, median={np.median(S_lin4_np):.8f}")

# ========== 诊断 5: 检查 z 的值 ==========
print("\nDiagnostic 5: RVQ latent z statistics...")
z_np = z_q.squeeze().cpu().numpy()
print(f"  z shape: {z_np.shape}")
print(f"  z mean: {z_np.mean():.6f}, std: {z_np.std():.6f}")
print(f"  z min: {z_np.min():.6f}, max: {z_np.max():.6f}")
print(f"  z is zero? {(np.abs(z_np) < 1e-6).mean()*100:.1f}% near zero")

# ========== 诊断 6: 完整前向但不做 STFT 后处理, 先看 STFT 频谱 ==========
print("\nDiagnostic 6: Check STFT of harmonic oscillator...")
with torch.no_grad():
    e_h6 = model.ddsp_decoder.harmonic_oscillator(f0_test, n_samp3)
    spec_mag = torch.stft(e_h6.squeeze(1), model.ddsp_decoder.fft_size, model.ddsp_decoder.hop_length,
                          window=model.ddsp_decoder.hann_window.to('cuda'), return_complex=True)
    spec_mag = spec_mag[:, :model.ddsp_decoder.fft_size // 2 + 1, :]
    mag = spec_mag.abs().squeeze().cpu().numpy()
print(f"  STFT magnitude shape: {mag.shape}")
print(f"  First few freqs of first frame: {mag[:20, 0]}")
print(f"  F0={f0_test[0,0,0].item():.1f}Hz -> bin {f0_test[0,0,0].item()*2048/24000:.1f}")
# 检查是否有谐波峰值
print(f"  Max in harmonic region (bin 0-500): {mag[:500, 0].max():.4f}")
print(f"  Mean in harmonic region: {mag[:500, 0].mean():.4f}")
print(f"  Top 10 peak bins & mags in first frame:")
top_idx = np.argsort(mag[:, 0])[-10:][::-1]
for idx in top_idx:
    freq = idx * 24000 / 2048
    print(f"    bin {idx} ({freq:.1f}Hz): {mag[idx, 0]:.4f}")

# ========== 诊断 7: 检查 env_net 偏置是否导致全零 ==========
print("\nDiagnostic 7: Check env_net bias/weights...")
for name, param in model.ddsp_decoder.env_net.named_parameters():
    print(f"  env_net.{name}: shape={param.shape}, mean={param.data.mean():.4f}, std={param.data.std():.4f}")

print("\nAll diagnostics done!")
