"""诊断 v2: 更长片段 + 交替音高, 逐层剥离找音高消失点"""
import sys, json, torch, numpy as np, soundfile as sf
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

ddsp = model.ddsp_decoder
hop = config.get('hop_length', 256)

# 构造测试 F0: 用 4 个不同音高交替, 每段 ~0.5 秒 (47帧)
# 415Hz (G#4), 330Hz (E4), 262Hz (C4), 523Hz (C5)
f0_vals = [415.3, 330.0, 262.0, 523.3]
seg_frames = 47  # ~0.5s
total_frames = len(f0_vals) * seg_frames  # 188 frames ≈ 2s

f0_np = np.zeros(total_frames, dtype=np.float32)
for i, fv in enumerate(f0_vals):
    f0_np[i*seg_frames:(i+1)*seg_frames] = fv

f0_t = torch.FloatTensor(f0_np).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, 188)
n_samples = total_frames * hop  # 48128 samples

print(f"Test: {total_frames} frames, {n_samples} samples ({n_samples/sr:.1f}s)")
print(f"F0 pattern: {f0_vals}")

# ====== Test A: 纯谐波振荡器 (绕过 STFT, 直接输出样本) ======
print("\n[A] Raw harmonic oscillator (no STFT, no env_net)...")
with torch.no_grad():
    e_h = ddsp.harmonic_oscillator(f0_t, n_samples)
wav_a = e_h.squeeze().cpu().numpy()
wav_a = wav_a / (np.abs(wav_a).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagA_raw_osc.wav', wav_a, sr)
print(f"  max={np.abs(wav_a).max():.4f}, rms={np.sqrt(np.mean(wav_a**2)):.4f}")

# ====== Test B: 纯谐波 + STFT/ISTFT 来回 (单位包络, 无衰减) ======
print("\n[B] Harmonic + STFT roundtrip (unity envelope)...")
with torch.no_grad():
    e_h2 = ddsp.harmonic_oscillator(f0_t, n_samples)
    spec = torch.stft(e_h2.squeeze(1), ddsp.fft_size, ddsp.hop_length,
                      window=ddsp.hann_window.to('cuda'), return_complex=True)
    spec = spec[:, :ddsp.fft_size // 2 + 1, :]
    x = torch.istft(spec, ddsp.fft_size, ddsp.hop_length,
                    window=ddsp.hann_window.to('cuda'), length=n_samples)
wav_b = x.squeeze().cpu().numpy()
wav_b = wav_b / (np.abs(wav_b).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagB_stft_rnd.wav', wav_b, sr)
print(f"  max={np.abs(wav_b).max():.4f}, rms={np.sqrt(np.mean(wav_b**2)):.4f}")

# ====== Test C: DDSP decoder 直接调用 (z=全零) ======
print("\n[C] DDSP decoder with z=zeros...")
with torch.no_grad():
    z_zeros = torch.zeros(1, 128, total_frames, device='cuda')
    w = ddsp(f0_t, z_zeros)
wav_c = w.squeeze().cpu().numpy()
wav_c = wav_c / (np.abs(wav_c).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagC_zeros_z.wav', wav_c, sr)
# 检查输出
with torch.no_grad():
    feat = torch.cat([z_zeros, f0_t], dim=1).transpose(1, 2)
    feat = ddsp.emformer_input_proj(feat)
    h, _ = ddsp.emformer(feat)
    h = ddsp.emformer_proj(h).transpose(1, 2)
    S_c_z = ddsp.env_net(h.transpose(1,2)).transpose(1,2)
    S_lin_z = ddsp._decompress_envelope(S_c_z)
    A_c_z = ddsp.ap_net(h.transpose(1,2)).transpose(1,2)
    gate_z = ddsp.weight_net(h.transpose(1,2)).transpose(1,2)
S_z = S_lin_z.squeeze().cpu().numpy()
A_z = A_c_z.squeeze().cpu().numpy()
g_z = gate_z.squeeze().cpu().numpy()
print(f"  S_lin: mean={S_z.mean():.4f}, median={np.median(S_z):.4f}, min={S_z.min():.4f}, max={S_z.max():.4f}")
print(f"  AP(A): mean={A_z.mean():.4f}, std={A_z.std():.4f}")
print(f"  gate(λ_h): mean={g_z.mean():.4f}")

# ====== Test D: 从 cond_encoder 取 z, 但只用前几帧做 DDSP ======
print("\n[D] DDSP with real z (first 188 frames from cond_encoder)...")
# 用完整的 midi 数据
import mido
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
f0_full_t = torch.FloatTensor(f0_frames_full).unsqueeze(0).unsqueeze(0).cuda()
nl_t = torch.LongTensor([seg_frames_list[i] for i, (typ, _, _, _) in enumerate(segments) if typ == 'note']).unsqueeze(0).cuda()

with torch.no_grad():
    mu = model.cond_encoder(phoneme_t, dur_t, f0_full_t)
    z_q, _, _ = model.rvq(mu)
    # 取前 188 帧
    z_188 = z_q[:, :, :total_frames]
    w_d = ddsp(f0_t, z_188)
wav_d = w_d.squeeze().cpu().numpy()
wav_d = wav_d / (np.abs(wav_d).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagD_real_z.wav', wav_d, sr)

# 检查 z 的统计
z_np = z_188.squeeze().cpu().numpy()
print(f"  z mean={z_np.mean():.4f}, std={z_np.std():.4f}, active_chans={(np.abs(z_np).mean(axis=0)>0.01).sum()}/128")
mu_np = mu[:, :, :total_frames].squeeze().cpu().numpy()
print(f"  mu mean={mu_np.mean():.4f}, std={mu_np.std():.4f}")

# ====== Test E: 谐波振荡器 + STFT + env_net(z_zeros) = 绕过 emformer ======
print("\n[E] Harmonic + env_net(z=zeros) bypass emformer...")
with torch.no_grad():
    z_z = torch.zeros(1, 128, total_frames, device='cuda')
    S_c_e = ddsp.env_net(z_z.transpose(1,2)).transpose(1,2)
    S_lin_e = ddsp._decompress_envelope(S_c_e)
    e_h_e = ddsp.harmonic_oscillator(f0_t, n_samples)
    spec_e = torch.stft(e_h_e.squeeze(1), ddsp.fft_size, ddsp.hop_length,
                        window=ddsp.hann_window.to('cuda'), return_complex=True)
    spec_e = spec_e[:, :ddsp.fft_size // 2 + 1, :]
    n_spec_e = spec_e.shape[-1]
    S_spec_e = F.interpolate(S_lin_e, size=n_spec_e, mode='linear')
    spec_filt_e = spec_e * S_spec_e
    x_e = torch.istft(spec_filt_e, ddsp.fft_size, ddsp.hop_length,
                      window=ddsp.hann_window.to('cuda'), length=n_samples)
wav_e = x_e.squeeze().cpu().numpy()
wav_e = wav_e / (np.abs(wav_e).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagE_zeroz_env.wav', wav_e, sr)

# ====== Test F: DDSP decoder 完整前向但 gate=1, AP=0 (纯谐波) ======
print("\n[F] DDSP with gate=1 AP=0 (pure harmonic forced)...")
orig_fwd = ddsp.forward
def pure_ddsp(f0_in, z_in, target_samples=None, cond=None):
    if f0_in.dim() == 2: f0_in = f0_in.unsqueeze(1)
    B, _, T = f0_in.shape
    feat = torch.cat([z_in, f0_in], dim=1).transpose(1, 2)
    feat = ddsp.emformer_input_proj(feat)
    h, _ = ddsp.emformer(feat)
    h = ddsp.emformer_proj(h).transpose(1, 2)
    S_c_f = ddsp.env_net(h.transpose(1,2)).transpose(1,2)
    S_lin_f = ddsp._decompress_envelope(S_c_f)
    n_s = T * ddsp.encoder_hop_length
    e_h_f = ddsp.harmonic_oscillator(f0_in, n_s)
    spec_f = torch.stft(e_h_f.squeeze(1), ddsp.fft_size, ddsp.hop_length,
                        window=ddsp.hann_window.to(z_in.device), return_complex=True)
    spec_f = spec_f[:, :ddsp.fft_size // 2 + 1, :]
    n_sf = spec_f.shape[-1]
    S_spec_f = F.interpolate(S_lin_f, size=n_sf, mode='linear')
    spec_filt_f = spec_f * S_spec_f  # no AP, no noise
    x_f = torch.istft(spec_filt_f, ddsp.fft_size, ddsp.hop_length,
                      window=ddsp.hann_window.to(z_in.device), length=n_s)
    return x_f.unsqueeze(1)

ddsp.forward = pure_ddsp
with torch.no_grad():
    w_f = ddsp(f0_t, z_188)
wav_f = w_f.squeeze().cpu().numpy()
wav_f = wav_f / (np.abs(wav_f).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagF_pure_emf.wav', wav_f, sr)
ddsp.forward = orig_fwd

# 打印 env_net 在真实 z + emformer 下的输出
with torch.no_grad():
    feat_test = torch.cat([z_188, f0_t], dim=1).transpose(1, 2)
    feat_test = ddsp.emformer_input_proj(feat_test)
    h_test, _ = ddsp.emformer(feat_test)
    h_test = ddsp.emformer_proj(h_test).transpose(1, 2)
    S_c_test = ddsp.env_net(h_test.transpose(1,2)).transpose(1,2)
    S_lin_test = ddsp._decompress_envelope(S_c_test)
S_lt = S_lin_test.squeeze().cpu().numpy()
print(f"  S_lin (z+emformer): mean={S_lt.mean():.4f}, median={np.median(S_lt):.4f}, min={S_lt.min():.6f}, max={S_lt.max():.4f}")

print("\nAll done! Key files:")
print("  diagA_raw_osc.wav — pure oscillator, no processing")
print("  diagB_stft_rnd.wav — STFT roundtrip only")
print("  diagC_zeros_z.wav — full DDSP, z=zeros")
print("  diagD_real_z.wav — full DDSP, real z")
print("  diagE_zeroz_env.wav — harmonic+env(z=zeros), no emformer")
print("  diagF_pure_emf.wav — forced gate=1 AP=0, with emformer+real z")
