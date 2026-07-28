"""诊断 v3: 用全曲的真实 F0 前 188 帧 vs 完整全曲，对比定位"""
import sys, json, torch, numpy as np, soundfile as sf
import torch.nn.functional as F
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel
import yaml, mido

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
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_089000.pt', map_location='cuda')
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)

ddsp = model.ddsp_decoder

# 解析 MIDI
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

print(f"Full song: {total_pf} frames, {total_pf*hop/sr:.1f}s")

# 获取真实的 z (从 cond_encoder + RVQ)
with torch.no_grad():
    mu = model.cond_encoder(phoneme_t, dur_t, f0_t)
    z_q, _, _ = model.rvq(mu)
print(f"  z_q shape: {z_q.shape}")
print(f"  mu shape: {mu.shape}")

# ====== Test G: 全曲的前 188 帧 (模拟 diagD 但用真实 F0) ======
print("\n[G] Full DDSP: first 188 frames of REAL song (matching diagD)...")
with torch.no_grad():
    f0_188 = f0_t[:, :, :188]
    z_188 = z_q[:, :, :188]
    w_g = ddsp(f0_188, z_188)
wav_g = w_g.squeeze().cpu().numpy()
wav_g = wav_g / (np.abs(wav_g).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagG_first188.wav', wav_g, sr)
f0_g = f0_frames_full[:188]
print(f"  F0 range: {f0_g.min():.1f} - {f0_g.max():.1f} Hz, unique: {np.unique(f0_g[f0_g>0])}")

# ====== Test H: 全曲完整 DDSP (与 test_normal 相同) ======
print("\n[H] Full DDSP: entire song (same as test_normal)...")
with torch.no_grad():
    w_h = ddsp(f0_t, z_q)
wav_h = w_h.squeeze().cpu().numpy()
wav_h = wav_h / (np.abs(wav_h).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagH_full.wav', wav_h, sr)

# ====== Test I: 全曲纯谐波振荡器 (完全绕过 DDSP, 直接输出 sawtooth) ======
print("\n[I] Pure sawtooth: entire song (NO DDSP at all)...")
n_samp_full = total_pf * hop
with torch.no_grad():
    e_h_full = ddsp.harmonic_oscillator(f0_t, n_samp_full)
wav_i = e_h_full.squeeze().cpu().numpy()
wav_i = wav_i / (np.abs(wav_i).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagI_raw_full.wav', wav_i, sr)
print(f"  length: {len(wav_i)/sr:.1f}s, rms: {np.sqrt(np.mean(wav_i**2)):.4f}")

# ====== Test J: 全曲纯谐波 + STFT/ISTFT 来回 (单位包络) ======
print("\n[J] Sawtooth + STFT roundtrip: entire song...")
with torch.no_grad():
    e_h_j = ddsp.harmonic_oscillator(f0_t, n_samp_full)
    spec_j = torch.stft(e_h_j.squeeze(1), ddsp.fft_size, ddsp.hop_length,
                        window=ddsp.hann_window.to('cuda'), return_complex=True)
    spec_j = spec_j[:, :ddsp.fft_size // 2 + 1, :]
    x_j = torch.istft(spec_j, ddsp.fft_size, ddsp.hop_length,
                      window=ddsp.hann_window.to('cuda'), length=n_samp_full)
wav_j = x_j.squeeze().cpu().numpy()
wav_j = wav_j / (np.abs(wav_j).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagJ_stft_full.wav', wav_j, sr)

# ====== Test K: forced pure harmonic, full song, manual (like diagF but full) ======
print("\n[K] Forced pure harmonic (AP=0 gate=1): entire song...")
orig_fwd = ddsp.forward
def pure_full(f0_in, z_in, target_samples=None, cond=None):
    if f0_in.dim() == 2: f0_in = f0_in.unsqueeze(1)
    B, _, T = f0_in.shape
    feat = torch.cat([z_in, f0_in], dim=1).transpose(1, 2)
    feat = ddsp.emformer_input_proj(feat)
    h, _ = ddsp.emformer(feat)
    h = ddsp.emformer_proj(h).transpose(1, 2)
    S_c_k = ddsp.env_net(h.transpose(1,2)).transpose(1,2)
    S_lin_k = ddsp._decompress_envelope(S_c_k)
    n_s = T * ddsp.encoder_hop_length
    e_h_k = ddsp.harmonic_oscillator(f0_in, n_s)
    spec_k = torch.stft(e_h_k.squeeze(1), ddsp.fft_size, ddsp.hop_length,
                        window=ddsp.hann_window.to(z_in.device), return_complex=True)
    spec_k = spec_k[:, :ddsp.fft_size // 2 + 1, :]
    n_sk = spec_k.shape[-1]
    S_spec_k = F.interpolate(S_lin_k, size=n_sk, mode='linear')
    spec_filt_k = spec_k * S_spec_k
    x_k = torch.istft(spec_filt_k, ddsp.fft_size, ddsp.hop_length,
                      window=ddsp.hann_window.to(z_in.device), length=n_s)
    return x_k.unsqueeze(1)

ddsp.forward = pure_full
with torch.no_grad():
    w_k = ddsp(f0_t, z_q)
wav_k = w_k.squeeze().cpu().numpy()
wav_k = wav_k / (np.abs(wav_k).max() + 1e-8)
sf.write('/root/autodl-tmp/yuaz_sgr/diagK_pure_full.wav', wav_k, sr)
ddsp.forward = orig_fwd

# 检查 emformer 输出在长序列上的统计
print("\n=== Emformer output stats (long vs short) ===")
with torch.no_grad():
    # 短序列 (188 frames)
    z_s = z_q[:, :, :188]; f0_s = f0_t[:, :, :188]
    feat_s = torch.cat([z_s, f0_s], dim=1).transpose(1, 2)
    feat_s = ddsp.emformer_input_proj(feat_s)
    h_s, _ = ddsp.emformer(feat_s)
    h_s = ddsp.emformer_proj(h_s).transpose(1, 2)
    S_c_s = ddsp.env_net(h_s.transpose(1,2)).transpose(1,2)
    S_lin_s = ddsp._decompress_envelope(S_c_s)
    
    # 长序列 (全曲)
    z_l = z_q; f0_l = f0_t
    feat_l = torch.cat([z_l, f0_l], dim=1).transpose(1, 2)
    feat_l = ddsp.emformer_input_proj(feat_l)
    h_l, _ = ddsp.emformer(feat_l)
    h_l = ddsp.emformer_proj(h_l).transpose(1, 2)
    S_c_l = ddsp.env_net(h_l.transpose(1,2)).transpose(1,2)
    S_lin_l = ddsp._decompress_envelope(S_c_l)

S_s = S_lin_s.squeeze().cpu().numpy()
S_l = S_lin_l.squeeze().cpu().numpy()
h_s_np = h_s.squeeze().cpu().numpy()
h_l_np = h_l.squeeze().cpu().numpy()

print(f"  h (short, 188fr): mean={h_s_np.mean():.4f}, std={h_s_np.std():.4f}")
print(f"  h (long, {total_pf}fr): mean={h_l_np.mean():.4f}, std={h_l_np.std():.4f}")
print(f"  S_lin (short): mean={S_s.mean():.4f}, median={np.median(S_s):.4f}")
print(f"  S_lin (long): mean={S_l.mean():.4f}, median={np.median(S_l):.4f}")

# 比较 emformer 输出的变化
print(f"\n  h correlation (first 188 frames vs long): {np.corrcoef(h_s_np[:188].flatten(), h_l_np[:188].flatten())[0,1]:.4f}")

print("\nKey files:")
print("  diagG_first188.wav — 全曲前188帧 DDSP (应与 diagD 一致)")
print("  diagH_full.wav — 全曲完整 DDSP (与 test_normal 相同)")
print("  diagI_raw_full.wav — 全曲纯锯齿波 (基准: 应有清晰音高)")
print("  diagJ_stft_full.wav — 全曲锯齿波+STFT来回")
print("  diagK_pure_full.wav — 全曲强制纯谐波 (AP=0 gate=1)")
