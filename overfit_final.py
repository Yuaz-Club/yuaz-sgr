"""单样本过拟合: 2000步 + AE重建 + 推理对比"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel
from losses.stft_loss import MultiResolutionSTFTLoss

# config
config = {}
for name in ['base', 'model']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        import yaml
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']

device = 'cuda'; sr = 24000; N = 2000
OUT = '/root/autodl-tmp/overfit_final'

# ── 数据 ──
from yuaz_sgr.data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json',
    bs=1, device=device,
)
batch = next(train_iter())
audio = batch['audio']; ph = batch['phoneme_ids']; dur = batch['durations']
f0_gt = batch['f0_audio']; nl = batch['note_lengths']
T_audio = audio.shape[-1]

os.makedirs(OUT, exist_ok=True)
sf.write(f'{OUT}/orig.wav', audio.squeeze().cpu().numpy() / (audio.abs().max()+1e-8).item(), sr)
print(f"audio={list(audio.shape)} dur={list(dur.shape)}")

# ── 模型 ──
model = SVSModel(config).to(device)
model.freeze_f0_encoder()
model.train()
print(f"params: {sum(p.numel() for p in model.generator_params())/1e6:.1f}M")

opt = torch.optim.Adam(model.generator_params(), lr=2e-4, betas=(0.8, 0.999))
scl = torch.cuda.amp.GradScaler()
stft_fn = MultiResolutionSTFTLoss()

print(f"\n过拟合 {N} steps\n")
t0 = time.time()
for s in range(N):
    with torch.cuda.amp.autocast():
        # AE
        z1, f0_e = model.encoder(audio, f0_override=f0_gt)
        z_q, _, rvq = model.rvq(z1)
        fw = model.ddsp_decoder(f0_e, z_q, target_samples=T_audio)
        stft = stft_fn(audio, fw)
        loss = stft + 0.5 * rvq

        # Flow
        mu = model.cond_encoder(ph, dur, f0_e.squeeze(1))
        T = min(z1.shape[-1], mu.shape[-1])
        loss = loss + 0.1 * F.mse_loss(mu[:,:,:T], z1[:,:,:T].detach())
        z0 = torch.randn_like(z1)
        t = torch.rand(1, device=device)
        z_t = (1-t)*z0[:,:,:T] + t*z1[:,:,:T].detach()
        go = model.flow_gen(z_t, t.unsqueeze(0), mu[:,:,:T])
        vp = go[:,:128,:]
        vt = z1[:,:,:T].detach() - z0[:,:,:T]
        cfm = (1.0/(1.0-t+1e-6)*((vp-vt)**2)).mean()
        loss = loss + cfm

    opt.zero_grad()
    scl.scale(loss).backward()
    scl.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.generator_params(), 5.0)
    scl.step(opt)
    scl.update()

    if s % 200 == 0 or s == N-1:
        print(f"step {s:04d} loss={loss.item():.3f} stft={stft.item():.2f} cfm={cfm.item():.3f} rvq={rvq.item():.4f} [{time.time()-t0:.0f}s]")

# ── AE 重建 ──
model.eval()
with torch.no_grad(), torch.cuda.amp.autocast():
    z1, f0_e = model.encoder(audio, f0_override=f0_gt)
    z_q, _, _ = model.rvq(z1)
    ae = model.ddsp_decoder(f0_e, z_q, target_samples=T_audio)
ae = ae.squeeze().cpu().numpy(); ae = ae/(np.abs(ae).max()+1e-8)
sf.write(f'{OUT}/ae.wav', ae, sr)
print(f"\nAE: {len(ae)/sr:.1f}s")

# ── 推理: 平滑F0 ──
f0r = f0_gt.squeeze().cpu().numpy()
f0s = gaussian_filter1d(f0r.astype(np.float64), sigma=6.0)
nl_c = nl.squeeze().cpu().numpy().astype(int)
f0_step = np.zeros_like(f0s); pos = 0
for n in nl_c:
    if n<=0: continue
    e = min(pos+n, len(f0_step))
    if e>pos: f0_step[pos:e] = np.mean(f0s[pos:e])
    pos = e
f0_t = torch.FloatTensor(f0_step[:len(f0r)]).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    w8, _, _, _, _, _ = model.forward_synthesis(ph, dur, f0_t, n_steps=8, note_lengths=nl)
    w1, _, _, _, _, _ = model.forward_synthesis(ph, dur, f0_t, n_steps=1, note_lengths=nl)

for name, w in [('infer_8', w8), ('infer_1', w1)]:
    w = w.squeeze().cpu().numpy(); w = w/(np.abs(w).max()+1e-8)
    sf.write(f'{OUT}/{name}.wav', w, sr)
    print(f"{name}: {len(w)/sr:.1f}s")

print(f"\n完成: {OUT}/orig.wav ae.wav infer_8.wav infer_1.wav")
