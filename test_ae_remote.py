"""AE 重建测试 — 用最新 checkpoint 跑一条验证数据"""
import os, sys, json, yaml, torch, numpy as np
import soundfile as sf

sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel
from data.gpu_dataset import create_gpu_iter
from losses.stft_loss import MultiResolutionSTFTLoss

# 配置
config = {}
for name in ['base', 'model', 'train']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']

# 加载 checkpoint
ckpt_path = '/root/autodl-tmp/yuaz_sgr/checkpoints/model_038000.pt'
device = 'cpu'

model = SVSModel(config)
ckpt = torch.load(ckpt_path, map_location=device)
model_sd = model.state_dict()
loaded = skipped = 0
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in model_sd and v.shape == model_sd[ck].shape:
        model_sd[ck] = v
        loaded += 1
    else:
        skipped += 1
model.load_state_dict(model_sd, strict=False)
model.eval()
print(f'checkpoint: step={ckpt["step"]}, loaded={loaded}, skipped={skipped}')

# 加载一条验证数据
_, val_iter = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json',
    bs=1, device=device
)
batch = next(val_iter())
audio = batch['audio']
f0_audio = batch['f0_audio']
target_samples = audio.shape[-1]
sr = 24000

ph_ids = batch['phoneme_ids']
durs = batch['durations']
# phoneme 解码
i2p = pm.get('i2p', {})
ph_list = []
for i in ph_ids.squeeze().tolist():
    idx = int(i)
    if idx > 0 and str(idx) in i2p:
        ph_list.append(i2p[str(idx)])
ph_str = ''.join(ph_list)
print(f'audio: {list(audio.shape)} ({target_samples/sr:.1f}s)')
if ph_str:
    print(f'phonemes: {ph_str[:80]}...')

# AE 重建
with torch.no_grad():
    z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
    z_q, indices, _ = model.rvq(z1)
    fake_wav = model.ddsp_decoder(f0_enc, z_q, target_samples=target_samples)

orig = audio.squeeze().numpy()
recon = fake_wav.squeeze().numpy()

# 归一化 & 保存
orig_norm = orig / (np.abs(orig).max() + 1e-8)
recon_norm = recon / (np.abs(recon).max() + 1e-8)

out_dir = '/root/autodl-tmp/ae_test_038k'
os.makedirs(out_dir, exist_ok=True)
sf.write(f'{out_dir}/orig.wav', orig_norm, sr)
sf.write(f'{out_dir}/recon.wav', recon_norm, sr)

# 指标
stft_fn = MultiResolutionSTFTLoss()
stft_val = stft_fn(audio, fake_wav).item()

# SNR
min_len = min(len(orig), len(recon))
signal_power = (orig[:min_len] ** 2).mean()
noise_power = ((orig[:min_len] - recon[:min_len]) ** 2).mean()
snr = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else float('inf')

# RVQ 使用率
total_codes = config['rvq']['codebook_size']
unique_codes = len(set(indices.numpy().flatten().tolist()))
usage = unique_codes / total_codes * 100

print(f'\n=== AE 重建指标 ===')
print(f'STFT loss: {stft_val:.4f}')
print(f'SNR:       {snr:.1f} dB')
print(f'RVQ usage: {unique_codes}/{total_codes} ({usage:.1f}%)')
print(f'Saved:     {out_dir}/orig.wav, recon.wav')
