"""F0 响应测试: 同一 z_q 用不同 F0 合成"""
import sys, json, yaml, torch, numpy as np, os
import soundfile as sf

sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel
from data.gpu_dataset import create_gpu_iter

config = {}
for name in ['base', 'model', 'train']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        config.update(yaml.safe_load(f))
config['conditioning']['num_phonemes'] = 60

model = SVSModel(config)
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_038000.pt', map_location='cpu')
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in model_sd and v.shape == model_sd[ck].shape:
        model_sd[ck] = v
model.load_state_dict(model_sd, strict=False)
model.eval()

_, val_iter = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json',
    bs=1, device='cpu'
)
batch = next(val_iter())
audio = batch['audio']
f0_audio = batch['f0_audio']
target_samples = audio.shape[-1]
sr = 24000

with torch.no_grad():
    z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
    z_q, indices, _ = model.rvq(z1)
    d = model.ddsp_decoder

    B, _, T_enc = f0_enc.shape
    print(f'T_enc={T_enc}, z1 shape={list(z1.shape)}, z_q shape={list(z_q.shape)}')

    # 1) 真实 F0
    fw_real = d(f0_enc, z_q, target_samples=target_samples)
    fw_real_np = fw_real.squeeze().numpy()

    # 2) 固定 220Hz
    f0_220 = torch.full((B, 1, T_enc), 220.0)
    fw_220 = d(f0_220, z_q, target_samples=target_samples)
    fw_220_np = fw_220.squeeze().numpy()

    # 3) 递增 100→500Hz
    f0_swp = torch.linspace(100, 500, T_enc).unsqueeze(0).unsqueeze(0)
    fw_swp = d(f0_swp, z_q, target_samples=target_samples)
    fw_swp_np = fw_swp.squeeze().numpy()

    # 4) 真实 F0 的 encoder F0 (检查是不是 f0_override 的问题)
    z1_no_override, f0_enc_raw = model.encoder(audio)  # 不用 override
    fw_raw = d(f0_enc_raw, z_q, target_samples=target_samples)
    fw_raw_np = fw_raw.squeeze().numpy()
    f0r_np = f0_enc_raw.squeeze().numpy()
    print(f'f0_enc_raw: min={f0r_np.min():.1f} max={f0r_np.max():.1f} mean_nonzero={f0r_np[f0r_np>0].mean():.1f}')

    # 5) 直接用 harmonic_oscillator 生成纯谐波 (无 envelope)
    e_h = d.harmonic_oscillator(f0_enc, target_samples)
    eh_np = e_h.squeeze().numpy()

    # 6) 固定 440Hz 纯谐波
    f0_440 = torch.full((B, 1, T_enc), 440.0)
    e_h440 = d.harmonic_oscillator(f0_440, target_samples)
    eh440_np = e_h440.squeeze().numpy()

out_dir = '/root/autodl-tmp/ae_f0_test'
os.makedirs(out_dir, exist_ok=True)

names = {
    'orig': audio.squeeze().numpy(),
    'recon_real_f0': fw_real_np,
    'recon_220hz': fw_220_np,
    'recon_swp_f0': fw_swp_np,
    'recon_raw_f0': fw_raw_np,
    'harmonic_only': eh_np,
    'harmonic_440hz': eh440_np,
}

for name, wav in names.items():
    wav_norm = wav / (np.abs(wav).max() + 1e-8)
    sf.write(f'{out_dir}/{name}.wav', wav_norm, sr)
    print(f'{name}: std={wav.std():.4f} max={np.abs(wav).max():.4f}')

# 分析 std 差异来判断 F0 是否影响输出
print(f'\n=== F0 响应判断 ===')
print(f'real_f0 std: {fw_real_np.std():.4f}')
print(f'220hz std:  {fw_220_np.std():.4f}')
print(f'swp_f0 std: {fw_swp_np.std():.4f}')
std_ratio = max(fw_220_np.std(), fw_swp_np.std()) / max(fw_real_np.std(), 1e-8)
print(f'ratio: {std_ratio:.4f} (越接近1说明F0对输出影响越小)')

# 看看 decoder 内部的 S_c, A_c 值
with torch.no_grad():
    feat = torch.cat([z_q, f0_enc], dim=1).transpose(1, 2)
    feat = d.emformer_input_proj(feat)
    h, _ = d.emformer(feat)
    h = d.emformer_proj(h).transpose(1, 2)
    S_c = d.env_net(h.transpose(1, 2)).transpose(1, 2)
    A_c = d.ap_net(h.transpose(1, 2)).transpose(1, 2)
    w = d.weight_net(h.transpose(1, 2)).transpose(1, 2)
    lambda_h = 1.0 / (w[:, 0:1, :] + 1e-8)
    lambda_n = 1.0 / (w[:, 1:2, :] + 1e-8)
    print(f'S_c: mean={S_c.mean().item():.4f} std={S_c.std().item():.4f} min={S_c.min().item():.4f} max={S_c.max().item():.4f}')
    print(f'A_c: mean={A_c.mean().item():.4f} std={A_c.std().item():.4f} min={A_c.min().item():.4f} max={A_c.max().item():.4f}')
    print(f'lambda_h: mean={lambda_h.mean().item():.4f} lambda_n: mean={lambda_n.mean().item():.4f}')

print(f'\nSaved to {out_dir}/')
