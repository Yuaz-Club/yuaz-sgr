"""诊断 F0 + DDSP decoder 内部"""
import sys, json, yaml, torch, numpy as np

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

# 检查 F0
f0_np = f0_audio.squeeze().numpy()
nonzero_f0 = f0_np[f0_np > 0]
print(f'=== F0 输入 ===')
print(f'F0 stats: min={f0_np.min():.1f} max={f0_np.max():.1f}')
print(f'nonzero mean={nonzero_f0.mean():.1f} std={nonzero_f0.std():.1f}')
print(f'zero ratio: {(f0_np==0).mean():.2%}')
print(f'nonzero range: {np.percentile(nonzero_f0, 5):.0f} ~ {np.percentile(nonzero_f0, 95):.0f} Hz')

# Encoder forward
with torch.no_grad():
    z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
    z_q, indices, _ = model.rvq(z1)

    print(f'\n=== Encoder 输出 ===')
    print(f'z1: shape={list(z1.shape)} mean={z1.mean().item():.4f} std={z1.std().item():.4f}')
    print(f'f0_enc: shape={list(f0_enc.shape)}')
    f0e_np = f0_enc.squeeze().numpy()
    print(f'f0_enc stats: min={f0e_np.min():.1f} max={f0e_np.max():.1f} mean={f0e_np[f0e_np>0].mean():.1f}')
    print(f'z_q: shape={list(z_q.shape)} mean={z_q.mean().item():.4f} std={z_q.std().item():.4f}')

    # DDSP decoder 内部诊断
    d = model.ddsp_decoder
    print(f'\n=== DDSP Decoder ===')
    print(f'n_harmonics={d.n_harmonics}, fft_size={d.fft_size}, hop={d.hop_length}, sr={d.sample_rate}')

    # env_net
    env = d.env_net(z_q)
    print(f'env: shape={list(env.shape)} mean={env.mean().item():.4f} std={env.std().item():.4f}')
    env_np = env.squeeze().numpy()
    # 看看各通道(stddev)有没有差异
    env_ch_std = env_np.std(axis=1)
    print(f'env per-channel std: min={env_ch_std.min():.4f} max={env_ch_std.max():.4f} mean={env_ch_std.mean():.4f}')
    if env_ch_std.max() - env_ch_std.min() < 0.01:
        print('WARNING: env_net output is flat across channels!')

    # 走完整 decoder
    fake_wav = d(f0_enc, z_q, target_samples=target_samples)
    print(f'\nfake_wav: shape={list(fake_wav.shape)} std={fake_wav.std().item():.4f}')

    # 试试不用 decoder，直接看 harmonics 生成
    f0_hz = f0_enc.squeeze(1)  # [B, T_f0]
    print(f'f0_hz: shape={list(f0_hz.shape)}')
    f0_hz_np = f0_hz.squeeze().numpy()
    print(f'f0_hz nonzero stats: mean={f0_hz_np[f0_hz_np>0].mean():.1f}')

    # 尝试用固定 z_q 看看变化
    # 直接调 ddsp_decoder 的 source module
    if hasattr(d, 'source'):
        with torch.no_grad():
            harm_amp = d.source(f0_hz, env)
            print(f'harm_amp: shape={list(harm_amp.shape)} std={harm_amp.std().item():.4f}')

    # 对比：直接用原始 F0 override 和一个假 F0
    f0_fake = torch.full_like(f0_enc, 220.0)  # 固定 A3
    with torch.no_grad():
        fw_fake_f0 = d(f0_fake, z_q, target_samples=target_samples)
    print(f'fake_f0(220Hz) wav std: {fw_fake_f0.std().item():.4f}')
    
    f0_sweep = torch.linspace(100, 400, f0_enc.shape[-1]).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        fw_sweep = d(f0_sweep, z_q, target_samples=target_samples)
    print(f'sweep_f0(100-400Hz) wav std: {fw_sweep.std().item():.4f}')

print(f'\n=== 结论 ===')
print(f'如果 fake_f0 和 sweep_f0 的 std 很小 -> decoder 确实不响应 F0')
print(f'如果 fw_sweep 听起来音高在变 -> decoder 正常，问题在 encoder 的 F0')
