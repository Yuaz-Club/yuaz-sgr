"""诊断: 解码器振幅链追踪"""
import torch, numpy as np, sys, json
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel

config = {}
for name in ['base', 'model']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        import yaml
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json') as f:
    config['conditioning']['num_phonemes'] = json.load(f)['num_phonemes']

device = 'cuda'; sr = 24000
model = SVSModel(config).to(device)
model.eval()
dd = model.ddsp_decoder

from yuaz_sgr.data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json', bs=1, device=device,
)
batch = next(train_iter())
audio = batch['audio']; f0 = batch['f0_audio']
T_audio = audio.shape[-1]

with torch.no_grad(), torch.cuda.amp.autocast():
    z1, f0_e = model.encoder(audio, f0_override=f0)
    z_q, _, _ = model.rvq(z1)
    B, _, T = f0_e.shape

    # Emformer
    feat = torch.cat([z_q, f0_e], dim=1).transpose(1,2)
    feat = dd.emformer_input_proj(feat)
    h, _ = dd.emformer(feat)
    h = dd.emformer_proj(h).transpose(1,2)
    
    S_c = dd.env_net(h.transpose(1,2)).transpose(1,2)
    A_c = dd.ap_net(h.transpose(1,2)).transpose(1,2)
    w = dd.weight_net(h.transpose(1,2)).transpose(1,2)
    lh = torch.clamp(1.0/(w[:,0:1,:]+1e-8), min=0)
    ln = torch.clamp(1.0/(w[:,1:2,:]+1e-8), min=0)
    
    print(f"S_c: min={S_c.min().item():.3f} max={S_c.max().item():.3f} mean={S_c.mean().item():.3f}")
    print(f"A_c: min={A_c.min().item():.3f} max={A_c.max().item():.3f} mean={A_c.mean().item():.3f}")
    print(f"λ_h={lh.mean().item():.3f} λ_n={ln.mean().item():.3f}")
    
    S_exp = torch.pow(10.0, torch.clamp(S_c, -4, 4))
    print(f"10^S_c: min={S_exp.min().item():.4f} max={S_exp.max().item():.4f} mean={S_exp.mean().item():.4f}")
    
    S_lin = torch.matmul(dd.mel_pinv, S_exp)
    print(f"S_lin: min={S_lin.min().item():.4f} max={S_lin.max().item():.4f} mean={S_lin.mean().item():.4f}")
    
    A_lin = A_c.transpose(1,2)
    A_lin = torch.nn.functional.interpolate(A_lin, size=1025, mode='linear').transpose(1,2)
    
    n_samples = T * dd.encoder_hop_length
    e_h = dd.harmonic_oscillator(f0_e, n_samples)
    print(f"e_h: min={e_h.min().item():.3f} max={e_h.max().item():.3f} rms={e_h.pow(2).mean().sqrt().item():.3f}")
    
    spec_h = torch.stft(e_h.squeeze(1), dd.fft_size, dd.hop_length, window=dd.hann_window.to(device), return_complex=True)
    spec_h = spec_h[:, :1025, :]
    nsf = spec_h.shape[-1]
    print(f"spec_h: frames={nsf} mean_abs={spec_h.abs().mean().item():.3f}")
    
    S_spec = torch.nn.functional.interpolate(S_lin, size=nsf, mode='linear')
    A_spec = torch.nn.functional.interpolate(A_lin, size=nsf, mode='linear')
    
    spec_h_filt = spec_h * (1.0 - A_spec) * S_spec
    print(f"spec_h_filt mean_abs={spec_h_filt.abs().mean().item():.4f}")
    
    x_h = torch.istft(spec_h_filt, dd.fft_size, dd.hop_length, window=dd.hann_window.to(device), length=n_samples)
    print(f"x_h: min={x_h.min().item():.3f} max={x_h.max().item():.3f} rms={x_h.pow(2).mean().sqrt().item():.4f}")
    
    e_n = torch.randn(B, n_samples, device=device)
    spec_n = torch.stft(e_n, dd.fft_size, dd.hop_length, window=dd.hann_window.to(device), return_complex=True)
    spec_n = spec_n[:, :1025, :nsf]
    spec_n_filt = spec_n * A_spec * S_spec
    x_n = torch.istft(spec_n_filt, dd.fft_size, dd.hop_length, window=dd.hann_window.to(device), length=n_samples)
    print(f"x_n: rms={x_n.pow(2).mean().sqrt().item():.4f}")
    
    lh_up = torch.nn.functional.interpolate(lh, size=n_samples, mode='linear')
    ln_up = torch.nn.functional.interpolate(ln, size=n_samples, mode='linear')
    wav = lh_up * x_h.unsqueeze(1) + ln_up * x_n.unsqueeze(1)
    print(f"\nwav final: rms={wav.pow(2).mean().sqrt().item():.3f} peak={wav.abs().max().item():.3f}")
    print(f"target RMS={audio.pow(2).mean().sqrt().item():.4f}")
