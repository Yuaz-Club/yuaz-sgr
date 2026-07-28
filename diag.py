"""诊断: AE 重建输出值检查"""
import torch, numpy as np, soundfile as sf, sys, json
from pathlib import Path
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

# 加载一条数据
from yuaz_sgr.data.gpu_dataset import create_gpu_iter
train_iter, _ = create_gpu_iter(
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset',
    '/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json',
    bs=1, device=device,
)
batch = next(train_iter())
audio = batch['audio']; f0 = batch['f0_audio']
T = audio.shape[-1]
print(f"audio: min={audio.min().item():.4f} max={audio.max().item():.4f} rms={audio.pow(2).mean().sqrt().item():.4f}")
print(f"f0: min={f0.min().item():.1f} max={f0.max().item():.1f} mean={f0[f0>1].mean().item():.1f}")

# 原始(未训练) encoder + decoder
with torch.no_grad(), torch.cuda.amp.autocast():
    z1, f0_e = model.encoder(audio, f0_override=f0)
    print(f"\nz1: shape={list(z1.shape)} min={z1.min().item():.4f} max={z1.max().item():.4f} std={z1.std().item():.4f}")
    
    z_q, idx, rvq_loss = model.rvq(z1)
    print(f"z_q: shape={list(z_q.shape)} min={z_q.min().item():.4f} max={z_q.max().item():.4f} std={z_q.std().item():.4f}")
    print(f"rvq idx unique: {[len(torch.unique(idx[:,i])) for i in range(min(3, idx.shape[1]))]}")
    
    wav = model.ddsp_decoder(f0_e, z_q, target_samples=T)
    print(f"\nwav: shape={list(wav.shape)} min={wav.min().item():.4f} max={wav.max().item():.4f} rms={wav.pow(2).mean().sqrt().item():.4f}")
    print(f"wav has_nan={torch.isnan(wav).any().item()} has_inf={torch.isinf(wav).any().item()}")

# 保存
wav_np = wav.squeeze().cpu().float().numpy()
orig_np = audio.squeeze().cpu().numpy()
print(f"\n原始 RMS={np.sqrt(np.mean(orig_np**2)):.4f}  peak={np.abs(orig_np).max():.4f}")
print(f"生成 RMS={np.sqrt(np.mean(wav_np**2)):.4f}  peak={np.abs(wav_np).max():.4f}")

# 不过归一化直接保存
sf.write('/root/autodl-tmp/diag_raw.wav', wav_np, sr)
wav_norm = wav_np / (np.abs(wav_np).max() + 1e-8)
sf.write('/root/autodl-tmp/diag_norm.wav', wav_norm, sr)
sf.write('/root/autodl-tmp/diag_orig.wav', orig_np / (np.abs(orig_np).max()+1e-8), sr)
print("保存完成: diag_raw.wav diag_norm.wav diag_orig.wav")
