"""验证 F0 替换效果: 对一个 npz 跑 CREPE, 检查 shape"""
import numpy as np, torch, torchcrepe, glob

SR, HOP = 24000, 256

files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/train/*.npz"))
f = files[0]
data = np.load(f, allow_pickle=True)
audio = data['audio']
old_f0 = data['f0']

print(f"CREPE output shape: {torchcrepe.predict.__name__}")
audio_t = torch.FloatTensor(audio).unsqueeze(0)
with torch.no_grad():
    f0_real = torchcrepe.predict(audio_t, SR, hop_length=HOP,
                                 fmin=50.0, fmax=2006.0,
                                 model='full', device='cuda', pad=True)
print(f"CREPE output shape: {f0_real.shape}")
f0_real_np = f0_real[0].cpu().numpy().astype(np.float32)
print(f"f0_real_np shape: {f0_real_np.shape}")

# 对齐到老F0长度
T_old = old_f0.shape[0]
T_new = f0_real_np.shape[0]
if T_new >= T_old:
    f0_new = f0_real_np[:T_old]
else:
    f0_new = np.pad(f0_real_np, (0, T_old - T_new))

print(f"老F0: {old_f0.shape}, mean={old_f0.mean():.0f}Hz, 唯一值={len(set(old_f0.astype(int)))}")
print(f"新F0: {f0_new.shape}, mean={f0_new[f0_new>50].mean():.0f}Hz, 唯一值={len(set(f0_new.astype(int)))}")
print(f"老F0[:40]: {old_f0[:40].astype(int)}")
print(f"新F0[:40]: {f0_new[:40].astype(int)}")
diff = np.abs(old_f0 - f0_new)
print(f"\n差异>5Hz: {(diff>5).mean()*100:.1f}%")
