"""检查音频帧能量分布，找出合理阈值"""
import numpy as np, glob
files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/train/*.npz"))
f = files[0]
d = np.load(f)
audio = d['audio']
HOP = 256
n_frames = len(audio) // HOP + 1
frame_energy = np.zeros(n_frames)
for t in range(n_frames):
    start = t * HOP
    end = min(start + HOP, len(audio))
    frame_energy[t] = np.sqrt(np.mean(audio[start:end]**2) + 1e-10)
energy_db = 20 * np.log10(frame_energy.clip(1e-10))

old_f0 = d['f0']
print(f"文件: {f.split('/')[-1]}")
print(f"帧能量: min={energy_db.min():.0f}dB max={energy_db.max():.0f}dB mean={energy_db.mean():.0f}dB")
print(f"能量分布: <-50dB: {(energy_db<-50).mean()*100:.0f}%  -50~-40dB: {((energy_db>=-50)&(energy_db<-40)).mean()*100:.0f}%  -40~-30dB: {((energy_db>=-40)&(energy_db<-30)).mean()*100:.0f}%  >-30dB: {(energy_db>=-30).mean()*100:.0f}%")
print(f"老F0非零: {(old_f0>50).mean()*100:.0f}%")

# 测试不同阈值
for th in [-30, -35, -40, -45, -50, -60]:
    voiced = (energy_db > th).mean() * 100
    print(f"  阈值>{th}dB: {voiced:.0f}% 有声音帧")
