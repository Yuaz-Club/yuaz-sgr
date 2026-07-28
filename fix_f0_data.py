"""
用 CREPE 提取真实 F0 替换 npz 里的平坦 MIDI F0
对静音帧将 F0 置 0，避免噪声误报
"""
import numpy as np, torch, torchcrepe, glob, os, warnings
from pathlib import Path

SR, HOP = 24000, 256

def replace_f0_in_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    audio = data['audio']
    old_f0 = data['f0']
    T_old = old_f0.shape[0]
    n_samples = len(audio)
    n_frames = n_samples // HOP + 1

    # 逐帧 RMS 能量掩码: 低于 -40dB 的帧认为无声音
    frame_energy = np.zeros(n_frames)
    for t in range(min(n_frames, T_old)):
        start = t * HOP
        end = min(start + HOP, n_samples)
        frame_energy[t] = np.sqrt(np.mean(audio[start:end]**2) + 1e-10)
    energy_db = 20 * np.log10(frame_energy.clip(1e-10))
    voiced_mask = energy_db > -40  # -40dB 以下视为静音

    # CREPE 提取
    audio_t = torch.FloatTensor(audio).unsqueeze(0)
    with torch.no_grad():
        f0_real = torchcrepe.predict(audio_t, SR, hop_length=HOP,
                                     fmin=50.0, fmax=2006.0,
                                     model='full', device='cuda', pad=True)
    f0_real_np = f0_real[0].cpu().numpy().astype(np.float32)

    # 对齐长度
    T_new = f0_real_np.shape[0]
    n_frames_use = min(T_old, T_new)
    f0_new = np.zeros(T_old, dtype=np.float32)
    
    # 仅保留有声音帧，其余置 0
    for t in range(n_frames_use):
        if t < len(voiced_mask) and voiced_mask[t]:
            f0_new[t] = f0_real_np[t]
    
    # 修改 f0 字段
    f0_new = f0_new.astype(np.float32)
    
    # 保存: 用 tempfile 确保原子写入
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix='.npz', dir=os.path.dirname(str(npz_path)))
    os.close(fd)
    np.savez(tmp_path,
             audio=data['audio'],
             phonemes=data['phonemes'],
             durations=data['durations'],
             f0=f0_new,
             lags=data['lags'],
             note_lengths=data['note_lengths'],
             song_id=data.get('song_id', ''))
    os.replace(tmp_path, str(npz_path))
    
    old_unique = len(set(old_f0.astype(int)))
    new_unique = len(set(f0_new[f0_new > 50].astype(int)))
    return old_unique, new_unique, old_f0.mean(), f0_new[f0_new > 50].mean() if (f0_new > 50).any() else 0


if __name__ == '__main__':
    base = Path("/root/autodl-tmp/yuaz_sgr/data/preprocessed")
    total_n = 0
    old_avg_unique = 0
    new_avg_unique = 0
    
    for split in ['train', 'val', 'test']:
        files = sorted((base / split).glob('*.npz'))
        print(f"[{split}] {len(files)} 个文件...", flush=True)
        for fpath in files:
            old_u, new_u, old_m, new_m = replace_f0_in_npz(str(fpath))
            old_avg_unique += old_u
            new_avg_unique += new_u
            total_n += 1
            if total_n % 200 == 0:
                print(f"  进度: {total_n} (of ~2200)", flush=True)
    
    print(f"\n完成! 处理 {total_n} 个文件")
    print(f"平均每段: 老F0唯一个={old_avg_unique/total_n:.1f}, 新F0唯一个={new_avg_unique/total_n:.1f}")
