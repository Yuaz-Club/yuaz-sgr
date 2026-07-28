import json, torch, numpy as np
from pathlib import Path

class GPUMemDataset:
    """数据预加载到 CPU 内存, 采样时移到 GPU(节省显存)"""
    def __init__(self, data_dir, phoneme_map_path, device='cuda',
                 segment_sec=4, sample_rate=24000, hop_length=256):
        self.device = device
        self.target_samples = segment_sec * sample_rate
        self.target_frames = int(segment_sec * sample_rate / hop_length) + 1

        with open(phoneme_map_path) as f:
            self.p2i = json.load(f)['p2i']

        data_dir = Path(data_dir)
        files = sorted(data_dir.glob('*.npz'))
        print(f"预加载 {len(files)} 段 ...", end=' ', flush=True)

        self.audio_l, self.f0_l, self.lags_l = [], [], []
        self.ph_ids_l, self.dur_l, self.nl_l = [], [], []

        for fpath in files:
            data = np.load(fpath, allow_pickle=True)
            a = data['audio']
            if a.shape[-1] < self.target_samples:
                a = np.pad(a, (0, self.target_samples - a.shape[-1]))
            self.audio_l.append(torch.from_numpy(a[:self.target_samples]).float())

            f0 = data['f0']
            if len(f0) < self.target_frames:
                f0 = np.pad(f0, (0, self.target_frames - len(f0)))
            self.f0_l.append(torch.from_numpy(f0[:self.target_frames]).float())

            lag = data['lags']
            if len(lag) < self.target_frames:
                lag = np.pad(lag, (0, self.target_frames - len(lag)))
            self.lags_l.append(torch.from_numpy(lag[:self.target_frames]).float())

            ph = data['phonemes'].tolist()
            self.ph_ids_l.append(torch.LongTensor([self.p2i.get(p, 0) for p in ph]))
            self.dur_l.append(torch.LongTensor(data['durations'].astype(np.int64)))
            self.nl_l.append(torch.LongTensor(data['note_lengths'].astype(np.int64)))

        self.n = len(files)
        print(f"完成 ({self.n} 段)")

    def sample_batch_indices(self, indices):
        """用指定索引采样 batch, 移到 GPU"""
        dev = self.device
        idx = torch.tensor(indices)
        audio = torch.stack([self.audio_l[i].to(dev) for i in idx]).unsqueeze(1)
        f0 = torch.stack([self.f0_l[i].to(dev) for i in idx])
        f0_audio = f0.unsqueeze(1)
        lags = torch.stack([self.lags_l[i].to(dev) for i in idx])
        lag_mask = torch.ones_like(f0, dtype=torch.bool, device=dev)
        ph_ids = torch.nn.utils.rnn.pad_sequence(
            [self.ph_ids_l[i].to(dev) for i in idx], batch_first=True)
        phoneme_mask = ph_ids != 0
        dur = torch.nn.utils.rnn.pad_sequence(
            [self.dur_l[i].to(dev) for i in idx], batch_first=True)
        nl = torch.nn.utils.rnn.pad_sequence(
            [self.nl_l[i].to(dev) for i in idx], batch_first=True)
        return {'audio': audio, 'phoneme_ids': ph_ids, 'phoneme_mask': phoneme_mask,
                'durations': dur, 'f0': f0, 'f0_audio': f0_audio, 'lags': lags,
                'lag_mask': lag_mask, 'note_lengths': nl}

def _make_gen(ds, bs):
    """创建可调用生成器, 挂载 n 属性便于外部获取数据集大小"""
    n = ds.n
    def gen():
        idxs = list(range(n))
        while True:
            np.random.shuffle(idxs)
            for s in range(0, n, bs):
                yield ds.sample_batch_indices(idxs[s:s+bs])
    gen.n = n  # 供 train.py 读取
    return gen

def create_gpu_iter(data_dir, phoneme_map_path, bs=256, device='cuda'):
    ds_train = GPUMemDataset(Path(data_dir)/'train', phoneme_map_path, device=device)
    ds_val   = GPUMemDataset(Path(data_dir)/'val',   phoneme_map_path, device=device)
    return _make_gen(ds_train, bs), _make_gen(ds_val, bs)
