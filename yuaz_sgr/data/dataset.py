# YUAZ SGR
"""
data/dataset.py
训练数据集类 — 全量预载入内存, 消除磁盘 I/O

论文 §4.1: 4秒片段, 24kHz, 动态 padding

加速: 构造时将所有 .npz 一次性载入内存 (1.4GB, 服务器内存充裕)
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SingingDataset(Dataset):
    """
    全量预载入: 构造时扫描所有 .npz 并加载到 self.data_list
    避免每次 __getitem__ 的磁盘 I/O 和 np.load 开销
    """
    def __init__(self, data_dir, phoneme_map_path, segment_sec=4,
                 sample_rate=24000, hop_length=256):
        self.segment_sec = segment_sec
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_rate = sample_rate / hop_length
        self.target_samples = segment_sec * sample_rate
        self.target_frames = int(segment_sec * self.frame_rate) + 1  # 固定帧数 = 375

        with open(phoneme_map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        self.p2i = mapping['p2i']
        self.num_phonemes = mapping['num_phonemes']

        data_dir = Path(data_dir)
        files = sorted(data_dir.glob('*.npz'))
        print(f"预载入 {len(files)} 段从 {data_dir} ...", end=' ', flush=True)

        self.data_list = []
        for fpath in files:
            data = np.load(fpath, allow_pickle=True)
            audio = data['audio']
            if audio.shape[-1] < self.target_samples:
                pad = self.target_samples - audio.shape[-1]
                audio = np.pad(audio, (0, pad))
            else:
                audio = audio[:self.target_samples]
            phonemes = data['phonemes'].tolist()
            self.data_list.append({
                'audio': torch.FloatTensor(audio).unsqueeze(0),
                'phoneme_ids': torch.LongTensor([self.p2i.get(p, 0) for p in phonemes]),
                'durations': torch.LongTensor(data['durations'].astype(np.int64)),
                'f0': torch.FloatTensor(self._pad_to_frames(data['f0'])),
                'f0_audio': torch.FloatTensor(self._pad_to_frames(data['f0'])).unsqueeze(0),
                'lags': torch.FloatTensor(self._pad_to_frames(data['lags'])),
                'note_lengths': torch.LongTensor(data['note_lengths'].astype(np.int64)),
            })
        print(f"完成 ({len(self.data_list)} 段)")

    def _pad_to_frames(self, arr):
        n = arr.shape[0]
        if n < self.target_frames:
            return np.pad(arr, (0, self.target_frames - n))
        return arr[:self.target_frames]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    audio = torch.stack([item['audio'] for item in batch])

    # pad_sequence 自动处理变长序列 (phoneme_ids, durations 仍变长)
    phoneme_ids = torch.nn.utils.rnn.pad_sequence(
        [item['phoneme_ids'] for item in batch], batch_first=True
    )
    phoneme_mask = phoneme_ids != 0

    durations = torch.nn.utils.rnn.pad_sequence(
        [item['durations'] for item in batch], batch_first=True
    )

    # f0, lags 已固定长度 (375帧), 直接 stack
    f0 = torch.stack([item['f0'] for item in batch])
    lag_mask = torch.ones_like(f0, dtype=torch.bool)

    lags = torch.stack([item['lags'] for item in batch])

    note_lengths = torch.nn.utils.rnn.pad_sequence(
        [item['note_lengths'] for item in batch], batch_first=True
    )

    # f0_audio
    f0_audio_list = []
    for item in batch:
        if item['f0_audio'] is not None:
            f0_audio_list.append(item['f0_audio'].squeeze(0))
        else:
            f0_audio_list.append(item['f0'])
    f0_audio = torch.stack(f0_audio_list).unsqueeze(1)

    return {
        'audio': audio,
        'phoneme_ids': phoneme_ids,
        'phoneme_mask': phoneme_mask,
        'durations': durations,
        'f0': f0,
        'f0_audio': f0_audio,
        'lags': lags,
        'lag_mask': lag_mask,
        'note_lengths': note_lengths,
    }


def create_dataloaders(preprocessed_dir, phoneme_map_path, batch_size=128,
                       num_workers=16, segment_sec=4, sample_rate=24000,
                       hop_length=256):
    preprocessed_dir = Path(preprocessed_dir)

    loaders = {}
    for split in ['train', 'val', 'test']:
        split_dir = preprocessed_dir / split
        if not split_dir.exists():
            print(f"跳过 {split}: {split_dir} 不存在")
            continue

        dataset = SingingDataset(
            str(split_dir), phoneme_map_path,
            segment_sec=segment_sec, sample_rate=sample_rate,
            hop_length=hop_length,
        )

        shuffle = (split == 'train')
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=True,
            drop_last=(split == 'train'),
        )
        loaders[split] = loader

    return loaders
