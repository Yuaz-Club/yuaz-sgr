"""GT 条件重建：喂原音频的 F0 与真实音素时长，检查模型与真实音频的差距

用法:
    python scripts/syn_gt.py <ckpt> <config_dir> <out_prefix> [样本数]
输出:
    <out_prefix>_syn{i}.wav / <out_prefix>_real{i}.wav
"""
import os
import sys
import argparse
import numpy as np
import torch
import yaml
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.svs_model import SVSModel
from data.dataset import SingingDataset
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('config_dir')
    ap.add_argument('out_prefix')
    ap.add_argument('n', nargs='?', type=int, default=3)
    ap.add_argument('--data_dir', default='../data/preprocessed')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cfg = {}
    for f in ('base.yaml', 'model.yaml', 'train.yaml'):
        cfg.update(yaml.safe_load(open(os.path.join(args.config_dir, f), encoding='utf-8')))

    model = SVSModel(cfg)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('ema') or ck['model']
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in sd.items()}, strict=False)
    model.to(args.device).eval()

    ds = SingingDataset(os.path.join(args.data_dir, 'val'),
                        os.path.join(args.data_dir, 'phoneme_map.json'))
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    out_dir = os.path.dirname(os.path.abspath(args.out_prefix)) or '.'
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.n:
                break
            audio = batch['audio'].to(args.device)
            f0 = batch['f0'].to(args.device).unsqueeze(1)
            ph = batch['phoneme_ids'].to(args.device)
            du = batch['durations'].to(args.device)
            z1, f0e = model.encoder(audio, f0_override=f0)
            T = min(z1.shape[-1], int(du.sum().item()))
            mu = model.cond_encoder(ph, du, f0e[:, :, :T])
            z = model.refiner(mu, steps=int(model.irr_steps))
            zq, _, _ = model.rvq(z)
            wav = model.ddsp_decoder(f0e[:, :, :T], zq,
                                     target_samples=audio.shape[-1], cond=mu)[..., :audio.shape[-1]]
            L = min(audio.shape[-1], wav.shape[-1])
            sf.write(f'{args.out_prefix}_syn{i}.wav', wav[0, 0, :L].cpu().numpy(), 24000)
            sf.write(f'{args.out_prefix}_real{i}.wav', audio[0, 0, :L].cpu().numpy(), 24000)
            print(f'saved {i} ({L / 24000:.2f}s)')


if __name__ == '__main__':
    main()
