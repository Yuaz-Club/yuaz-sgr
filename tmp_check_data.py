import numpy as np, os, glob
train_dir = '/root/autodl-tmp/yuaz_sgr/data/preprocessed/train'
files = sorted(glob.glob(f'{train_dir}/*.npz'))
print(f'Found {len(files)} npz files')

# Load first few
all_durs = []
all_note = []
for fp in files[:30]:
    d = np.load(fp)
    durs = d['durations']
    nls = d.get('note_lengths', np.array([]))
    all_durs.append(durs[durs > 0])
    all_note.append(nls[nls > 0])

all_durs = np.concatenate(all_durs) if all_durs else np.array([])
all_note = np.concatenate(all_note) if all_note else np.array([])

print(f'\nDuration GT stats ({len(all_durs)} phonemes):')
print(f'  min={all_durs.min():.0f} max={all_durs.max():.0f}')
print(f'  mean={all_durs.mean():.1f} std={all_durs.std():.1f}')
print(f'  P10={np.percentile(all_durs,10):.0f} P50={np.percentile(all_durs,50):.0f} P90={np.percentile(all_durs,90):.0f}')

print(f'\nNote length stats ({len(all_note)} notes):')
print(f'  min={all_note.min():.0f} max={all_note.max():.0f}')
print(f'  mean={all_note.mean():.1f} std={all_note.std():.1f}')

# MSE if we always predict mean
mean_pred = all_durs.mean()
mse_vs_mean = ((all_durs - mean_pred) ** 2).mean()
print(f'\nIf predict always {mean_pred:.1f}: MSE={mse_vs_mean:.1f}')
print(f'(Our dur_loss ~300-1500, so model is barely better than constant prediction)')
