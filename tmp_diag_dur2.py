import sys, json, torch, numpy as np
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
from models.svs_model import SVSModel
import yaml

config = {}
for name in ['base', 'model']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
        config.update(yaml.safe_load(f))
with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json') as f:
    pm = json.load(f)
config['conditioning']['num_phonemes'] = pm['num_phonemes']

model = SVSModel(config).to('cuda')
model.eval()
ckpt = torch.load('/root/autodl-tmp/yuaz_sgr/checkpoints/model_081000.pt', map_location='cuda')
msd = model.state_dict()
for k, v in ckpt['model'].items():
    ck = k.replace('_orig_mod.', '')
    if ck in msd and v.shape == msd[ck].shape:
        msd[ck] = v
model.load_state_dict(msd, strict=False)

# Load a few real training batches
import pickle, glob
data_dir = '/root/autodl-tmp/yuaz_sgr/data/preprocessed'
files = sorted(glob.glob(f'{data_dir}/*.pt'))[:20]

all_dur_gt = []
all_dur_pred = []
all_note_len = []
all_ph_types = []
VOWELS = {'ue','ui','ai','e','ei','ian','eng','v','un','iang','ing',
          'ong','an','i','ao','en','a','o','er','u','ang','ou'}

for fpath in files[:5]:
    try:
        data = torch.load(fpath, map_location='cpu', weights_only=False)
    except:
        continue
    # This is a preprocessed segment
    if isinstance(data, dict) and 'phoneme_ids' in data and 'durations' in data:
        ph_ids = data['phoneme_ids'].unsqueeze(0).cuda()
        durs = data['durations'].unsqueeze(0).cuda()
        f0 = data.get('f0', data.get('f0_audio', None))
        if f0 is None:
            continue
        f0 = f0.unsqueeze(0).unsqueeze(0).cuda() if f0.dim() == 2 else f0.unsqueeze(0).cuda()
        nl = data.get('note_lengths', None)
        if nl is None:
            continue
        nl = nl.unsqueeze(0).cuda()

        with torch.no_grad():
            mu = model.cond_encoder(ph_ids, durs, f0.squeeze(1))
            f0_1d = f0.squeeze(1)
            _, dur_pred = model.temporal_predictor(mu, f0_1d, durs, nl, phoneme_ids=ph_ids)

        # Collect per-phoneme data
        for b in range(durs.shape[0]):
            for p in range(durs.shape[1]):
                d = durs[b, p].item()
                if d > 0:
                    all_dur_gt.append(d)
                    all_dur_pred.append(dur_pred[b, p].item())
                    # Find parent note
                    ptype = 'V' if pm['i2p'].get(str(ph_ids[b,p].item()), 'SP') in VOWELS else 'C'
                    all_ph_types.append(ptype)

print(f'Samples: {len(all_dur_gt)}')
print(f'GT  dur: mean={np.mean(all_dur_gt):.1f} std={np.std(all_dur_gt):.1f} min={min(all_dur_gt):.1f} max={max(all_dur_gt):.1f}')
print(f'Pred dur: mean={np.mean(all_dur_pred):.1f} std={np.std(all_dur_pred):.1f} min={min(all_dur_pred):.1f} max={max(all_dur_pred):.1f}')

# Correlation
gt = np.array(all_dur_gt)
pred = np.array(all_dur_pred)
corr = np.corrcoef(gt, pred)[0,1]
print(f'Correlation: {corr:.4f}')

# Per-type
for pt in ['C', 'V']:
    mask = [t == pt for t in all_ph_types]
    g = gt[mask]; p = pred[mask]
    print(f'{pt}: n={len(g)} GT_mean={g.mean():.1f} Pred_mean={p.mean():.1f} Corr={np.corrcoef(g,p)[0,1]:.4f}')

# Pred distribution
print(f'\nPred percentiles: P10={np.percentile(pred,10):.1f} P50={np.percentile(pred,50):.1f} P90={np.percentile(pred,90):.1f}')
print(f'GT percentiles:   P10={np.percentile(gt,10):.1f} P50={np.percentile(gt,50):.1f} P90={np.percentile(gt,90):.1f}')
