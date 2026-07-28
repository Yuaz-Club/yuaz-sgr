import torch, sys, json, yaml
import numpy as np
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')

with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json') as f:
    pm = json.load(f)
config = {}
for n in ['base', 'model']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{n}.yaml') as f:
        config.update(yaml.safe_load(f))
config['conditioning']['num_phonemes'] = pm['num_phonemes']

from models.svs_model import SVSModel
from yuaz_sgr.data.gpu_dataset import create_gpu_iter

# Load trained model weights
model = SVSModel(config).cuda()
# We just trained, don't load checkpoint - run quick forward pass through trained model

# Instead, let's check: is the model still in memory from overfit?
# Let's just check the saved overfit output directly

# Load the infer_pred_f0 using librosa and check
import soundfile as sf
w, sr = sf.read('/root/autodl-tmp/overfit_v3/infer_pred_f0.wav')
print(f'infer_pred_f0: shape={w.shape} min={w.min():.6f} max={w.max():.6f} nan={np.isnan(w).any()} inf={np.isinf(w).any()}')

w2, sr = sf.read('/root/autodl-tmp/overfit_v3/infer_orig_f0.wav')
print(f'infer_orig_f0: shape={w2.shape} min={w2.min():.6f} max={w2.max():.6f} nan={np.isnan(w2).any()} inf={np.isinf(w2).any()}')

w3, sr = sf.read('/root/autodl-tmp/overfit_v3/ae.wav')
print(f'ae: shape={w3.shape} min={w3.min():.6f} max={w3.max():.6f} nan={np.isnan(w3).any()} inf={np.isinf(w3).any()}')

w4, sr = sf.read('/root/autodl-tmp/overfit_v3/orig.wav')
print(f'orig: shape={w4.shape} min={w4.min():.6f} max={w4.max():.6f} nan={np.isnan(w4).any()} inf={np.isinf(w4).any()}')
