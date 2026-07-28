import torch as t
import numpy as np

ck = t.load('/root/autodl-tmp/checkpoints/model_003315.pt', map_location='cpu', weights_only=False)
sd = ck['model']

print("=== weight_net params ===")
for k in ['ddsp_decoder.weight_net.0.weight','ddsp_decoder.weight_net.0.bias','ddsp_decoder.weight_net.2.weight','ddsp_decoder.weight_net.2.bias']:
    v = sd[k].numpy()
    print(f"{k}: mean={v.mean():.6f} std={v.std():.6f} min={v.min():.6f} max={v.max():.6f}")

print("\n=== mel_pinv ===")
mp = sd['ddsp_decoder.mel_pinv'].numpy()
print(f"shape={mp.shape} min={mp.min():.6f} max={mp.max():.6f} mean={mp.mean():.6f}")

import sys
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
import yaml

config = {}
for n in ['base','model','train']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{n}.yaml') as f:
        config.update(yaml.safe_load(f))

from models.svs_model import SVSModel
model = SVSModel(config)
model.load_state_dict(sd, strict=False)
model.eval()

import soundfile as sf
d,_ = sf.read('/root/autodl-tmp/inference_output/ref_2044001628.wav')
audio = t.FloatTensor(d).unsqueeze(0).unsqueeze(0)

with t.no_grad():
    z1, f0_enc = model.encoder(audio)
    z_q, _, _ = model.rvq(z1)
    wav = model.ddsp_decoder(f0_enc, z_q, target_samples=audio.shape[-1])

wav_np = wav.squeeze().numpy()
print(f"\n=== DDSP output stats (NO tanh in model) ===")
print(f"min={wav_np.min():.6f} max={wav_np.max():.6f} mean={wav_np.mean():.6f} std={wav_np.std():.6f}")
print(f"p99={np.percentile(np.abs(wav_np), 99):.6f}")
print(f"% over 0.9 = {(np.abs(wav_np)>0.9).mean()*100:.2f}%")
print(f"% over 0.99 = {(np.abs(wav_np)>0.99).mean()*100:.2f}%")
