import torch, sys, json, yaml
sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')

with open('/root/autodl-tmp/yuaz_sgr/data/preprocessed_subset/phoneme_map.json') as f:
    pm = json.load(f)
config = {}
for n in ['base', 'model']:
    with open(f'/root/autodl-tmp/yuaz_sgr/config/{n}.yaml') as f:
        config.update(yaml.safe_load(f))
config['conditioning']['num_phonemes'] = pm['num_phonemes']

from models.svs_model import SVSModel
model = SVSModel(config).cuda()

# 测试随机输入
mu = torch.randn(1, 128, 100).cuda()
with torch.no_grad():
    out = model.f0_predictor(mu)
    print(f'Random mu: min={out.min().item():.6f} max={out.max().item():.6f} nan={torch.isnan(out).any().item()} inf={torch.isinf(out).any().item()}')

# 测试极端输入
mu2 = torch.randn(1, 128, 100).cuda() * 100
with torch.no_grad():
    out2 = model.f0_predictor(mu2)
    print(f'Extreme mu (x100): min={out2.min().item():.6f} max={out2.max().item():.6f} nan={torch.isnan(out2).any().item()} inf={torch.isinf(out2).any().item()}')

print('\nF0Predictor params stats:')
for name, p in model.f0_predictor.named_parameters():
    print(f'  {name}: mean={p.data.mean().item():.6f} std={p.data.std().item():.6f} nan={torch.isnan(p.data).any().item()}')
