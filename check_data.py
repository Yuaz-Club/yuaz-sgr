"""检查训练数据内容（服务器上跑）"""
import numpy as np, os
dp = '/root/autodl-tmp/yuaz_sgr/data/preprocessed/train'
for f in sorted(os.listdir(dp))[:5]:
    d = np.load(os.path.join(dp, f), allow_pickle=True)
    print('===', f, '===')
    print('  audio:', d['audio'].shape, 'range:', round(d['audio'].min(),4), '~', round(d['audio'].max(),4))
    ph = d['phonemes']
    print('  phonemes:', len(ph), list(ph[:15]))
    dur = d['durations']
    print('  durations:', len(dur), list(dur[:15]), 'sum=', dur.sum())
    l = d['lags']
    print('  lags:', l[:10].tolist(), '... sum=', round(l.sum(),1))
    f0 = d['f0']
    nonzero = int((f0>0).sum())
    print('  f0:', f0.shape, 'nonzero:', nonzero, '/', len(f0))
    if nonzero:
        print('  f0 range:', round(float(f0[f0>0].min()),1), '~', round(float(f0.max()),1), 'Hz')
