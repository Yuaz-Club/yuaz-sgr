"""检查服务器 phoneme_map vs 推理包 phoneme_map"""
import json, os

# 服务器
srv = json.load(open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json'))
print('=== 服务器 phoneme_map ===')
print(f'num_phonemes: {srv["num_phonemes"]}')
print(f'p2i keys: {list(srv["p2i"].keys())}')
print(f'p2i values: {list(srv["p2i"].values())}')

# 也检查所有训练数据收集到的音素集
import numpy as np
base = '/root/autodl-tmp/yuaz_sgr/data/preprocessed/train'
all_ph = set()
for f in os.listdir(base)[:100]:
    d = np.load(os.path.join(base, f), allow_pickle=True)
    for p in d['phonemes']:
        all_ph.add(str(p))
print(f'\n前100文件实际音素 ({len(all_ph)}): {sorted(all_ph)}')

# 检查推理包（本地）
inf = json.load(open('/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json'))
print(f'\n推理包 phoneme_map: {inf["num_phonemes"]} 音素')
print(f'推理包 keys: {list(inf["p2i"].keys())}')
print(f'推理包 values: {list(inf["p2i"].values())}')

# 比较
if srv['p2i'] == inf['p2i']:
    print('\n✅ 完全一致!')
else:
    print('\n❌ 不一致!')
    srv_only = set(srv['p2i'].keys()) - set(inf['p2i'].keys())
    inf_only = set(inf['p2i'].keys()) - set(srv['p2i'].keys())
    if srv_only: print(f'  服务器独有: {srv_only}')
    if inf_only: print(f'  推理包独有: {inf_only}')
    # 对比公共音素的 ID
    for k in sorted(set(srv['p2i'].keys()) & set(inf['p2i'].keys())):
        if srv['p2i'][k] != inf['p2i'][k]:
            print(f'  ID不一致: {k}: 服务器={srv["p2i"][k]}, 推理包={inf["p2i"][k]}')
