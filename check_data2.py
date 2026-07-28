import os, json

base = '/root/autodl-tmp/yuaz_sgr/data/preprocessed'
for split in ['train', 'val', 'test']:
    path = os.path.join(base, split)
    files = [f for f in os.listdir(path) if f.endswith('.npz')]
    print(f'{split}: {len(files)} files')

m = json.load(open(os.path.join(base, 'phoneme_map.json')))
print(f'\nphoneme_map: {m["num_phonemes"]} phonemes')
print(f'p2i: {list(m["p2i"].keys())}')

import subprocess
r = subprocess.run(['du', '-sh', base], capture_output=True, text=True)
print(f'\nTotal disk: {r.stdout.strip()}')
