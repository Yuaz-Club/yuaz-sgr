import xml.etree.ElementTree as ET
tree = ET.parse(r'd:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\【调音者：想吃素的BeaR】【熊】这世界那么多人【无参】.vsqx')
root = tree.getroot()
ns = '{http://www.yamaha.co.jp/vocaloid/schema/vsq4/}'

vspart = root.find(f'.//{ns}vsPart')

# 按 note 标签收集
all_notes = []
for child in vspart:
    if child.tag == f'{ns}note':
        t = child.find(f'{ns}t')
        dur = child.find(f'{ns}dur')
        n_num = child.find(f'{ns}n')
        lyric = child.find(f'{ns}y')
        if t is not None:
            all_notes.append({
                't': int(t.text),
                'dur': int(dur.text) if dur is not None else 0,
                'n': int(n_num.text) if n_num is not None else 0,
                'lyric': lyric.text if lyric is not None else '',
            })

print(f'有效音符: {len(all_notes)}')
for i, n in enumerate(all_notes[:15]):
    print(f"  [{i:3d}] t={n['t']:6d} dur={n['dur']:5d} note={n['n']:3d} lyric='{n['lyric']}'")

# Lyric 统计
lyrics = [n['lyric'] for n in all_notes if n['lyric']]
unique = sorted(set(lyrics))
print(f'\n唯一歌词 ({len(unique)}): {unique[:30]}')

# Tempo
tempo_elem = root.find(f'.//{ns}tempo')
tempo_v = int(tempo_elem.find(f'{ns}v').text) if tempo_elem is not None else 12000
print(f'\nTempo: {tempo_v} (={tempo_v/100:.1f} BPM)')

# 总时长
if all_notes:
    last_t = all_notes[-1]['t'] + all_notes[-1]['dur']
    # VSQ4 tick: quarter = 480 ticks? Let me assume 480
    ticks_per_beat = 480
    secs = (last_t / ticks_per_beat) * (60 / (tempo_v / 100))
    print(f'总 tick: {last_t}, 估计时长: {secs:.1f}s ({secs/60:.1f}min)')
