import xml.etree.ElementTree as ET
tree = ET.parse(r'd:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\【调音者：想吃素的BeaR】【熊】这世界那么多人【无参】.vsqx')
root = tree.getroot()
ns = '{http://www.yamaha.co.jp/vocaloid/schema/vsq4/}'

# vsTrack
vstrack = root.find(f'{ns}vsTrack')
print("=== vsTrack 子元素 ===")
for child in vstrack:
    print(f"  {child.tag} ({len(list(child))} children)")
    if 'musicalPart' in child.tag:
        for note in child.findall(f'.//{ns}note'):
            t = note.find(f'{ns}t')
            dur = note.find(f'{ns}dur')
            n = note.find(f'{ns}n')
            lyric = note.find(f'{ns}y')
            phnms = note.find(f'{ns}p')
            t_val = int(t.text) if t is not None else 0
            dur_val = int(dur.text) if dur is not None else 0
            n_val = int(n.text) if n is not None else 0
            y_val = lyric.text if lyric is not None else ''
            p_val = phnms.text if phnms is not None else ''
            print(f"  t={t_val} dur={dur_val} note={n_val} lyric='{y_val}' phnms='{p_val}'")
