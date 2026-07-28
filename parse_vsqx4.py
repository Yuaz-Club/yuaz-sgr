import xml.etree.ElementTree as ET
tree = ET.parse(r'd:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\【调音者：想吃素的BeaR】【熊】这世界那么多人【无参】.vsqx')
root = tree.getroot()
ns = '{http://www.yamaha.co.jp/vocaloid/schema/vsq4/}'

# 进入 vsTrack -> vsPart
vstrack = root.find(f'{ns}vsTrack')
vspart = vstrack.find(f'{ns}vsPart')
print("=== vsPart 子元素 ===")
for child in list(vspart)[:3]:
    print(f"  {child.tag}: {child.text[:100] if child.text else '(empty)'}")

# 看看 vsPart 的 note 元素
# vsPart 可能直接包含 note 作为子元素，或者包含子结构
notes = vspart.findall(f'{ns}note')
print(f"\nvsPart 直接 note: {len(notes)}")

# 如果没有，可能是在 musicalPart 里
mparts = vspart.findall(f'.//{ns}musicalPart')
print(f"musicalPart: {len(mparts)}")

if mparts:
    for mp in mparts[:2]:
        print(f"\nmusicalPart 子元素:")
        for child in mp:
            print(f"  tag={child.tag.split('}')[-1]}, text={str(child.text)[:80]}")
        notes_in_mp = mp.findall(f'.//{ns}note')
        print(f"  notes: {len(notes_in_mp)}")
        if notes_in_mp:
            for n in notes_in_mp[:5]:
                t = n.find(f'{ns}t')
                dur = n.find(f'{ns}dur')
                note_num = n.find(f'{ns}n')
                lyric = n.find(f'{ns}y')
                print(f"  t={t.text if t is not None else '?'}, dur={dur.text if dur is not None else '?'}, n={note_num.text if note_num is not None else '?'}, y={lyric.text if lyric is not None else '?'}")
else:
    # vsPart 可能直接就是 note 元素
    print(f"\nvsPart 前3子元素:")
    for child in list(vspart)[:5]:
        attrs = {c.tag.split('}')[-1]: c.text for c in child}
        print(f"  {attrs}")
