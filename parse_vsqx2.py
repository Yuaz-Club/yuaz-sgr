import xml.etree.ElementTree as ET
tree = ET.parse(r'd:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\【调音者：想吃素的BeaR】【熊】这世界那么多人【无参】.vsqx')
root = tree.getroot()
ns = {'x': 'http://www.yamaha.co.jp/vocaloid/schema/vsq4/'}

# 打印第一层结构
print("=== 根子元素 ===")
for child in root:
    print(f"  {child.tag}: {len(list(child))} 个子元素")

# 深入 musPart
parts = root.findall('.//x:musicalPart', ns)
print(f"\n=== musicalPart 数: {len(parts)} ===")

for p_idx, part in enumerate(parts):
    print(f"\nPart {p_idx}:")
    # 找所有 note
    notes = part.findall('.//x:note', ns)
    print(f"  notes: {len(notes)}")
    if notes:
        for i, n in enumerate(notes[:10]):
            children = {c.tag.split('}')[-1]: c.text for c in n}
            print(f"  [{i}] {children}")

# 找 tempo (可能在 auPart 或 masterTrack)
tempos = root.findall('.//{http://www.yamaha.co.jp/vocaloid/schema/vsq4/}tempo')
print(f"\n=== Tempo: {len(tempos)} 个 ===")
for t in tempos:
    print(f"  {ET.tostring(t, encoding='unicode')[:200]}")
