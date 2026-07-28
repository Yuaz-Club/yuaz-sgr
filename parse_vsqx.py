import xml.etree.ElementTree as ET
tree = ET.parse(r'd:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\【调音者：想吃素的BeaR】【熊】这世界那么多人【无参】.vsqx')
root = tree.getroot()
ns = {'x': 'http://www.yamaha.co.jp/vocaloid/schema/vsq4/'}

notes = root.findall('.//x:note', ns)
print(f'音符数: {len(notes)}')
for i, n in enumerate(notes[:15]):
    tick = n.find('x:tick', ns)
    dur = n.find('x:durTick', ns)
    note_num = n.find('x:noteNum', ns)
    lyric = n.find('x:lyric', ns)
    phnms = n.find('x:phnms', ns)
    tick_v = int(tick.text) if tick is not None else 0
    dur_v = int(dur.text) if dur is not None else 0
    note_v = int(note_num.text) if note_num is not None else 0
    lyric_v = lyric.text if lyric is not None else ''
    phnms_v = phnms.text if phnms is not None else ''
    print(f'  {i}: tick={tick_v}, dur={dur_v}, note={note_v}, lyric={lyric_v}, phnms={phnms_v}')

# 找 tempo
tempos = root.findall('.//x:tempo', ns)
print(f'\nTempo 数: {len(tempos)}')
for i, t in enumerate(tempos[:5]):
    pos = t.find('x:posTick', ns)
    bpm = t.find('x:bpm', ns)
    print(f'  {i}: posTick={pos.text if pos is not None else "?"}, bpm={bpm.text if bpm is not None else "?"}')
