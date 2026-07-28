"""验证 UST 和 MIDI 推理管线的音素输入是否正确"""
import json, mido, os, sys

BASE = "/root/autodl-tmp/yuaz_sgr"
sys.path.insert(0, BASE)

pm = json.load(open(os.path.join(BASE, "data/preprocessed/phoneme_map.json")))
p2i = pm["p2i"]
p2i_set = set(p2i.keys())

# ===== UST 版的 PINYIN_TO_PHONEMES =====
PT = {
    'sheng': ['sh','eng'], 'er': ['e','r'], 'wei': ['w','ei'],
    'ren': ['r','en'], 'wu': ['w','u'], 'zui': ['z','ui'],
    'ni': ['n','i'], 'bu': ['b','u'], 'xu': ['x','v'],
    'yao': ['y','ao'], 'bao': ['b','ao'], 'qian': ['q','ian'],
    'wan': ['w','an'], 'dei': ['d','ei'], 'ai': ['a','i'],
    'e': ['e'], 'ou': ['o','u'], 'bi': ['b','i'], 'bo': ['b','o'],
    'an': ['a','n'], 'yu': ['y','v'], 'u': ['u'], 'gong': ['g','ong'],
    'te': ['t','e'], 'mi': ['m','i'], 'xuan': ['x','van'],
    'hua': ['h','ua'], 'ru': ['r','u'], 'guo': ['g','uo'],
    'ting': ['t','ing'], 'rang': ['r','ang'],
    'wo': ['w','o'], 'pei': ['p','ei'], 'jing': ['j','ing'],
    'ken': ['k','en'], 'hai': ['h','ai'],
    'pu': ['p','u'], 'de': ['d','e'],
    'ei': ['e','i'], 'ao': ['a','o'], 're': ['r','e'],
    'li': ['l','i'], 'fu': ['f','u'], 'rui': ['r','ui'],
    'na': ['n','a'], 'duo': ['d','uo'], 'mei': ['m','ei'],
    'gui': ['g','ui'], 'you': ['y','ou'], 'ji': ['j','i'],
    'yi': ['y','i'], 'hao': ['h','ao'], 'shi': ['sh','i'],
    'kai': ['k','ai'], 'fan': ['f','an'],
    'a': ['a'], 'o': ['o'], 'y': ['y'], 'v': ['v'],
}

# ===== 检查 UST =====
print("=" * 60)
print("  玫瑰少年 UST 歌词映射分析")
print("=" * 60)

ust_lyrics = set()
with open(os.path.join(BASE, "inference_input/玫瑰少年-autosave_FL Keys_04.ust"), "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line.startswith("Lyric="):
            l = line.split("=", 1)[1].strip().lower()
            if l not in ("r", "+", "-R", ""):
                ust_lyrics.add(l)

correct, wrong = 0, 0
for lyric in sorted(ust_lyrics):
    if lyric in PT:
        phs = PT[lyric]
        ids = [p2i.get(ph, -1) for ph in phs]
        missing = [ph for ph, i in zip(phs, ids) if i == -1]
        if missing:
            print(f"  ✗ {lyric:10s} → PT分解 {str(phs):25s} 缺失: {missing}")
            wrong += 1
        else:
            print(f"  ✓ {lyric:10s} → PT分解 {str(phs):25s}")
            correct += 1
    elif lyric in p2i_set:
        print(f"  ✓ {lyric:10s} → 直接音素 id={p2i[lyric]}")
        correct += 1
    else:
        print(f"  ✗ {lyric:10s} → FALLBACK 到 'a'! '{lyric}' 不在 PT 也不在 p2i")
        wrong += 1

print(f"\nUST 结果: 正确 {correct}, 错误 {wrong}\n")

# ===== 检查 MIDI =====
print("=" * 60)
print("  思华年 MIDI 歌词映射分析")
print("=" * 60)

from midi_infer import PINYIN_TO_PHONEMES as MP

mid = mido.MidiFile(os.path.join(BASE, "inference_input/思华年.mid"))
midi_lyrics = set()
for t in mid.tracks:
    at = 0
    for m in t:
        at += m.time
        if m.type in ("lyrics", "text", "cue_marker"):
            l = m.text.strip().lower()
            if l != "+":
                midi_lyrics.add(l)

correct2, wrong2 = 0, 0
for lyric in sorted(midi_lyrics):
    if lyric in MP:
        phs = MP[lyric]
        ids = [p2i.get(ph, -1) for ph in phs]
        missing = [ph for ph, i in zip(phs, ids) if i == -1]
        if missing:
            print(f"  ✗ {lyric:10s} → MP分解 {str(phs):25s} 缺失: {missing}")
            wrong2 += 1
        else:
            print(f"  ✓ {lyric:10s} → MP分解 {str(phs):25s}")
            correct2 += 1
    else:
        print(f"  ✗ {lyric:10s} → 不在 MP 映射, fallback到 'a'")
        wrong2 += 1

print(f"\nMIDI 结果: 正确 {correct2}, 错误 {wrong2}")
