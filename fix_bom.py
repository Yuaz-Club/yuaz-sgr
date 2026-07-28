import json
f = open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json", "r", encoding="utf-8-sig")
d = json.load(f)
f.close()
json.dump(d, open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("fixed")
