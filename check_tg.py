import textgrid
tg = textgrid.TextGrid.fromFile("/root/autodl-tmp/yuaz_sgr/opencpop/textgrids/2001.TextGrid")
print([t.name for t in tg.tiers])
for t in tg.tiers:
    print(t.name, len(t))
