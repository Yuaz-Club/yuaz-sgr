import sys, yaml, json
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
config = {}
for n in ["base", "model", "train"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    config["conditioning"]["num_phonemes"] = json.load(f)["num_phonemes"]
from models.svs_model import SVSModel
m = SVSModel(config)
gen = sum(p.numel() for p in m.generator_params())
disc = sum(p.numel() for p in m.discriminator_params())
print(f"Generator: {gen/1e6:.1f}M  Discriminator: {disc/1e6:.1f}M  Total: {(gen+disc)/1e6:.1f}M")
print("论文 Table 1: Ours=18.6M（仅生成器, 推理时 D 不需要）")
