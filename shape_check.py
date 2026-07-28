import sys, torch, yaml, json, numpy as np
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
config = {}
for n in ["base", "model"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    pm = json.load(f)
config["conditioning"]["num_phonemes"] = pm["num_phonemes"]
from models.svs_model import SVSModel
m = SVSModel(config).cuda().eval()
ckpt = torch.load("/root/autodl-tmp/checkpoints/model_020000.pt", weights_only=False)
m.load_state_dict(ckpt["model"], strict=False)
dd = m.ddsp_decoder
import glob
files = sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
data = np.load(files[0], allow_pickle=True)
a = torch.FloatTensor(data["audio"].squeeze()[:96000]).unsqueeze(0).unsqueeze(0).cuda()
with torch.no_grad():
    z1, f0 = m.encoder(a)
    z_q, _, _ = m.rvq(z1)
    B, _, T = z_q.shape
    if f0.dim() == 2: f0 = f0.unsqueeze(1)
    feat = torch.cat([z_q, f0], dim=1).transpose(1, 2)
    print("feat shape:", feat.shape)
    feat = dd.emformer_input_proj(feat)
    print("after input_proj:", feat.shape)
    emf_raw = dd.emformer(feat)
    emf = emf_raw[0] if isinstance(emf_raw, tuple) else emf_raw
    print("emformer type:", type(dd.emformer).__name__)
    print("emf shape:", emf.shape)
    Sc = dd.env_net(emf)
    print("\nenv_net(emf) -> S_c shape:", Sc.shape)
    Ac = dd.ap_net(emf)
    print("ap_net(emf)  -> A_c shape:", Ac.shape)
    w = dd.weight_net(emf)
    print("weight_net(emf) -> w shape:", w.shape)
    
    # _decompress_envelope call — does it work without transpose?
    try:
        S_lin = dd._decompress_envelope(Sc)
        print("_decompress_envelope(S_c): S_lin shape =", S_lin.shape)
    except Exception as e:
        print("_decompress_envelope FAILED:", e)
    
    fake = dd(f0, z_q, target_samples=a.shape[-1])
    print("Forward output shape:", fake.shape)
    print("All OK!")
