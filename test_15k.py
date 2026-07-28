import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, soundfile as sf, glob, scipy.signal
config={}
for n in ["base", "model"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    pm=json.load(f)
config["conditioning"]["num_phonemes"]=pm["num_phonemes"]
from models.svs_model import SVSModel
m=SVSModel(config).cuda().eval()
ckpt=torch.load("/root/autodl-tmp/checkpoints/model_015000.pt", weights_only=False)
m.load_state_dict(ckpt["model"], strict=False)
files=sorted(glob.glob("/root/autodl-tmp/yuaz_sgr/data/preprocessed/test/*.npz"))
os.makedirs("/root/autodl-tmp/inference_output", exist_ok=True)
for npz in files[:2]:
    data=np.load(npz, allow_pickle=True)
    a_orig=data["audio"].squeeze()
    a=torch.FloatTensor(a_orig[:96000]).unsqueeze(0).unsqueeze(0).cuda()
    name=os.path.basename(npz).replace(".npz","")
    with torch.no_grad():
        z1,f0=m.encoder(a)
        z_q,_,_=m.rvq(z1)
        fake=m.ddsp_decoder(f0,z_q,target_samples=a.shape[-1])
        dd=m.ddsp_decoder
        if f0.dim()==2: f0=f0.unsqueeze(1)
        feat=torch.cat([z_q,f0],dim=1).transpose(1,2)
        feat=dd.emformer_input_proj(feat)
        h,_=dd.emformer(feat)
        h=dd.emformer_proj(h).transpose(1,2)
        Sc=dd.env_net(h.transpose(1,2)).transpose(1,2)
        Ac=dd.ap_net(h.transpose(1,2)).transpose(1,2)
        w=dd.weight_net(h.transpose(1,2)).transpose(1,2)
        S_lin=dd._decompress_envelope(Sc)
        S_lin_high=S_lin[0,400:,:].mean().item()
        S_lin_low=S_lin[0,:50,:].mean().item()
        lh=torch.clamp(1.0/(w[:,0:1,:]+1e-8),min=0).mean().item()
        ln_=torch.clamp(1.0/(w[:,1:2,:]+1e-8),min=0).mean().item()
        zero_frac=(S_lin<1e-6).float().mean().item()*100
    wav=fake.squeeze().cpu().numpy()
    f_o,P_o=scipy.signal.welch(wav,24000,nperseg=2048)
    f0v=f0[f0>50].mean().item() if (f0>50).any() else 0
    hp=sum(P_o[np.abs(f_o-f0v*m)<=20].sum() for m in [1,2,3,4,5] if f0v*m<12000)
    tilt=(P_o[np.abs(f_o-2000)<=200].sum())/(P_o[np.abs(f_o-8000)<=500].sum()+1e-8)
    print(f"{name}: rms={np.sqrt(np.mean(wav**2)):.4f} max={np.abs(wav).max():.2f} F0={f0v:.0f}Hz harm/total={hp/P_o.sum():.4f} tilt={tilt:.2f}")
    print(f"  lh={lh:.3f} ln={ln_:.3f} S_zeros={zero_frac:.0f}% 高频/低频={S_lin_high/S_lin_low:.2f}")
    sf.write(f"/root/autodl-tmp/inference_output/ckpt15k_{name}.wav", wav, 24000)
print("Done!")
