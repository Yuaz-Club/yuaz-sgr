import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
import yaml, json, soundfile as sf, glob, scipy.signal
config={}
for n in ["base","model"]:
    config.update(yaml.safe_load(open(f"/root/autodl-tmp/yuaz_sgr/config/{n}.yaml")))
with open("/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json") as f:
    pm=json.load(f)
config["conditioning"]["num_phonemes"]=pm["num_phonemes"]
from models.svs_model import SVSModel
m=SVSModel(config).cuda().eval()
ckpt=torch.load("/root/autodl-tmp/checkpoints/model_030000.pt", weights_only=False)
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
        nonzero_frac=(S_lin>1e-6).float().mean().item()*100
    wav=fake.squeeze().cpu().numpy()
    f_o,P_o=scipy.signal.welch(wav,24000,nperseg=2048)
    f0v=f0[f0>50].mean().item() if (f0>50).any() else 0
    hp=sum(P_o[np.abs(f_o-f0v*m)<=20].sum() for m in [1,2,3,4,5] if f0v*m<12000)
    harm_ratio=hp/P_o.sum()
    tilt=(P_o[np.abs(f_o-2000)<=200].sum())/(P_o[np.abs(f_o-8000)<=500].sum()+1e-8)
    orig_wav=a_orig[:96000]
    f_o2,P_o2=scipy.signal.welch(orig_wav,24000,nperseg=2048)
    f0v2=0
    for freq in [261,293,329,349,392,440,493,523,587,659,698,784,880,987,1046,1174]:
        if P_o2[np.abs(f_o2-freq)<=5].sum()>0:
            f0v2=freq; break
    hp2=sum(P_o2[np.abs(f_o2-f0v2*m)<=20].sum() for m in [1,2,3,4,5] if f0v2*m<12000) if f0v2>0 else 0
    harm2=hp2/P_o2.sum() if P_o2.sum()>0 else 0
    tilt2=(P_o2[np.abs(f_o2-2000)<=200].sum())/(P_o2[np.abs(f_o2-8000)<=500].sum()+1e-8)
    print(f"{name}:")
    print(f"  重建: F0={f0v:.0f}Hz harm={harm_ratio:.3f} tilt={tilt:.2f} lh={lh:.3f} ln={ln_:.3f} Lo/Hi包络比={S_lin_low/S_lin_high:.2f} S_nonzero={nonzero_frac:.0f}%")
    print(f"  原声: F0~={f0v2:.0f}Hz harm={harm2:.4f} tilt={tilt2:.2f}")
    sf.write(f"/root/autodl-tmp/inference_output/ckpt30k_{name}.wav", wav, 24000)
    sf.write(f"/root/autodl-tmp/inference_output/ckpt30k_{name}_orig.wav", orig_wav, 24000)
print("Done!")
