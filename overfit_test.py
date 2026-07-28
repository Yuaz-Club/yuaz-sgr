# YUAZ SGR — 单样本过拟合快速验证
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'yuaz_sgr'))
from models.svs_model import SVSModel
from losses.stft_loss import MultiResolutionSTFTLoss


def load_config(config_dir):
    config = {}
    for name in ['base', 'model']:
        with open(os.path.join(config_dir, f'{name}.yaml')) as f:
            import yaml
            config.update(yaml.safe_load(f))
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='yuaz_sgr/data/preprocessed_subset')
    parser.add_argument('--phoneme_map', type=str, default='yuaz_sgr/data/preprocessed_subset/phoneme_map.json')
    parser.add_argument('--config_dir', type=str, default='yuaz_sgr/config')
    parser.add_argument('--output_dir', type=str, default='overfit_output')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config_dir)
    with open(args.phoneme_map) as f:
        pm = json.load(f)
    config['conditioning']['num_phonemes'] = pm['num_phonemes']

    # ── 数据 ──
    from yuaz_sgr.data.gpu_dataset import create_gpu_iter
    train_iter, _ = create_gpu_iter(args.data_dir, args.phoneme_map, bs=1, device=args.device)
    batch = next(train_iter())
    audio = batch['audio']; phonemes = batch['phoneme_ids']; durations = batch['durations']
    f0_audio = batch['f0_audio']; note_lengths = batch['note_lengths']
    target_samples = audio.shape[-1]; sr = 24000
    print(f"audio={list(audio.shape)} f0={list(f0_audio.shape)} dur={list(durations.shape)} T={target_samples//256}")

    # 保存原始音频
    sf.write(os.path.join(args.output_dir, 'orig.wav'), audio.squeeze().cpu().numpy(), sr)

    # ── 模型 ──
    model = SVSModel(config).to(args.device)
    model.freeze_f0_encoder()
    model.train()
    print(f"参数量: {sum(p.numel() for p in model.generator_params())/1e6:.1f}M")

    # ── 优化器 (高 lr 快速过拟合) ──
    optim_g = torch.optim.Adam(model.generator_params(), lr=2e-4, betas=(0.8, 0.999))
    scaler = torch.cuda.amp.GradScaler()
    stft_fn = MultiResolutionSTFTLoss()

    n_steps = args.steps
    print(f"\n过拟合 {n_steps} steps (joint, 极速)\n")
    t0 = time.time()

    for step in range(n_steps):
        with torch.cuda.amp.autocast():
            # AE 路径
            z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
            z_q, _, rvq_loss = model.rvq(z1)
            fake_wav = model.ddsp_decoder(f0_enc, z_q, target_samples=target_samples)
            total = stft_fn(audio, fake_wav) + 0.5 * rvq_loss

            # Flow matching
            f0_sq = f0_enc.squeeze(1)
            mu = model.cond_encoder(phonemes, durations, f0_sq)
            T = min(z1.shape[-1], mu.shape[-1])
            total = total + 0.1 * F.mse_loss(mu[:,:,:T], z1[:,:,:T].detach())

            z0 = torch.randn_like(z1)
            t = torch.rand(1, device=z1.device)
            z_t = (1-t)*z0[:,:,:T] + t*z1[:,:,:T].detach()
            gen_out = model.flow_gen(z_t, t.unsqueeze(0), mu[:,:,:T])
            v_pred = gen_out[:,:128,:]
            v_tgt = z1[:,:,:T].detach() - z0[:,:,:T]
            cfm = (1.0/(1.0-t+1e-6) * ((v_pred-v_tgt)**2)).mean()
            total = total + cfm

            f0r = gen_out[:,128:129,:] / 440.0
            f0t = f0_enc[:,:,:T] / 440.0
            if f0t.shape[-1] != f0r.shape[-1]:
                f0t = F.interpolate(f0t, size=f0r.shape[-1])
            total = total + 0.01 * F.mse_loss(f0r, f0t.detach())

        optim_g.zero_grad()
        scaler.scale(total).backward()
        scaler.unscale_(optim_g)
        torch.nn.utils.clip_grad_norm_(model.generator_params(), 5.0)
        scaler.step(optim_g)
        scaler.update()

        if step % 50 == 0 or step == n_steps - 1:
            print(f"step {step:04d} total={total.item():.3f} stft={stft_fn(audio,fake_wav).item():.2f} cfm={cfm.item():.3f} [{time.time()-t0:.0f}s]")

    # ── AE 重建 ──
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        z1, f0_enc = model.encoder(audio, f0_override=f0_audio)
        z_q, _, _ = model.rvq(z1)
        ae = model.ddsp_decoder(f0_enc, z_q, target_samples=target_samples)
    ae = ae.squeeze().cpu().numpy(); ae = ae / (np.abs(ae).max()+1e-8)
    sf.write(os.path.join(args.output_dir, 'ae.wav'), ae, sr)
    print(f"AE 重建: {len(ae)/sr:.1f}s")

    # ── 推理 (平滑 F0) ──
    f0_raw = f0_audio.squeeze().cpu().numpy()
    f0_sm = gaussian_filter1d(f0_raw.astype(np.float64), sigma=6.0)
    nl = note_lengths.squeeze().cpu().numpy().astype(int)
    f0_step = np.zeros_like(f0_sm); pos = 0
    for n in nl:
        if n <= 0: continue
        end = min(pos+n, len(f0_step))
        if end > pos: f0_step[pos:end] = np.mean(f0_sm[pos:end])
        pos = end
    f0_t = torch.FloatTensor(f0_step[:len(f0_raw)]).unsqueeze(0).unsqueeze(0).to(args.device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        wav, _, _, f0o, _, _ = model.forward_synthesis(phonemes, durations, f0_t, n_steps=8, note_lengths=note_lengths)
    wav = wav.squeeze().cpu().numpy(); wav = wav / (np.abs(wav).max()+1e-8)
    sf.write(os.path.join(args.output_dir, 'infer_8.wav'), wav, sr)

    with torch.no_grad(), torch.cuda.amp.autocast():
        wav1, _, _, _, _, _ = model.forward_synthesis(phonemes, durations, f0_t, n_steps=1, note_lengths=note_lengths)
    wav1 = wav1.squeeze().cpu().numpy(); wav1 = wav1 / (np.abs(wav1).max()+1e-8)
    sf.write(os.path.join(args.output_dir, 'infer_1.wav'), wav1, sr)

    print(f"推理 8步={len(wav)/sr:.1f}s 1步={len(wav1)/sr:.1f}s")
    print(f"输出: orig.wav ae.wav infer_8.wav infer_1.wav")


if __name__ == '__main__':
    main()
