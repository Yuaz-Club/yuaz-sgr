# YUAZ SGR
"""scripts/inference"""
import os, sys, yaml, json, argparse, warnings, time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.svs_model import SVSModel

warnings.filterwarnings("ignore")

def load_config(config_dir):
    config = {}
    for name in ['base', 'model', 'train']:
        path = os.path.join(config_dir, f'{name}.yaml')
        with open(path, 'r') as f:
            config.update(yaml.safe_load(f))
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config_dir', default='config')
    parser.add_argument('--test_dir', default='data/preprocessed/test')
    parser.add_argument('--output_dir', default='inference_output')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=8)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config_dir)
    sr = config.get('sample_rate', 24000)

    phoneme_map_path = os.path.join(os.path.dirname(args.test_dir), 'phoneme_map.json')
    p2i, i2p = {}, {}
    if os.path.exists(phoneme_map_path):
        with open(phoneme_map_path, 'r') as f:
            mp = json.load(f)
            p2i = mp.get('p2i', {})
            i2p = mp.get('i2p', {})

    model = SVSModel(config).to(device)
    model.eval()

    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    ckpt_sd = ckpt['model']
    for k, v in ckpt_sd.items():
        clean_k = k.replace('_orig_mod.', '')
        if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
            model_sd[clean_k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_sd, strict=False)
    print(f"  加载: {loaded}, 跳过: {skipped}, step={ckpt.get('step', '?')}")

    if args.eval:
        try:
            from utils.metrics import evaluate_all
            eval_available = True
            print("  评估模块已加载 (pyworld)")
        except Exception as e:
            eval_available = False
            print(f"  评估模块不可用: {e}")

    all_files = sorted(Path(args.test_dir).glob('*.npz'))
    results = {'MCD': [], 'F0-RMSE': [], 'V/UV_F1': [], 'RTF_AE': [], 'RTF_FM': []}
    valid_count = 0

    for npz_path in all_files:
        if valid_count >= args.num_samples:
            break
        data = np.load(npz_path, allow_pickle=True)
        audio_np = data['audio']
        if np.abs(audio_np).max() <= 0.001:
            continue

        valid_count += 1
        name = npz_path.stem
        print(f"\n[{valid_count}/{args.num_samples}] {name}")

        audio_np_1d = audio_np.squeeze()
        audio = torch.FloatTensor(audio_np_1d).unsqueeze(0).unsqueeze(0).to(device)
        dur_sec = audio.shape[-1] / sr

        t0 = time.perf_counter()
        with torch.no_grad():
            z1, f0_enc = model.encoder(audio)
            z_q, indices, _ = model.rvq(z1)
            fake_wav = model.ddsp_decoder(f0_enc, z_q, target_samples=audio.shape[-1])
        t1 = time.perf_counter()
        rtf_ae = (t1 - t0) / dur_sec

        recon = fake_wav.squeeze().cpu().numpy()
        orig = audio.squeeze().cpu().numpy()
        sf.write(os.path.join(args.output_dir, f'{name}_orig.wav'), orig, sr)
        sf.write(os.path.join(args.output_dir, f'{name}_ae.wav'), recon, sr)

        fm_output = None
        rtf_fm = float('nan')
        if 'phonemes' in data and 'durations' in data and 'f0' in data:
            phones = data['phonemes']
            if hasattr(phones, 'tolist'):
                phones = phones.tolist()
            ph_ids = [p2i.get(p, 0) for p in phones]
            dur_np = data['durations'].astype(np.int64)

            f0_np = data['f0'].astype(np.float32)
            f0_np = np.pad(f0_np, (0, max(0, 376 - len(f0_np))))[:376]

            ph = torch.LongTensor(ph_ids).unsqueeze(0).to(device)
            dur = torch.LongTensor(dur_np).unsqueeze(0).to(device)
            f0_in = torch.FloatTensor(f0_np).unsqueeze(0).to(device)

            if 'note_lengths' in data:
                nl = torch.LongTensor(data['note_lengths'].astype(np.int64)).unsqueeze(0).to(device)
            else:
                nl = None

            t0 = time.perf_counter()
            with torch.no_grad():
                wav_fm, _, _, _, _, _ = model.forward_synthesis(
                    ph, dur, f0_in, n_steps=args.n_steps, note_lengths=nl
                )
            t1 = time.perf_counter()
            rtf_fm = (t1 - t0) / dur_sec

            fm_output = wav_fm.squeeze().cpu().numpy()
            sf.write(os.path.join(args.output_dir, f'{name}_fm.wav'), fm_output, sr)

        ae_str = f"AE_RTF={rtf_ae:.4f}"
        fm_str = f"FM_RTF={rtf_fm:.4f}" if fm_output is not None else ""
        print(f"  {ae_str}  {fm_str}")

        if args.eval and eval_available and fm_output is not None:
            wav_ae = fake_wav.squeeze().cpu().numpy()
            min_len = min(len(orig), len(fm_output), len(wav_ae))
            metrics = evaluate_all(orig[:min_len], fm_output[:min_len],
                                   f0_np[:min_len//sr*376] if 'f0' in data else None,
                                   f0_np[:min_len//sr*376] if 'f0' in data else None, sr)
            for k in ['MCD', 'F0-RMSE', 'V/UV_F1']:
                results[k].append(metrics.get(k, float('nan')))
            print(f"  MCD={metrics.get('MCD', float('nan')):.2f}  "
                  f"F0-RMSE={metrics.get('F0-RMSE', float('nan')):.2f}  "
                  f"V/UV={metrics.get('V/UV_F1', float('nan')):.3f}")
            results['RTF_AE'].append(rtf_ae)
            results['RTF_FM'].append(rtf_fm)

    print(f"\n{'='*60}")
    print(f"共 {valid_count} 个样本")
    if args.eval and results['MCD']:
        print(f"{'指标':<12} {'均值':>10}")
        print(f"{'-'*22}")
        for k in ['MCD', 'F0-RMSE', 'V/UV_F1', 'RTF_AE', 'RTF_FM']:
            vals = [v for v in results[k] if np.isfinite(v)]
            if vals:
                print(f"{k:<12} {np.mean(vals):>10.4f}")
    print(f"输出: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main()
