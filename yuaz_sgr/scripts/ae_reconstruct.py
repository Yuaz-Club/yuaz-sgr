# YUAZ SGR
"""AE 重建验证脚本 加载 checkpoint, 对验证集音频做 encoder → RVQ → DDSP decoder 重建, 保存原始 / 重建波形供主观听感对比"""
import os, sys, yaml, argparse
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.svs_model import SVSModel

def load_config(config_dir):
    config = {}
    for name in ['base', 'model']:
        path = os.path.join(config_dir, f'{name}.yaml')
        with open(path) as f:
            config.update(yaml.safe_load(f))
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--config_dir', type=str, default='config')
    parser.add_argument('--output_dir', type=str, default='ae_reconstructions')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config_dir)
    pm_path = os.path.join(os.path.dirname(args.data_dir), 'phoneme_map.json')
    if os.path.exists(pm_path):
        import json
        with open(pm_path) as f:
            pm = json.load(f)
        if 'conditioning' not in config:
            config['conditioning'] = {}
        config['conditioning']['num_phonemes'] = pm['num_phonemes']
        print(f"音素数: {pm['num_phonemes']}")
    else:
        config['conditioning']['num_phonemes'] = 60

    # 构建模型
    model = SVSModel(config).to(args.device)
    model.eval()

    # 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_sd = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in ckpt['model'].items():
        clean_k = k.replace('_orig_mod.', '')
        if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
            model_sd[clean_k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_sd, strict=False)
    print(f"Checkpoint 加载完成: {loaded} 参数匹配, {skipped} 跳过")
    step = ckpt.get('step', 0)
    print(f"从 step {step} 恢复")

    # 加载验证集音频
    val_files = sorted(Path(args.data_dir).glob('*.npz'))
    print(f"验证集共 {len(val_files)} 段, 取前 {args.num_samples} 段重建")

    sr = config.get('sample_rate', 24000)
    model.ddsp_decoder.eval()

    for idx, fpath in enumerate(val_files[:args.num_samples]):
        data = np.load(fpath, allow_pickle=True)
        audio = data['audio']
        f0 = data['f0']

        # 对齐到 4 秒
        target_samples = 4 * sr
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        audio = audio[:target_samples]

        # 推理
        audio_t = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(args.device)  # (1,1,T)
        f0_t = torch.FloatTensor(f0[:len(f0)]).unsqueeze(0).to(args.device)  # (1,F)

        with torch.no_grad():
            z1, f0_out, z_q, fake_wav, rvq_loss = model.forward_ae(audio_t, f0_override=f0_t)

        fake_wav = fake_wav.squeeze().cpu().numpy()

        # 保存原始和重建
        base_name = fpath.stem
        sf.write(os.path.join(args.output_dir, f'{base_name}_orig.wav'), audio, sr)
        sf.write(os.path.join(args.output_dir, f'{base_name}_recon.wav'), fake_wav, sr)

        # 简单数值对比
        orig_norm = np.sqrt(np.mean(audio ** 2))
        recon_norm = np.sqrt(np.mean(fake_wav ** 2))
        print(f"  [{idx+1}/{args.num_samples}] {base_name}: "
              f"rvq_loss={rvq_loss.item():.4f}  "
              f"orig_rms={orig_norm:.4f}  recon_rms={recon_norm:.4f}")

    print(f"\n完成! 输出目录: {args.output_dir}")
    print(f"文件列表:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  {f}")

if __name__ == '__main__':
    main()
