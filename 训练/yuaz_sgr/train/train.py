# YUAZ SGR
"""
train/train.py
论文 §4.2: 主训练入口

三阶段训练 (250k steps):
  阶段 1 (0~20k):    AE 预训练 — 波形自重建
  阶段 2 (20k~200k): 引入 MPD+MSD 判别器
  阶段 3 (200k~250k): 全联合优化 — 接入 Flow Matching + 时间模型
"""
import os, sys, yaml, argparse, time, json
from pathlib import Path
import torch
import torch.nn as nn

# CUDA 内存优化: expandable_segments 缓解碎片化
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.svs_model import SVSModel
from data.dataset import collate_fn
from train.trainer import Trainer


def load_config(config_dir):
    """加载 YAML 配置文件并合并"""
    config = {}
    for name in ['base', 'model', 'train']:
        path = os.path.join(config_dir, f'{name}.yaml')
        with open(path, 'r') as f:
            config.update(yaml.safe_load(f))
    return config


def main():
    parser = argparse.ArgumentParser(description='DDSP + Flow Matching SVS 训练')
    parser.add_argument('--config_dir', type=str, default='config',
                        help='配置文件目录')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='预处理后的数据目录')
    parser.add_argument('--phoneme_map', type=str, required=True,
                        help='音素映射表路径')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='模型输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    args = parser.parse_args()

    # 启用 TF32 + cuDNN benchmark (A800 加速, 论文无冲突)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # 加载配置
    config = load_config(args.config_dir)
    print(f"配置加载完成: {list(config.keys())}")

    # 加载真实的音素数并更新配置
    with open(args.phoneme_map, 'r') as f:
        pm = json.load(f)
    if 'conditioning' not in config:
        config['conditioning'] = {}
    config['conditioning']['num_phonemes'] = pm['num_phonemes']
    print(f"音素数: {pm['num_phonemes']}")

    # 创建数据集 (CPU 预加载, 每批移到 GPU)
    from data.gpu_dataset import create_gpu_iter
    train_iter, val_iter = create_gpu_iter(
        args.data_dir, args.phoneme_map,
        bs=config['batch_size'], device=args.device,
    )

    n_train = train_iter.n
    B = config['batch_size']
    print(f"训练数据: {n_train} 段, batch_size={B}, 每 epoch {n_train // B} batches")

    # 创建模型
    model = SVSModel(config)

    # 冻结 CREPE F0 encoder
    model.freeze_f0_encoder()

    # torch.compile 加速约 10%-15%, 但每次重启 checkpoint 加载时
    # compile 重写计算图会导致参数名漂移 (跳过 430 个参数),
    # 判别器重新随机初始化。禁用 compile 避免此问题。
    # if args.device == 'cuda':
    #     try:
    #         model = torch.compile(model, mode='default')
    #         print("torch.compile 已启用 (mode=default)")
    #     except Exception as e:
    #         print(f"torch.compile 未启用: {e}")

    # 统计参数量
    gen_params = sum(p.numel() for p in model.generator_params())
    disc_params = sum(p.numel() for p in model.discriminator_params())
    print(f"参数量: Generator={gen_params/1e6:.1f}M, Discriminator={disc_params/1e6:.1f}M")

    # 创建 Trainer
    trainer = Trainer(model, config, device=args.device)

    # 启用 TF32 + cuDNN benchmark 加速 (RTX 4090 支持)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"TF32: matmul={torch.backends.cuda.matmul.allow_tf32}  cudnn={torch.backends.cudnn.allow_tf32}  benchmark={torch.backends.cudnn.benchmark}  precision={torch.get_float32_matmul_precision()}")

    # 恢复训练
    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)
        # 强制重算 mel_basis + mel_pinv + hann_window
        # (应对架构变更: fft_size 1024→2048, 维度 513→1025)
        # + 重置时序预测器的新层 (conv 预处理器，架构变更)
        with torch.no_grad():
            import librosa
            mel_basis_new = torch.FloatTensor(
                librosa.filters.mel(sr=model.ddsp_decoder.sample_rate,
                                    n_fft=model.ddsp_decoder.fft_size,
                                    n_mels=128, fmin=40)
            ).to(model.ddsp_decoder.mel_basis.device)
            if mel_basis_new.shape != model.ddsp_decoder.mel_basis.shape:
                print(f"  [恢复] mel_basis 维度变更: {list(model.ddsp_decoder.mel_basis.shape)} → {list(mel_basis_new.shape)}")
            model.ddsp_decoder.mel_basis[...] = mel_basis_new
            mel_pinv_new = torch.linalg.pinv(mel_basis_new)
            mel_pinv_new = torch.clamp(mel_pinv_new, min=0)
            if mel_pinv_new.shape != model.ddsp_decoder.mel_pinv.shape:
                print(f"  [恢复] mel_pinv 维度变更: {list(model.ddsp_decoder.mel_pinv.shape)} → {list(mel_pinv_new.shape)}")
            model.ddsp_decoder.mel_pinv[...] = mel_pinv_new.to(model.ddsp_decoder.mel_pinv.device)
            hann_new = torch.hann_window(model.ddsp_decoder.fft_size).to(model.ddsp_decoder.hann_window.device)
            model.ddsp_decoder.hann_window[...] = hann_new
        print(f"从 step {start_step} 恢复训练（mel_basis/mel_pinv/hann 已重算）")

    # 日志
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'training_log.jsonl')

    # ── 训练循环 ──
    total_steps = config['total_steps']
    print(f"\n开始训练: {start_step} → {total_steps} steps\n")

    step = start_step
    epoch = 0
    best_val_loss = float('inf')
    eval_interval = config.get('eval_interval', 10000)

    train_start_time = time.time()
    log_time = train_start_time

    while step < total_steps:
        epoch += 1
        epoch_losses = {}
        epoch_batches = 0

        for batch in train_iter():

            losses, stage = trainer.train_step(batch, step)
            step += 1
            epoch_batches += 1

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v

            # 日志
            if step % config['log_interval'] == 0:
                now = time.time()
                elapsed = now - train_start_time
                step_elapsed = now - log_time
                log_time = now
                steps_per_sec = config['log_interval'] / max(step_elapsed, 1e-8)
                remaining = total_steps - step
                eta_sec = remaining / max(steps_per_sec, 1e-8)
                eta_h, eta_m = int(eta_sec // 3600), int((eta_sec % 3600) // 60)
                elapsed_h, elapsed_m = int(elapsed // 3600), int((elapsed % 3600) // 60)

                log_str = f"Step {step:06d} [{stage}]"
                for k, v in losses.items():
                    log_str += f"  {k}={v:.4f}"
                log_str += f"  [{elapsed_h}h{elapsed_m:02d}<{eta_h}h{eta_m:02d} {steps_per_sec:.1f}it/s]"
                print(log_str)

                with open(log_path, 'a') as f:
                    log_entry = {'step': step, 'stage': stage, **losses}
                    f.write(json.dumps(log_entry) + '\n')

            # 保存 checkpoint
            if step % config['save_interval'] == 0:
                ckpt_path = trainer.save_checkpoint(step, args.output_dir)
                print(f"  → Checkpoint: {ckpt_path}")

            # 验证集评估
            if step % eval_interval == 0 and step > start_step:
                val_losses = trainer.validate(val_iter, step)
                avg_val = val_losses.get('total', 0)
                val_msg = f"  [验证] step={step:06d}"
                for k, v in val_losses.items():
                    val_msg += f"  val_{k}={v:.4f}"
                print(val_msg)

                with open(log_path, 'a') as f:
                    log_entry = {'step': step, 'stage': stage, 'type': 'val', **val_losses}
                    f.write(json.dumps(log_entry) + '\n')

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    cv_path = trainer.save_checkpoint(step, args.output_dir)
                    bv_path = os.path.join(args.output_dir, 'best_val_model.pt')
                    import shutil
                    if os.path.exists(cv_path):
                        shutil.copy(cv_path, bv_path)
                    print(f"  → 最佳验证模型: {bv_path} (val_loss={avg_val:.4f})")

            if step >= total_steps:
                break

        # Epoch 汇总
        avg_losses = {k: v / epoch_batches for k, v in epoch_losses.items()}
        avg_total = avg_losses.get('total', 0)
        print(f"Epoch {epoch:03d} avg_loss={avg_total:.4f}  stage={stage}")


    # 最终保存
    final_path = trainer.save_checkpoint(total_steps, args.output_dir)
    print(f"\n训练完成! 最终模型: {final_path}")


if __name__ == '__main__':
    main()