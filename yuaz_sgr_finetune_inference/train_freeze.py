"""
train_freeze.py — 冻结解码器重训
流程: 加载合并检查点 → 冻结解码器 → 只训练 conditioning/flow/rvq/encoder
"""
import os, sys, yaml, argparse, time, json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yuaz_sgr'))
from models.svs_model import SVSModel
from data.gpu_dataset import create_gpu_iter


def main():
    parser = argparse.ArgumentParser(description='冻结解码器重训')
    parser.add_argument('--config_dir', type=str, default='yuaz_sgr/config')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed')
    parser.add_argument('--phoneme_map', type=str, default='data/preprocessed/phoneme_map.json')
    parser.add_argument('--checkpoint', type=str, default='merged_checkpoint.pt')
    parser.add_argument('--output_dir', type=str, default='freeze_ckpts')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 加载配置
    config = {}
    for name in ['base', 'model', 'train']:
        with open(os.path.join(args.config_dir, f'{name}.yaml'), 'r') as f:
            config.update(yaml.safe_load(f))
    finetune_path = os.path.join(args.config_dir, 'finetune.yaml')
    if os.path.exists(finetune_path):
        with open(finetune_path, 'r') as f:
            config.update(yaml.safe_load(f))
    
    with open(args.phoneme_map, 'r') as f:
        pm = json.load(f)
    config['conditioning'] = config.get('conditioning', {})
    config['conditioning']['num_phonemes'] = pm['num_phonemes']
    print(f'音素数: {pm["num_phonemes"]}')
    print(f'配置加载完成: {list(config.keys())}')

    # 数据集
    train_iter, val_iter = create_gpu_iter(
        args.data_dir, args.phoneme_map,
        bs=config['batch_size'], device=args.device,
    )
    print(f'训练数据: {train_iter.n} 段, batch_size={config["batch_size"]}')

    # 模型
    model = SVSModel(config)
    model.freeze_f0_encoder()

    # 加载合并检查点 (基座解码器 + 微调其余)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd, strict=False)
    print(f'检查点加载完成: {len(sd)} keys')

    # 冻结解码器
    model.freeze_decoder()
    model.to(args.device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数: {total/1e6:.1f}M, 可训练: {trainable/1e6:.1f}M')

    # 优化器 (只包含非解码器参数)
    freeze_params = model.freeze_decoder_params()
    lr = config['optimizer']['lr']
    betas = config['optimizer']['betas']
    optim_g = torch.optim.Adam(freeze_params, lr=lr, betas=betas)
    optim_d = torch.optim.Adam(
        model.discriminator_params(), lr=lr, betas=betas
    )
    scaler = GradScaler('cuda', enabled=(config['precision'] == 'fp16'))

    # 损失函数
    from losses.stft_loss import MultiResolutionSTFTLoss
    from losses.gan_loss import GANLoss
    stft_loss_fn = MultiResolutionSTFTLoss()
    gan_loss_fn = GANLoss(model.mpd, model.msd)
    loss_weights = config['loss_weights']

    # 阶段
    ae_steps = config['stages']['ae']['end']
    gan_steps = config['stages']['gan']['end']

    # 日志
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'freeze_train_log.jsonl')

    total_steps = config['total_steps']
    log_interval = config.get('log_interval', 100)
    save_interval = config.get('save_interval', 1000)
    eval_interval = config.get('eval_interval', 2000)

    print(f'\n开始训练: 0 → {total_steps} steps\n')

    step = 0
    best_val = float('inf')
    start_time = time.time()
    log_time = start_time

    while step < total_steps:
        for batch in train_iter():
            stage = 'ae' if step <= ae_steps else ('gan' if step <= gan_steps else 'joint')

            # 前向
            with autocast(enabled=(config['precision'] == 'fp16')):
                from train.trainer import compute_total_loss
                total_loss, losses, fake_wav, fake_wav_flow, d_loss, cfm_loss = compute_total_loss(
                    model, batch, stage, stft_loss_fn, gan_loss_fn, loss_weights
                )

            # 生成器 backward
            optim_g.zero_grad()
            need_retain = (stage == 'joint')
            scaler.scale(total_loss).backward(retain_graph=need_retain)
            if cfm_loss is not None:
                scaler.scale(cfm_loss).backward()
            scaler.unscale_(optim_g)
            torch.nn.utils.clip_grad_norm_(freeze_params, max_norm=5.0)

            # 判别器 backward
            if stage != 'ae':
                optim_d.zero_grad()
                disc_wav = fake_wav_flow if stage == 'joint' else fake_wav
                d_loss = gan_loss_fn.compute_d_loss(batch['audio'], disc_wav)
                scaler.scale(d_loss).backward()
                scaler.unscale_(optim_d)
                torch.nn.utils.clip_grad_norm_(model.discriminator_params(), max_norm=5.0)
                losses['d_loss'] = d_loss.item()

            scaler.step(optim_g)
            if stage != 'ae':
                scaler.step(optim_d)
            scaler.update()

            step += 1

            # 日志
            if step % log_interval == 0:
                now = time.time()
                elapsed = now - start_time
                sps = log_interval / max(now - log_time, 1e-8)
                remaining = total_steps - step
                eta = remaining / max(sps, 1e-8)
                eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)
                el_h, el_m = int(elapsed // 3600), int((elapsed % 3600) // 60)

                log_str = f'Step {step:05d} [{stage}]'
                for k, v in losses.items():
                    log_str += f'  {k}={v:.4f}'
                log_str += f'  [{el_h}h{el_m:02d}<{eta_h}h{eta_m:02d} {sps:.1f}it/s]'
                print(log_str)

                with open(log_path, 'a') as f:
                    f.write(json.dumps({'step': step, 'stage': stage, **losses}) + '\n')
                log_time = now

            # 保存
            if step % save_interval == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                path = os.path.join(args.output_dir, f'freeze_{step:06d}.pt')
                raw_sd = model.state_dict()
                clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
                torch.save({'step': step, 'model': clean_sd}, path)
                print(f'  -> Checkpoint: {path}')

            if step >= total_steps:
                break

    # 最终
    final_path = os.path.join(args.output_dir, f'freeze_final.pt')
    raw_sd = model.state_dict()
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
    torch.save({'step': total_steps, 'model': clean_sd}, final_path)
    print(f'\n训练完成! 最终模型: {final_path}')


if __name__ == '__main__':
    main()
