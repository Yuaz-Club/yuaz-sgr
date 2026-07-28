#!/usr/bin/env python3
"""
测试脚本：使用 OpenCPOP 单条数据验证 SVS 框架可行性

测试内容：
1. 配置加载
2. 模型创建 + 参数统计
3. 数据加载（一条数据）
4. 三阶段前向传播（AE / GAN / Joint）
5. 梯度反向传播验证
6. GPU 内存和速度基准测试
"""
import os, sys, json, time, yaml
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'yuaz_sgr'))

from models.svs_model import SVSModel
from data.dataset import SingingDataset, collate_fn
from train.trainer import Trainer, compute_total_loss, compute_flow_matching_loss
from losses.stft_loss import MultiResolutionSTFTLoss
from losses.gan_loss import GANLoss
from losses.cfm_loss import FlowMatchingLoss


def load_config(config_dir):
    config = {}
    for name in ['base', 'model', 'train']:
        path = os.path.join(config_dir, f'{name}.yaml')
        with open(path, 'r') as f:
            config.update(yaml.safe_load(f))
    return config


def test_model_creation(config):
    print("\n" + "="*60)
    print("[测试1] 模型创建与参数统计")
    print("="*60)
    model = SVSModel(config)
    
    gen_params = sum(p.numel() for p in model.generator_params())
    disc_params = sum(p.numel() for p in model.discriminator_params())
    total_params = gen_params + disc_params
    
    print(f"  生成器参数: {gen_params/1e6:.2f}M")
    print(f"  判别器参数: {disc_params/1e6:.2f}M")
    print(f"  总参数量:   {total_params/1e6:.2f}M")
    
    model.freeze_f0_encoder()
    
    return model


def test_data_loading(data_dir, phoneme_map_path, config):
    print("\n" + "="*60)
    print("[测试2] 单条数据加载")
    print("="*60)
    
    dataset = SingingDataset(
        str(Path(data_dir) / 'train'),
        phoneme_map_path,
        segment_sec=config['segment_seconds'],
        sample_rate=config['sample_rate'],
        hop_length=config['hop_length'],
    )
    
    item = dataset[0]
    print(f"  数据项 keys: {list(item.keys())}")
    print(f"  audio shape:   {item['audio'].shape}")
    print(f"  phoneme_ids:   {item['phoneme_ids'].shape}")
    print(f"  durations:     {item['durations'].shape}")
    print(f"  f0:            {item['f0'].shape}")
    print(f"  lags:          {item['lags'].shape}")
    print(f"  note_lengths:  {item['note_lengths'].shape}")
    
    batch = collate_fn([item])
    print(f"\n  Batch keys: {list(batch.keys())}")
    print(f"  batch audio shape:      {batch['audio'].shape}")
    print(f"  batch phoneme_ids shape: {batch['phoneme_ids'].shape}")
    print(f"  batch f0 shape:         {batch['f0'].shape}")
    
    return dataset, batch


def test_forward_pass(model, batch, config, device='cuda'):
    print("\n" + "="*60)
    print("[测试3] 三阶段前向传播测试")
    print("="*60)
    
    # 移动到 GPU
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
    model = model.to(device)
    
    stft_loss_fn = MultiResolutionSTFTLoss()
    gan_loss_fn = GANLoss(model.mpd, model.msd)
    cfm_loss_fn = FlowMatchingLoss()
    
    stages = ['ae', 'gan', 'joint']
    results = {}
    
    for stage in stages:
        print(f"\n  --- 阶段: {stage.upper()} ---")
        
        if stage == 'ae':
            # 纯 AE 前向
            torch.cuda.synchronize()
            t0 = time.time()
            
            total_loss, losses, fake_wav = compute_total_loss(
                model, batch_gpu, stage, stft_loss_fn, gan_loss_fn
            )
            
            torch.cuda.synchronize()
            t1 = time.time()
            
            print(f"    前向时间: {(t1-t0)*1000:.1f}ms")
            print(f"    fake_wav shape: {fake_wav.shape}")
            for k, v in losses.items():
                print(f"    {k}: {v:.6f}")
            
            # 反向传播
            total_loss.backward()
            print(f"    反向传播: OK")
            
        elif stage == 'gan':
            model.zero_grad()
            total_loss, losses, fake_wav = compute_total_loss(
                model, batch_gpu, stage, stft_loss_fn, gan_loss_fn
            )
            total_loss.backward()
            
            # 判别器损失
            d_loss = gan_loss_fn.discriminator_loss(
                batch_gpu['audio'], fake_wav.detach()
            )
            d_loss.backward()
            
            print(f"    GAN generator loss: {losses.get('gan', 0):.6f}")
            print(f"    Feature matching: {losses.get('fm', 0):.6f}")
            print(f"    Discriminator loss: {d_loss.item():.6f}")
            
        elif stage == 'joint':
            model.zero_grad()
            total_loss, losses, fake_wav = compute_total_loss(
                model, batch_gpu, stage, stft_loss_fn, gan_loss_fn
            )
            total_loss.backward()
            
            # CFM 独立损失
            cfm_loss = compute_flow_matching_loss(
                model, batch_gpu, cfm_loss_fn
            )
            cfm_loss.backward()
            
            print(f"    Feature loss: {losses.get('feature', 0):.6f}")
            print(f"    Lag loss: {losses.get('lag', 0):.6f}")
            print(f"    Dur loss: {losses.get('dur', 0):.6f}")
            print(f"    CFM loss: {cfm_loss.item():.6f}")
        
        results[stage] = losses
    
    return results


def test_training_step(model, batch, config, device='cuda'):
    print("\n" + "="*60)
    print("[测试4] 完整训练步骤测试 (含 Trainer)")
    print("="*60)
    
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
    model = model.to(device)
    
    trainer = Trainer(model, config, device=device)
    
    stages = ['ae', 'gan', 'joint']
    for stage in stages:
        print(f"\n  --- Trainer 阶段: {stage.upper()} ---")
        
        # 模拟 step 编号
        if stage == 'ae':
            step = 10000
        elif stage == 'gan':
            step = 50000
        else:
            step = 220000
        
        torch.cuda.synchronize()
        t0 = time.time()
        
        losses, stage_name = trainer.train_step(batch_gpu, step)
        
        torch.cuda.synchronize()
        t1 = time.time()
        
        print(f"    训练时间: {(t1-t0)*1000:.1f}ms")
        for k, v in losses.items():
            print(f"    {k}: {v:.6f}")


def test_memory_benchmark(model, batch, device='cuda'):
    print("\n" + "="*60)
    print("[测试5] GPU 内存基准测试")
    print("="*60)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
    
    # 模拟完整训练步骤 (joint 阶段)
    stft_loss_fn = MultiResolutionSTFTLoss()
    gan_loss_fn = GANLoss(model.mpd, model.msd)
    
    # 前向
    total_loss, losses, fake_wav = compute_total_loss(
        model, batch_gpu, 'joint', stft_loss_fn, gan_loss_fn
    )
    
    # 反向
    total_loss.backward()
    
    torch.cuda.synchronize()
    
    max_memory = torch.cuda.max_memory_allocated() / 1024**2
    current_memory = torch.cuda.memory_allocated() / 1024**2
    
    print(f"  当前 GPU 内存: {current_memory:.0f} MB")
    print(f"  峰值 GPU 内存: {max_memory:.0f} MB")
    print(f"  GPU 总内存:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")


def main():
    print("="*60)
    print("YUAZ SGR 框架可行性测试 - OpenCPOP 单条数据")
    print("="*60)
    
    # 路径
    project_dir = Path(__file__).parent
    yuaz_dir = project_dir / 'yuaz_sgr'
    config_dir = yuaz_dir / 'config'
    data_dir = yuaz_dir / 'data' / 'preprocessed'
    phoneme_map_path = str(data_dir / 'phoneme_map.json')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载配置
    print("\n加载配置...")
    config = load_config(str(config_dir))
    print(f"  基础配置: SR={config['sample_rate']}Hz, "
          f"Hop={config['hop_length']}, "
          f"Segment={config['segment_seconds']}s")
    print(f"  训练配置: batch_size={config['batch_size']}, "
          f"total_steps={config['total_steps']}")
    
    # 2. 创建模型
    model = test_model_creation(config)
    
    # 3. 数据加载
    dataset, batch = test_data_loading(data_dir, phoneme_map_path, config)
    
    # 4. 检查 phoneme_map 是否匹配模型要求
    with open(phoneme_map_path, 'r') as f:
        pm = json.load(f)
    config['conditioning']['num_phonemes'] = pm['num_phonemes']
    print(f"\n  音素表: {pm['num_phonemes']} 个音素")
    
    # 5. 前向传播测试
    test_forward_pass(model, batch, config, device)
    
    # 6. 完整训练步骤测试
    test_training_step(model, batch, config, device)
    
    # 7. 内存基准测试
    test_memory_benchmark(model, batch, device)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == '__main__':
    main()
