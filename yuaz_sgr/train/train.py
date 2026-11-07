# YUAZ SGR
"""train/train"""
import os, sys, yaml, argparse, time, json, signal, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# CUDA 内存优化: expandable_segments 缓解碎片化
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.svs_model import SVSModel
from data.dataset import collate_fn
from train.trainer import Trainer
from train.progress import ProgressDashboard

def _spec_image(wav, sr=24000, n_fft=1024, hop=256, max_bins=400):
    """波形 → 归一化 dB 频谱图 (1,H,W)，供 TensorBoard 显示"""
    import torch
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    spec = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft),
                      return_complex=True).abs()
    db = 20 * torch.log10(spec + 1e-6)
    db = db[:max_bins]
    db = ((db - db.max() + 60.0).clamp(min=0) / 60.0)
    return db.unsqueeze(0)

def log_val_media(model, writer, step, npz_path, device='cuda', data_dir=None):
    """把一条验证样本的合成音频 + 频谱图写进 TensorBoard"""
    import torch, numpy as np, json
    d = np.load(npz_path, allow_pickle=True)
    audio = torch.FloatTensor(d['audio']).view(1, 1, -1).to(device)
    f0 = torch.FloatTensor(d['f0']).view(1, 1, -1).to(device)
    ph_arr = d['phonemes']
    if len(ph_arr) and isinstance(ph_arr[0], str):
        pm_path = os.path.join(data_dir or '', 'phoneme_map.json')
        p2i = json.load(open(pm_path, encoding='utf-8'))['p2i']
        ids = [p2i.get(str(p), p2i.get('SP', 1)) for p in ph_arr]
    else:
        ids = [int(p) for p in ph_arr]
    ph = torch.LongTensor(np.asarray(ids, dtype=np.int64)).unsqueeze(0).to(device)
    du = torch.LongTensor(np.asarray(d['durations'], dtype=np.int64)).unsqueeze(0).to(device)
    with torch.no_grad():
        z1, f0e = model.encoder(audio, f0_override=f0)
        T = min(z1.shape[-1], int(du.sum().item()))
        mu = model.cond_encoder(ph, du, f0e[:, :, :T])
        z = model.refiner(mu, steps=int(getattr(model, 'irr_steps', 4)))
        zq, _, _ = model.rvq(z)
        wav = model.ddsp_decoder(f0e[:, :, :T], zq,
                                 target_samples=audio.shape[-1], cond=mu)[..., :audio.shape[-1]]
    syn = wav[0, 0].float().cpu().numpy()
    real = audio[0, 0].float().cpu().numpy()
    peak = max(1e-6, float(np.abs(syn).max()))
    writer.add_audio('val/syn', torch.from_numpy(syn / peak), step, sample_rate=24000)
    writer.add_audio('val/real', torch.from_numpy(real), step, sample_rate=24000)
    writer.add_image('val/spec_syn', _spec_image(syn / peak), step)
    writer.add_image('val/spec_real', _spec_image(real), step)

def load_config(config_dir):
    """加载 YAML 配置文件并合并（支持 finetune.yaml 覆盖）"""
    config = {}
    for name in ['base', 'model', 'train']:
        path = os.path.join(config_dir, f'{name}.yaml')
        with open(path, 'r') as f:
            config.update(yaml.safe_load(f))
    # finetune.yaml 可选加载，覆盖基础配置
    finetune_path = os.path.join(config_dir, 'finetune.yaml')
    if os.path.exists(finetune_path):
        with open(finetune_path, 'r') as f:
            finetune_cfg = yaml.safe_load(f)
        config.update(finetune_cfg)
        print(f"  finetune.yaml  {len(finetune_cfg)} keys")
    return config

def find_latest_ckpt(output_dir):
    """返回 output_dir 下按 step 从大到小排序的 checkpoint 列表"""
    import glob, re
    cks = glob.glob(os.path.join(output_dir, 'model_*.pt'))
    def step_of(p):
        m = re.search(r'model_(\d+)\.pt$', os.path.basename(p))
        return int(m.group(1)) if m else -1
    cks = [c for c in cks if step_of(c) >= 0]
    cks.sort(key=step_of, reverse=True)
    return cks

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"  seed {seed}  deterministic {deterministic}")

def print_logo():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        logo_path = Path(__file__).resolve().parent.parent / "YUAZ LOGO.txt"
        if logo_path.exists():
            sys.stdout.write(logo_path.read_text(encoding="utf-8").rstrip("\n") + "\n")
            sys.stdout.flush()
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description='DDSP + IRR SVS 训练')
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
                        help="从 checkpoint 恢复; 传 'auto'/'latest' 自动选取 output_dir 内最新断点")
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--tf_logdir', type=str, default=None,
                        help='TensorBoard 日志目录 (默认 <log_dir>/tb)')
    parser.add_argument('--max_hours', type=float, default=0.0,
                        help='运行达到该小时数后保存并正常退出 (0=不限), 用于跨 job 续训')
    parser.add_argument('--keep_ckpts', type=int, default=3,
                        help='仅保留最近 N 个 model_*.pt 断点 (0=不清理)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (固定后可复现)')
    parser.add_argument('--deterministic', action='store_true',
                        help='启用确定性算法, 关闭 cudnn autotune (更慢, 更严格可复现)')
    args = parser.parse_args()

    set_seed(args.seed, args.deterministic)
    print_logo()

    # TF32 + cuDNN benchmark
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # 加载配置
    config = load_config(args.config_dir)

    # 加载真实的音素数并更新配置
    with open(args.phoneme_map, 'r') as f:
        pm = json.load(f)
    if 'conditioning' not in config:
        config['conditioning'] = {}
    config['conditioning']['num_phonemes'] = pm['num_phonemes']

    # 创建数据集 (CPU 预加载, 每批移到 GPU)
    from data.gpu_dataset import create_gpu_iter
    train_iter, val_iter = create_gpu_iter(
        args.data_dir, args.phoneme_map,
        bs=config['batch_size'], device=args.device,
    )

    n_train = train_iter.n
    B = config['batch_size']
    print(f"data {n_train}  batch {B}  phon {pm['num_phonemes']}")

    # resume: 'auto'/'latest' 取最新断点
    resume_path = None
    if args.resume:
        if str(args.resume).lower() in ('auto', 'latest'):
            cands = find_latest_ckpt(args.output_dir)
            if cands:
                resume_path = cands[0]
                print(f"  auto-resume {os.path.basename(resume_path)}")
            else:
                print("  auto-resume: 无断点, 从 0 开始")
        elif os.path.exists(args.resume):
            resume_path = args.resume
        else:
            print(f"  resume 路径不存在: {args.resume}, 从 0 开始")

    # 先应用 checkpoint 标志位，再建模型
    if resume_path:
        try:
            _ck = torch.load(resume_path, map_location='cpu', weights_only=True)
            _flags = _ck.get('flags', {}) if isinstance(_ck, dict) else {}
            if _flags:
                config.setdefault('conditioning', {})
                config.setdefault('encoder', {})
                if 'decoder_cond' in _flags:
                    config['decoder_cond'] = bool(_flags['decoder_cond'])
                if 'use_energy' in _flags:
                    config['conditioning']['use_energy'] = bool(_flags['use_energy'])
                if 'f0_predictor' in _flags:
                    config['conditioning']['f0_predictor'] = str(_flags['f0_predictor'])
                if 'encoder_feature' in _flags:
                    config['encoder']['feature'] = str(_flags['encoder_feature'])
                if 'irr_steps' in _flags:
                    config.setdefault('generator', {})['irr_steps'] = int(_flags['irr_steps'])
                print(f"  flags {os.path.basename(resume_path)}: {_flags}")
        except Exception as e:
            print(f"  flags read failed: {e}")

    # 创建模型
    model = SVSModel(config)

    # 冻结 F0 encoder
    model.freeze_f0_encoder()

    # torch.compile 会让 checkpoint 参数名漂移，禁用

    # 统计参数量
    gen_params = sum(p.numel() for p in model.generator_params())
    disc_params = sum(p.numel() for p in model.discriminator_params())
    print(f"params  G {gen_params/1e6:.1f}M  D {disc_params/1e6:.1f}M")

    # 创建 Trainer
    trainer = Trainer(model, config, device=args.device)

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"tf32 {torch.backends.cuda.matmul.allow_tf32}  amp {config['precision']}")

    # 恢复训练 (auto 模式依次尝试最新断点, 损坏则回退到更早的断点)
    start_step = 0
    if resume_path:
        cands = [resume_path]
        if str(args.resume).lower() in ('auto', 'latest'):
            cands = find_latest_ckpt(args.output_dir)
        for p in cands:
            try:
                ckpt_raw = torch.load(p, map_location='cpu')

                if isinstance(ckpt_raw, dict) and 'step' not in ckpt_raw:
                    sd = ckpt_raw.get('model', ckpt_raw)  # 兼容两种子格式
                    safe_path = p + '.wrapped.pt'
                    torch.save({
                        'step': 0, 'model': sd,
                        'optim_g': {}, 'optim_d': {}, 'scaler': {},
                    }, safe_path)
                    start_step = trainer.load_checkpoint(safe_path)
                else:
                    start_step = trainer.load_checkpoint(p)
                resume_path = p
                break
            except Exception as e:
                print(f"  load failed {os.path.basename(p)}: {e}")
                start_step = 0

        with torch.no_grad():
            import librosa
            mel_basis_new = torch.FloatTensor(
                librosa.filters.mel(sr=model.ddsp_decoder.sample_rate,
                                    n_fft=model.ddsp_decoder.fft_size,
                                    n_mels=model.ddsp_decoder.n_mels, fmin=40)
            ).to(model.ddsp_decoder.mel_basis.device)
            if mel_basis_new.shape == model.ddsp_decoder.mel_basis.shape:
                model.ddsp_decoder.mel_basis[...] = mel_basis_new
            mel_pinv_new = torch.linalg.pinv(mel_basis_new)
            mel_pinv_new = torch.clamp(mel_pinv_new, min=0)
            if model.ddsp_decoder.mel_pinv.shape == mel_pinv_new.shape:
                model.ddsp_decoder.mel_pinv[...] = mel_pinv_new.to(model.ddsp_decoder.mel_pinv.device)
            hann_new = torch.hann_window(model.ddsp_decoder.fft_size).to(model.ddsp_decoder.hann_window.device)
            if model.ddsp_decoder.hann_window.shape == hann_new.shape:
                model.ddsp_decoder.hann_window[...] = hann_new
        print(f"resume {start_step}  buffers recomputed")

    # 日志
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'training_log.jsonl')

    # TensorBoard: 标量 + 验证集样音/频谱
    tf_logdir = args.tf_logdir or os.path.join(args.log_dir, 'tb')
    os.makedirs(tf_logdir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=tf_logdir)
        print(f"tensorboard logdir {tf_logdir}")
    except Exception as e:
        tb_writer = None
        print(f"tensorboard disabled: {e}")
    _val_media_file = None
    _val_dir = os.path.join(args.data_dir, 'val')
    if os.path.isdir(_val_dir):
        _cands = sorted(f for f in os.listdir(_val_dir) if f.endswith('.npz'))
        if _cands:
            _val_media_file = os.path.join(_val_dir, _cands[0])

    # ── 训练循环 ──
    total_steps = config['total_steps']
    _st = config['stages']
    dash = ProgressDashboard(
        total_steps,
        stage_ends=[("ae", _st['ae']['end']), ("gan", _st['gan']['end']),
                    ("joint", total_steps)],
        log_interval=config['log_interval'],
        start_step=start_step,
    )
    dash.log(f"run {start_step}->{total_steps}  log_every {config['log_interval']}")

    step = start_step
    epoch = 0
    best_val_loss = float('inf')
    eval_interval = config.get('eval_interval', 10000)

    train_start_time = time.time()
    log_time = train_start_time

    # SIGTERM/SIGINT：存盘后退出
    stop_flag = [False]
    def _sig_stop(signum, frame):
        stop_flag[0] = True
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _sig_stop)
        except Exception:
            pass

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

                dash.draw(step, stage, losses, steps_per_sec, elapsed,
                          getattr(trainer, 'current_lr', config['optimizer']['lr']))

                with open(log_path, 'a') as f:
                    log_entry = {'step': step, 'stage': stage, **losses}
                    f.write(json.dumps(log_entry) + '\n')

                if tb_writer is not None:
                    for k, v in losses.items():
                        if isinstance(v, (int, float)):
                            tb_writer.add_scalar(f'train/{k}', v, step)
                    tb_writer.add_scalar('train/lr', getattr(trainer, 'current_lr', 0.0), step)
                    tb_writer.flush()

            # 保存 checkpoint
            if step % config['save_interval'] == 0:
                ckpt_path = trainer.save_checkpoint(step, args.output_dir)
                if args.keep_ckpts > 0:
                    for old in find_latest_ckpt(args.output_dir)[args.keep_ckpts:]:
                        try:
                            os.remove(old)
                        except OSError:
                            pass
                dash.log(f"save {ckpt_path}")

            # 收到退出信号或达到时间上限：保存后退出
            if stop_flag[0] or (args.max_hours > 0 and (time.time() - train_start_time) > args.max_hours * 3600):
                ckpt_path = trainer.save_checkpoint(step, args.output_dir)
                dash.close()
                print(f"STOP {step}  {ckpt_path}")
                return

            # 验证集评估
            if step % eval_interval == 0 and step > start_step:
                val_losses = trainer.validate(val_iter, step)
                avg_val = val_losses.get('total', 0)
                dash.val(avg_val)
                val_msg = f"val {step}"
                for k, v in val_losses.items():
                    val_msg += f"  {k}={v:.4f}"
                dash.log(val_msg)

                with open(log_path, 'a') as f:
                    log_entry = {'step': step, 'stage': stage, 'type': 'val', **val_losses}
                    f.write(json.dumps(log_entry) + '\n')

                if tb_writer is not None:
                    for k, v in val_losses.items():
                        if isinstance(v, (int, float)):
                            tb_writer.add_scalar(f'val/{k}', v, step)
                    if _val_media_file:
                        try:
                            log_val_media(trainer.model, tb_writer, step,
                                          _val_media_file, device=args.device,
                                          data_dir=args.data_dir)
                        except Exception as e:
                            print(f"tb media failed: {e}")
                    tb_writer.flush()

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    cv_path = trainer.save_checkpoint(step, args.output_dir)
                    bv_path = os.path.join(args.output_dir, 'best_val_model.pt')
                    import shutil
                    if os.path.exists(cv_path):
                        shutil.copy(cv_path, bv_path)
                    dash.log(f"best {bv_path}  total={avg_val:.4f}")

            if step >= total_steps:
                break

        # Epoch 汇总
        avg_losses = {k: v / epoch_batches for k, v in epoch_losses.items()}
        avg_total = avg_losses.get('total', 0)
        dash.log(f"epoch {epoch}  loss {avg_total:.4f}  {stage}")

    # 最终保存
    dash.close()
    final_path = trainer.save_checkpoint(total_steps, args.output_dir)
    with open(os.path.join(args.output_dir, 'TRAIN_DONE'), 'w') as f:
        f.write(f"completed at step {total_steps}\n")
    print(f"done  {final_path}")

if __name__ == '__main__':
    main()