# YUAZ SGR
"""
utils/metrics.py
评估指标

论文 Table 1:
  MCD (Mel-Cepstral Distortion)
  F0-RMSE (F0 Root Mean Square Error)
  V/UV F1 (Voiced/Unvoiced accuracy)
"""
import torch
import numpy as np
import pyworld as pw


def compute_mcd(ref_wav, syn_wav, sr=24000, n_fft=1024, hop_length=256):
    """
    Mel-Cepstral Distortion (dB)
    论文: 使用 WORLD vocoder 提取特征
    """
    # WORLD 分析
    ref_f0, ref_t = pw.dio(ref_wav.astype(np.float64), sr)
    ref_f0 = pw.stonemask(ref_wav.astype(np.float64), ref_f0, ref_t, sr)
    ref_sp = pw.cheaptrick(ref_wav.astype(np.float64), ref_f0, ref_t, sr)
    ref_mcep = pw.code_spectral_envelope(ref_sp, sr, 24)

    syn_f0, syn_t = pw.dio(syn_wav.astype(np.float64), sr)
    syn_f0 = pw.stonemask(syn_wav.astype(np.float64), syn_f0, syn_t, sr)
    syn_sp = pw.cheaptrick(syn_wav.astype(np.float64), syn_f0, syn_t, sr)
    syn_mcep = pw.code_spectral_envelope(syn_sp, sr, 24)

    # 对齐帧数
    min_len = min(ref_mcep.shape[0], syn_mcep.shape[0])
    ref_mcep = ref_mcep[:min_len]
    syn_mcep = syn_mcep[:min_len]

    # MCD = (10/ln10) * sqrt(2 * Σ(mcep_diff)²) / T
    diff = ref_mcep - syn_mcep
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(diff ** 2))
    return float(mcd)


def compute_f0_rmse(ref_f0, syn_f0, voiced_only=True):
    """
    F0 Root Mean Square Error (Hz)
    论文: 仅计算 voiced 帧
    """
    if voiced_only:
        mask = (ref_f0 > 0) & (syn_f0 > 0)
    else:
        mask = np.ones_like(ref_f0, dtype=bool)

    if mask.sum() == 0:
        return float('inf')

    diff = ref_f0[mask] - syn_f0[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_vuv_f1(ref_f0, syn_f0):
    """
    Voiced/Unvoiced F1 score
    论文: V/UV accuracy
    """
    ref_voiced = (ref_f0 > 0).astype(int)
    syn_voiced = (syn_f0 > 0).astype(int)

    # 对齐
    min_len = min(len(ref_voiced), len(syn_voiced))
    ref_voiced = ref_voiced[:min_len]
    syn_voiced = syn_voiced[:min_len]

    tp = ((ref_voiced == 1) & (syn_voiced == 1)).sum()
    fp = ((ref_voiced == 0) & (syn_voiced == 1)).sum()
    fn = ((ref_voiced == 1) & (syn_voiced == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def evaluate_all(ref_wav, syn_wav, ref_f0, syn_f0, sr=24000):
    """计算所有指标"""
    mcd = compute_mcd(ref_wav, syn_wav, sr)
    f0_rmse = compute_f0_rmse(ref_f0, syn_f0)
    vuv_f1 = compute_vuv_f1(ref_f0, syn_f0)
    return {'MCD': mcd, 'F0-RMSE': f0_rmse, 'V/UV_F1': vuv_f1}