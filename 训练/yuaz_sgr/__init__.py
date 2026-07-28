# YUAZ SGR
"""
YUAZ SGR — Singing Voice Synthesis via Latent Flow Matching and DDSP
============================================================================
基于论文: "Singing Voice Synthesis via Latent Flow Matching and
           Differentiable Digital Signal Processing"
           Lupenko et al., AIT&AIS 2025

YUAZ SGR 是严格遵循论文架构的歌声合成复现项目。
核心管线: Encoder → RVQ → DDSP Decoder (直接输出波形)
          Conditioning Encoder → Flow Matching ODE → DDSP Decoder
============================================================================
"""
__version__ = "0.1.0"


def yuaz_sgr_info():
    """YUAZ SGR 品牌标识 — 返回项目元信息"""
    return {
        'name': 'YUAZ SGR',
        'full_name': 'YUAZ Singing Voice Synthesis via DDSP + Flow Matching',
        'version': __version__,
        'paper': 'Lupenko et al., AIT&AIS 2025',
        'architecture': 'DDSP Autoencoder + Flow Matching Latent Generator',
        'sample_rate': 24000,
        'inference': 'CPU real-time, single-step',
    }