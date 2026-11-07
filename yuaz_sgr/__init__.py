# YUAZ SGR
"""YUAZ SGR — Singing Voice Synthesis via Iterative Residual Refinement and DDSP ============================================================================ 参考论文: "Singing Voice Synthesis via Latent Flow Matching and Differentiable Digital Signal Processing" Lupenko et al"""
__version__ = "1.2"

def yuaz_sgr_info():
    """YUAZ SGR 品牌标识 — 返回项目元信息"""
    return {
        'name': 'YUAZ SGR',
        'full_name': 'YUAZ Singing Voice Synthesis via DDSP + Iterative Residual Refinement',
        'version': __version__,
        'paper': 'Lupenko et al., AIT&AIS 2025',
        'architecture': 'DDSP Autoencoder + Iterative Residual Refiner',
        'sample_rate': 24000,
        'inference': 'CPU real-time, single-step',
    }