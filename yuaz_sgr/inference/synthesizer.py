# YUAZ SGR — Inference Synthesizer
"""
synthesizer.py — 推理核心 API

用法:
    from yuaz_sgr import YuazSinger
    singer = YuazSinger("checkpoint.pt", device="cuda")
    wav = singer.sing(phoneme_ids, durations, f0_curve)
    singer.save("output.wav", wav)

或直接从 MIDI + 歌词:
    wav = singer.sing_from_midi("song.mid", "你好啊", "output.wav")
"""

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

# 项目根目录 (yuaz_sgr_inference/)
_PACKAGE_ROOT = Path(__file__).parent.parent

# 中文拼音音节 → OpenCPOP 音素列表
PINYIN_TO_PHONEMES = {
    # 声母 (单音素)
    "b": ["b"], "p": ["p"], "m": ["m"], "f": ["f"],
    "d": ["d"], "t": ["t"], "n": ["n"], "l": ["l"],
    "g": ["g"], "k": ["k"], "h": ["h"],
    "j": ["j"], "q": ["q"], "x": ["x"],
    "zh": ["zh"], "ch": ["ch"], "sh": ["sh"], "r": ["r"],
    "z": ["z"], "c": ["c"], "s": ["s"],
    "w": ["w"], "y": ["y"],
    # 单韵母 (单音素)
    "a": ["a"], "o": ["o"], "e": ["e"],
    "i": ["i"], "u": ["u"], "v": ["v"],
    "er": ["er"],
    # 复韵母
    "ai": ["ai"], "ei": ["ei"], "ao": ["ao"], "ou": ["ou"],
    "an": ["an"], "en": ["en"], "ang": ["ang"], "eng": ["eng"], "ong": ["ong"],
    "ia": ["ia"], "ie": ["ie"], "iao": ["iao"], "iu": ["iu"],
    "ian": ["ian"], "in": ["in"], "iang": ["iang"], "ing": ["ing"], "iong": ["iong"],
    "ua": ["ua"], "uo": ["uo"], "uai": ["uai"], "ui": ["ui"],
    "uan": ["uan"], "un": ["un"], "uang": ["uang"],
    "ve": ["ve"], "van": ["van"], "vn": ["vn"],
    # 完整拼音 (声母+韵母 → 两个音素)
    "ba": ["b", "a"], "bo": ["b", "o"], "bi": ["b", "i"], "bu": ["b", "u"],
    "bai": ["b", "ai"], "bei": ["b", "ei"], "bao": ["b", "ao"],
    "ban": ["b", "an"], "ben": ["b", "en"], "bang": ["b", "ang"], "beng": ["b", "eng"],
    "bian": ["b", "ian"], "biao": ["b", "iao"], "bie": ["b", "ie"],
    "bin": ["b", "in"], "bing": ["b", "ing"],
    "pa": ["p", "a"], "po": ["p", "o"], "pi": ["p", "i"], "pu": ["p", "u"],
    "pai": ["p", "ai"], "pei": ["p", "ei"], "pao": ["p", "ao"], "pou": ["p", "ou"],
    "pan": ["p", "an"], "pen": ["p", "en"], "pang": ["p", "ang"], "peng": ["p", "eng"],
    "pian": ["p", "ian"], "piao": ["p", "iao"], "pie": ["p", "ie"],
    "pin": ["p", "in"], "ping": ["p", "ing"],
    "ma": ["m", "a"], "mo": ["m", "o"], "me": ["m", "e"], "mi": ["m", "i"], "mu": ["m", "u"],
    "mai": ["m", "ai"], "mei": ["m", "ei"], "mao": ["m", "ao"], "mou": ["m", "ou"],
    "man": ["m", "an"], "men": ["m", "en"], "mang": ["m", "ang"], "meng": ["m", "eng"],
    "mian": ["m", "ian"], "miao": ["m", "iao"], "mie": ["m", "ie"], "miu": ["m", "iu"],
    "min": ["m", "in"], "ming": ["m", "ing"],
    "fa": ["f", "a"], "fo": ["f", "o"], "fu": ["f", "u"],
    "fei": ["f", "ei"], "fou": ["f", "ou"],
    "fan": ["f", "an"], "fen": ["f", "en"], "fang": ["f", "ang"], "feng": ["f", "eng"],
    "da": ["d", "a"], "de": ["d", "e"], "di": ["d", "i"], "du": ["d", "u"],
    "dai": ["d", "ai"], "dei": ["d", "ei"], "dao": ["d", "ao"], "dou": ["d", "ou"],
    "dan": ["d", "an"], "den": ["d", "en"], "dang": ["d", "ang"], "deng": ["d", "eng"], "dong": ["d", "ong"],
    "dian": ["d", "ian"], "diao": ["d", "iao"], "die": ["d", "ie"], "diu": ["d", "iu"],
    "duan": ["d", "uan"], "dui": ["d", "ui"], "dun": ["d", "un"], "duo": ["d", "uo"],
    "ta": ["t", "a"], "te": ["t", "e"], "ti": ["t", "i"], "tu": ["t", "u"],
    "tai": ["t", "ai"], "tao": ["t", "ao"], "tou": ["t", "ou"],
    "tan": ["t", "an"], "tang": ["t", "ang"], "teng": ["t", "eng"], "tong": ["t", "ong"],
    "tian": ["t", "ian"], "tiao": ["t", "iao"], "tie": ["t", "ie"],
    "tuan": ["t", "uan"], "tui": ["t", "ui"], "tun": ["t", "un"], "tuo": ["t", "uo"],
    "na": ["n", "a"], "ne": ["n", "e"], "ni": ["n", "i"], "nu": ["n", "u"], "nv": ["n", "v"],
    "nai": ["n", "ai"], "nei": ["n", "ei"], "nao": ["n", "ao"], "nou": ["n", "ou"],
    "nan": ["n", "an"], "nen": ["n", "en"], "nang": ["n", "ang"], "neng": ["n", "eng"], "nong": ["n", "ong"],
    "nian": ["n", "ian"], "niang": ["n", "iang"], "niao": ["n", "iao"], "nie": ["n", "ie"], "niu": ["n", "iu"],
    "nin": ["n", "in"], "ning": ["n", "ing"],
    "nuan": ["n", "uan"], "nuo": ["n", "uo"], "nve": ["n", "ve"],
    "la": ["l", "a"], "le": ["l", "e"], "li": ["l", "i"], "lu": ["l", "u"], "lv": ["l", "v"],
    "lai": ["l", "ai"], "lei": ["l", "ei"], "lao": ["l", "ao"], "lou": ["l", "ou"],
    "lan": ["l", "an"], "lang": ["l", "ang"], "leng": ["l", "eng"], "long": ["l", "ong"],
    "lian": ["l", "ian"], "liang": ["l", "iang"], "liao": ["l", "iao"], "lie": ["l", "ie"], "liu": ["l", "iu"],
    "lin": ["l", "in"], "ling": ["l", "ing"],
    "luan": ["l", "uan"], "lun": ["l", "un"], "luo": ["l", "uo"], "lve": ["l", "ve"],
    "ga": ["g", "a"], "ge": ["g", "e"], "gu": ["g", "u"],
    "gai": ["g", "ai"], "gei": ["g", "ei"], "gao": ["g", "ao"], "gou": ["g", "ou"],
    "gan": ["g", "an"], "gen": ["g", "en"], "gang": ["g", "ang"], "geng": ["g", "eng"], "gong": ["g", "ong"],
    "gua": ["g", "ua"], "guai": ["g", "uai"], "guan": ["g", "uan"], "guang": ["g", "uang"],
    "gui": ["g", "ui"], "gun": ["g", "un"], "guo": ["g", "uo"],
    "ka": ["k", "a"], "ke": ["k", "e"], "ku": ["k", "u"],
    "kai": ["k", "ai"], "kao": ["k", "ao"], "kou": ["k", "ou"],
    "kan": ["k", "an"], "ken": ["k", "en"], "kang": ["k", "ang"], "keng": ["k", "eng"], "kong": ["k", "ong"],
    "kua": ["k", "ua"], "kuai": ["k", "uai"], "kuan": ["k", "uan"], "kuang": ["k", "uang"],
    "kui": ["k", "ui"], "kun": ["k", "un"], "kuo": ["k", "uo"],
    "ha": ["h", "a"], "he": ["h", "e"], "hu": ["h", "u"],
    "hai": ["h", "ai"], "hei": ["h", "ei"], "hao": ["h", "ao"], "hou": ["h", "ou"],
    "han": ["h", "an"], "hen": ["h", "en"], "hang": ["h", "ang"], "heng": ["h", "eng"], "hong": ["h", "ong"],
    "hua": ["h", "ua"], "huai": ["h", "uai"], "huan": ["h", "uan"], "huang": ["h", "uang"],
    "hui": ["h", "ui"], "hun": ["h", "un"], "huo": ["h", "uo"],
    "ji": ["j", "i"], "ju": ["j", "v"],
    "jia": ["j", "ia"], "jie": ["j", "ie"], "jiao": ["j", "iao"], "jiu": ["j", "iu"],
    "jian": ["j", "ian"], "jin": ["j", "in"], "jiang": ["j", "iang"], "jing": ["j", "ing"], "jiong": ["j", "iong"],
    "juan": ["j", "van"], "jue": ["j", "ve"], "jun": ["j", "vn"],
    "qi": ["q", "i"], "qu": ["q", "v"],
    "qia": ["q", "ia"], "qie": ["q", "ie"], "qiao": ["q", "iao"], "qiu": ["q", "iu"],
    "qian": ["q", "ian"], "qin": ["q", "in"], "qiang": ["q", "iang"], "qing": ["q", "ing"], "qiong": ["q", "iong"],
    "quan": ["q", "van"], "que": ["q", "ve"], "qun": ["q", "vn"],
    "xi": ["x", "i"], "xu": ["x", "v"],
    "xia": ["x", "ia"], "xie": ["x", "ie"], "xiao": ["x", "iao"], "xiu": ["x", "iu"],
    "xian": ["x", "ian"], "xin": ["x", "in"], "xiang": ["x", "iang"], "xing": ["x", "ing"], "xiong": ["x", "iong"],
    "xuan": ["x", "van"], "xue": ["x", "ve"], "xun": ["x", "vn"],
    "zhi": ["zh", "i"], "zha": ["zh", "a"], "zhe": ["zh", "e"], "zhu": ["zh", "u"],
    "zhai": ["zh", "ai"], "zhao": ["zh", "ao"], "zhou": ["zh", "ou"],
    "zhan": ["zh", "an"], "zhen": ["zh", "en"], "zhang": ["zh", "ang"], "zheng": ["zh", "eng"], "zhong": ["zh", "ong"],
    "zhua": ["zh", "ua"], "zhuai": ["zh", "uai"], "zhuan": ["zh", "uan"], "zhuang": ["zh", "uang"],
    "zhui": ["zh", "ui"], "zhun": ["zh", "un"], "zhuo": ["zh", "uo"],
    "chi": ["ch", "i"], "cha": ["ch", "a"], "che": ["ch", "e"], "chu": ["ch", "u"],
    "chai": ["ch", "ai"], "chao": ["ch", "ao"], "chou": ["ch", "ou"],
    "chan": ["ch", "an"], "chen": ["ch", "en"], "chang": ["ch", "ang"], "cheng": ["ch", "eng"], "chong": ["ch", "ong"],
    "chua": ["ch", "ua"], "chuai": ["ch", "uai"], "chuan": ["ch", "uan"], "chuang": ["ch", "uang"],
    "chui": ["ch", "ui"], "chun": ["ch", "un"], "chuo": ["ch", "uo"],
    "shi": ["sh", "i"], "sha": ["sh", "a"], "she": ["sh", "e"], "shu": ["sh", "u"],
    "shai": ["sh", "ai"], "shei": ["sh", "ei"], "shao": ["sh", "ao"], "shou": ["sh", "ou"],
    "shan": ["sh", "an"], "shen": ["sh", "en"], "shang": ["sh", "ang"], "sheng": ["sh", "eng"],
    "shua": ["sh", "ua"], "shuai": ["sh", "uai"], "shuan": ["sh", "uan"], "shuang": ["sh", "uang"],
    "shui": ["sh", "ui"], "shun": ["sh", "un"], "shuo": ["sh", "uo"],
    "ri": ["r", "i"], "re": ["r", "e"], "ru": ["r", "u"],
    "rao": ["r", "ao"], "rou": ["r", "ou"],
    "ran": ["r", "an"], "ren": ["r", "en"], "rang": ["r", "ang"], "reng": ["r", "eng"], "rong": ["r", "ong"],
    "rua": ["r", "ua"], "ruan": ["r", "uan"], "rui": ["r", "ui"], "run": ["r", "un"], "ruo": ["r", "uo"],
    "zi": ["z", "i"], "za": ["z", "a"], "ze": ["z", "e"], "zu": ["z", "u"],
    "zai": ["z", "ai"], "zei": ["z", "ei"], "zao": ["z", "ao"], "zou": ["z", "ou"],
    "zan": ["z", "an"], "zen": ["z", "en"], "zang": ["z", "ang"], "zeng": ["z", "eng"], "zong": ["z", "ong"],
    "zuan": ["z", "uan"], "zui": ["z", "ui"], "zun": ["z", "un"], "zuo": ["z", "uo"],
    "ci": ["c", "i"], "ca": ["c", "a"], "ce": ["c", "e"], "cu": ["c", "u"],
    "cai": ["c", "ai"], "cao": ["c", "ao"], "cou": ["c", "ou"],
    "can": ["c", "an"], "cen": ["c", "en"], "cang": ["c", "ang"], "ceng": ["c", "eng"], "cong": ["c", "ong"],
    "cuan": ["c", "uan"], "cui": ["c", "ui"], "cun": ["c", "un"], "cuo": ["c", "uo"],
    "si": ["s", "i"], "sa": ["s", "a"], "se": ["s", "e"], "su": ["s", "u"],
    "sai": ["s", "ai"], "sao": ["s", "ao"], "sou": ["s", "ou"],
    "san": ["s", "an"], "sen": ["s", "en"], "sang": ["s", "ang"], "seng": ["s", "eng"], "song": ["s", "ong"],
    "suan": ["s", "uan"], "sui": ["s", "ui"], "sun": ["s", "un"], "suo": ["s", "uo"],
    "ya": ["y", "a"], "yo": ["y", "o"], "ye": ["y", "e"], "yi": ["y", "i"], "yu": ["y", "v"],
    "yao": ["y", "ao"], "you": ["y", "ou"],
    "yan": ["y", "an"], "yin": ["y", "in"], "yang": ["y", "ang"], "ying": ["y", "ing"], "yong": ["y", "ong"],
    "yuan": ["y", "van"], "yue": ["y", "ve"], "yun": ["y", "vn"],
    "wa": ["w", "a"], "wo": ["w", "o"], "wu": ["w", "u"],
    "wai": ["w", "ai"], "wei": ["w", "ei"],
    "wan": ["w", "an"], "wen": ["w", "en"], "wang": ["w", "ang"], "weng": ["w", "eng"],
}

# 清辅音声母：训练标注在这些帧上 F0=0
UNVOICED_INITIALS = {
    "b", "p", "f", "d", "t", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "z", "c", "s",
}

# 声母时长表（帧, 93.75Hz）：训练集逐音素中位数
CONSONANT_FRAME_TABLE = {
    "b": 5, "d": 5, "g": 5,
    "t": 8, "k": 8, "p": 8,
    "zh": 8, "j": 9, "z": 10,
    "h": 11, "c": 12, "ch": 12, "q": 13,
    "f": 14, "s": 15, "sh": 16, "x": 16,
    "m": 10, "n": 9, "l": 8, "r": 9, "w": 8, "y": 9,
}

# 爆破音：真实爆破段只有 10~25ms，给太长会变成"呼"而不是"啪"
PLOSIVES = {"b", "d", "g", "p", "t", "k"}

# 爆破音默认时长（秒）：实测 32ms 听感最干脆，且不吞音
DEFAULT_PLOSIVE_SECONDS = 0.032
DEFAULT_CONS_SECONDS = {p: DEFAULT_PLOSIVE_SECONDS for p in PLOSIVES}

def normalize_pinyin(syl):
    """拼音归一化: 去声调数字/空格, ü→v, 保留字母"""
    s = str(syl).strip().lower().replace("ü", "v")
    s = "".join(ch for ch in s if "a" <= ch <= "z")
    return s

def midi_to_f0(midi_pitch):
    """MIDI pitch → F0 (Hz). A4=440Hz, MIDI=69"""
    return 440.0 * (2 ** ((np.asarray(midi_pitch, dtype=np.float64) - 69) / 12))

# 清浊先验：清辅音/塞擦音/静音走噪声通路，其余走谐波通路
UNVOICED_PHONEMES = {"p", "t", "k", "f", "s", "sh", "x", "h",
                     "c", "ch", "q", "SP", "AP"}

def build_voicing_mask(i2p, phoneme_ids, durations, total_frames=None, smooth_frames=1):
    """清浊 mask: 1=浊音/谐波, 0=清音/噪声。i2p 为 {音素ID: 名称}。"""
    frames = []
    for pid, d in zip(np.atleast_1d(phoneme_ids), np.atleast_1d(durations)):
        name = i2p.get(str(int(pid)), "")
        val = 0.0 if name in UNVOICED_PHONEMES else 1.0
        frames.append(np.full(max(1, int(d)), val, dtype=np.float32))
    mask = np.concatenate(frames) if frames else np.ones(1, dtype=np.float32)
    if total_frames is not None and len(mask) != total_frames:
        idx = np.clip((np.arange(total_frames) * len(mask) / total_frames).astype(int),
                      0, len(mask) - 1)
        mask = mask[idx]
    if smooth_frames and smooth_frames > 0 and len(mask) > 1:
        k = 2 * int(smooth_frames) + 1
        # 用 hanning(k+2) 去掉两端零点
        win = np.hanning(k + 2)[1:-1].astype(np.float32)
        s = win.sum()
        win = win / s if s > 0 else np.ones(k, dtype=np.float32) / k
        mask = np.convolve(mask, win, mode='same').astype(np.float32)
    return mask

def apply_portamento(f0_curve, note_starts, note_lengths, frames):
    """音符边界做线性滑音，避免 F0 硬跳变。frames 为滑音长度，<=0 不处理。"""
    if frames <= 0 or not note_starts:
        return f0_curve
    f0 = np.array(f0_curve, dtype=np.float32, copy=True)
    L = len(f0)
    for k in range(1, len(note_starts)):
        s = int(note_starts[k])
        if s <= 0 or s >= L:
            continue
        nl = int(note_lengths[k]) if k < len(note_lengths) else 0
        a, b = f0[s - 1], f0[s]
        if a <= 0 or b <= 0 or abs(a - b) < 1e-3:
            continue
        n = int(min(frames, s, nl))
        if n <= 0:
            continue
        i0 = s - n
        f0[i0:s] = np.linspace(f0[i0], b, n)
    return f0

def natural_f0_curve(f0_curve, frame_rate, vibrato_cents=25.0, vibrato_rate=5.5,
                     min_hold_sec=0.32, smooth_frames=3):
    """按谱面生成自然 F0：轻度平滑 + 长音加颤音（确定性，不调用 F0Predictor）。"""
    f0 = np.asarray(f0_curve, dtype=np.float64).copy()
    if len(f0) == 0:
        return f0
    voiced = f0 > 0
    if smooth_frames > 1:
        k = int(smooth_frames)
        pad = k // 2
        v = voiced.astype(np.float64)
        num = np.convolve(f0 * v, np.ones(k) / k, mode='same')
        den = np.convolve(v, np.ones(k) / k, mode='same')
        sm = num / np.clip(den, 1e-3, None)
        f0 = np.where(voiced, sm, f0)
    if vibrato_cents <= 0:
        return f0
    # 找连续浊音段
    idx = 0
    T = len(f0)
    while idx < T:
        if not voiced[idx]:
            idx += 1
            continue
        j = idx
        while j < T and voiced[j]:
            j += 1
        seg_len = j - idx
        if seg_len / frame_rate >= min_hold_sec:
            t = np.arange(seg_len) / frame_rate
            ramp_in = np.clip(np.arange(seg_len) / (0.12 * frame_rate), 0, 1)
            ramp_out = np.clip((seg_len - np.arange(seg_len)) / (0.12 * frame_rate), 0, 1)
            w = np.minimum(ramp_in, ramp_out)
            cents = vibrato_cents * np.sin(2 * np.pi * vibrato_rate * t) * w
            f0[idx:j] = f0[idx:j] * (2.0 ** (cents / 1200.0))
        idx = j
    return f0

def apply_consonant_gain(wav, i2p, phoneme_ids, durations, target_ratio=0.5,
                         max_gain=6.0, hop=256, ramp_frames=2):
    """把清辅音帧的 RMS 拉到 target_ratio × 元音 RMS（模型输出的辅音常偏小）。"""
    wav = np.asarray(wav, dtype=np.float32)
    total = int(np.sum(durations))
    n = min(total, len(wav) // hop)
    if n <= 1 or target_ratio <= 0:
        return wav
    x = wav[: n * hop].reshape(n, hop).astype(np.float64)
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-12)

    voiced, cons = [], []
    fi = 0
    for pid, d in zip(np.atleast_1d(phoneme_ids), np.atleast_1d(durations)):
        d = int(d)
        name = i2p.get(str(int(pid)), "")
        sl = rms[fi:fi + d]
        if name in UNVOICED_INITIALS:
            cons.append((fi, d, float(np.mean(sl)) if len(sl) else 0.0))
        elif name not in ("SP", "AP", "?"):
            if len(sl):
                voiced.append(float(np.mean(sl)))
        fi += d
    if not cons or not voiced:
        return wav
    v_rms = float(np.median(voiced))
    gains = np.ones(n, dtype=np.float64)
    for start, d, c_rms in cons:
        if c_rms <= 1e-8:
            continue
        g = float(np.clip(target_ratio * v_rms / c_rms, 1.0, max_gain))
        gains[start:start + d] = g
    if ramp_frames > 0 and n > 2:
        k = np.hanning(2 * ramp_frames + 1)
        k = k / k.sum()
        gains = np.convolve(gains, k, mode='same')
    out = wav.copy()
    g_samp = np.repeat(gains, hop).astype(np.float32)
    out[: n * hop] = (wav[: n * hop] * g_samp).astype(np.float32)
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out *= 0.99 / peak
    return out

def apply_spectral_tilt(wav, sample_rate=24000, tilt_db=0.0, pivot_hz=2000.0):
    """高频搁架：pivot_hz 以上每倍频程加 tilt_db dB，0 表示关闭。"""
    if not tilt_db:
        return np.asarray(wav, dtype=np.float32)
    wav = np.asarray(wav, dtype=np.float64)
    n = len(wav)
    if n < 16:
        return wav.astype(np.float32)
    spec = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    gain_db = tilt_db * np.clip(np.log2(np.maximum(freqs, pivot_hz) / pivot_hz), 0.0, None)
    spec = spec * (10.0 ** (gain_db / 20.0))
    out = np.fft.irfft(spec, n=n).astype(np.float32)
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out *= 0.99 / peak
    return out

class YuazSinger:
    """YUAZ SGR 推理器

    参数:
        checkpoint_path: 模型检查点 .pt 文件路径
        config_dir: 配置文件目录, 包含 base.yaml 和 model.yaml
        device: "cpu" 或 "cuda"

    示例:
        singer = YuazSinger("checkpoint_200000.pt", device="cuda")
        wav = singer.sing(phoneme_ids, durations, f0_curve, use_predicted_f0=True)
        singer.save("output.wav", wav)
    """

    def __init__(self, checkpoint_path, config_dir=None, device="cpu", use_silence_gate=False):
        self.device = device

        # 配置
        if config_dir is None:
            config_dir = _PACKAGE_ROOT / "yuaz_sgr" / "assets"
        config_dir = Path(config_dir)

        with open(config_dir / "base.yaml", "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        with open(config_dir / "model.yaml", "r", encoding="utf-8") as f:
            model_cfg_file = yaml.safe_load(f)

        config = {**base_cfg, **model_cfg_file}
        if "encoder" not in config:
            config["encoder"] = {}
        config["encoder"].setdefault("hidden_dim", 512)
        config["encoder"].setdefault("latent_dim", 128)
        config["encoder"].setdefault("n_mfcc", base_cfg.get("n_mfcc", 30))
        config["encoder"].setdefault("n_fft", base_cfg.get("n_fft", 1024))
        config["encoder"].setdefault("hop_length", base_cfg.get("hop_length", 256))
        config["encoder"].setdefault("sample_rate", base_cfg.get("sample_rate", 24000))

        self.config = config

        phoneme_map_path = config_dir / "phoneme_map.json"
        with open(phoneme_map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.p2i = mapping["p2i"]
        self.i2p = mapping["i2p"]

        # 辅音时长统计表（训练数据统计，可选）
        self.cons_dur_table = None
        _tbl_path = os.path.join(os.path.dirname(__file__), "assets", "cons_dur_table.json")
        if os.path.exists(_tbl_path):
            try:
                with open(_tbl_path, "r", encoding="utf-8") as f:
                    self.cons_dur_table = json.load(f)
            except Exception:
                self.cons_dur_table = None

        self.sample_rate = config.get("sample_rate", 24000)
        self.hop_length = config.get("hop_length", 256)
        self.frame_rate = self.sample_rate / self.hop_length  # 93.75 Hz

        from .models.svs_model import SVSModel

        # 先读 checkpoint 标志位，再建模型
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt.get("ema") or ckpt["model"]
        else:
            sd = ckpt
        flags = ckpt.get("flags", {}) if isinstance(ckpt, dict) else {}
        if flags:
            config.setdefault("conditioning", {})
            config.setdefault("encoder", {})
            if "decoder_cond" in flags:
                config["decoder_cond"] = bool(flags["decoder_cond"])
            if "use_energy" in flags:
                config["conditioning"]["use_energy"] = bool(flags["use_energy"])
            if "f0_predictor" in flags:
                config["conditioning"]["f0_predictor"] = str(flags["f0_predictor"])
            if "encoder_feature" in flags:
                config["encoder"]["feature"] = str(flags["encoder_feature"])
            config.setdefault("ddsp", {})
            if "dec_mels" in flags:
                config["ddsp"]["n_mels"] = int(flags["dec_mels"])
            if "dec_res_rank" in flags:
                config["ddsp"]["spectral_residual_rank"] = int(flags["dec_res_rank"])
            if "irr_steps" in flags:
                config.setdefault("generator", {})["irr_steps"] = int(flags["irr_steps"])
            print(f"  模型标志位: {flags}")

        self.model = SVSModel(config)
        # 容错加载：缺权重只告警
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        _buf = ('mel_basis', 'mel_pinv', 'hann_window', 'ema_size', 'ema_sum')
        missing = [k for k in missing if not k.endswith(_buf)]
        if missing:
            print(f"  [警告] {len(missing)} 个权重缺失 (随机初始化): {missing[:5]}"
                  f"{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  [信息] 忽略 checkpoint 中 {len(unexpected)} 个多余权重")
        self.model.eval()
        self.model.to(device)

        # 默认关闭帧级静音门控
        if hasattr(self.model.ddsp_decoder, "silence_gate"):
            self.model.ddsp_decoder.silence_gate = bool(use_silence_gate)

        step = ckpt.get("step", "unknown")
        print(f"YuazSinger 加载完成: step={step}, device={device}")

    @torch.no_grad()
    def sing(self, phoneme_ids, durations, f0_curve,
             note_lengths=None, predict_durations=False,
             use_predicted_f0=True, f0_mode=None, vibrato_cents=25.0, vibrato_rate=5.5,
             n_steps=8, irr_steps=None,
             rvq_residual_alpha=0.2, max_chunk_frames=2048,
             chunk_overlap_frames=96, energy=None,
             mu_refine=False, mu_f0=None,
             voicing_prior=False, voicing_harmonic=0.15, voicing_aperiodic=0.60,
             voicing_smooth=1, consonant_gain=0.0, consonant_max_gain=6.0,
             spectral_tilt_db=0.0):
        """音素 + 时长 + F0 → 波形

        phoneme_ids/durations: (P,)；f0_curve: (T_frames,) 帧级 Hz；
        irr_steps 为 IRR 步数（默认 4，越大细节越多）。
        返回 shape (T_samples,) float32 24kHz。
        """
        if irr_steps is not None:
            self.model.irr_steps = int(irr_steps)

        # 转换输入
        ph  = torch.LongTensor(np.asarray(phoneme_ids)).unsqueeze(0).to(self.device)  # (1, P)
        dur = torch.LongTensor(np.asarray(durations)).unsqueeze(0).to(self.device)     # (1, P)
        f0  = torch.FloatTensor(np.asarray(f0_curve)).unsqueeze(0).to(self.device)     # (1, T_f)

        nl = None
        if note_lengths is not None:
            nl = torch.LongTensor(np.asarray(note_lengths)).unsqueeze(0).to(self.device)  # (1, N)

        en = None
        if energy is not None:
            en = torch.FloatTensor(np.asarray(energy)).unsqueeze(0).to(self.device)

        muf = None
        if mu_f0 is not None:
            muf = torch.FloatTensor(np.asarray(mu_f0)).unsqueeze(0).to(self.device)

        # 预测 F0 时默认两遍 μ：用预测出的（平滑）F0 重建条件 μ，贴合训练分布
        if use_predicted_f0 and muf is None and not mu_refine:
            mu_refine = True

        if f0_mode is None:
            f0_mode = 'natural' if use_predicted_f0 else 'input'
        if f0_mode == 'natural':
            import numpy as _np
            _f0 = _np.asarray(f0_curve, dtype=_np.float64).copy()
            _f0 = natural_f0_curve(_f0, self.frame_rate,
                                   vibrato_cents=vibrato_cents,
                                   vibrato_rate=vibrato_rate)
            f0 = torch.FloatTensor(_f0).unsqueeze(0).to(self.device)
            use_predicted_f0 = False

        voi = None
        if voicing_prior:
            mask = build_voicing_mask(self.i2p, np.asarray(phoneme_ids),
                                      np.asarray(durations),
                                      smooth_frames=voicing_smooth)
            voi = torch.FloatTensor(mask).unsqueeze(0).to(self.device)  # (1, T)

        wav, _, _, _, _, _ = self.model.forward_synthesis(
            ph, dur, f0,
            note_lengths=nl,
            predict_durations=predict_durations,
            use_predicted_f0=use_predicted_f0,
            n_steps=n_steps,
            energy=en,
            mu_refine=mu_refine,
            mu_f0=muf,
            rvq_residual_alpha=rvq_residual_alpha,
            max_chunk_frames=max_chunk_frames,
            chunk_overlap_frames=chunk_overlap_frames,
            voicing=voi,
            voicing_harmonic=voicing_harmonic,
            voicing_aperiodic=voicing_aperiodic,
        )
        wav_np = wav[0, 0].cpu().numpy().astype(np.float32)
        if spectral_tilt_db:
            wav_np = apply_spectral_tilt(wav_np, self.sample_rate, spectral_tilt_db)
        if consonant_gain and consonant_gain > 0 and not predict_durations:
            wav_np = apply_consonant_gain(
                wav_np, self.i2p, np.asarray(phoneme_ids), np.asarray(durations),
                target_ratio=consonant_gain, max_gain=consonant_max_gain,
                hop=self.hop_length)
        return wav_np

    @torch.no_grad()
    def sing_from_midi(self, midi_path, lyrics=None, output_path=None,
                       predict_durations=False, use_predicted_f0=True,
                       f0_mode=None, vibrato_cents=25.0, vibrato_rate=5.5,
                       n_steps=8, irr_steps=None, consonant_frames=12, consonant_ratio=0.0,
                       plosive_frames=0, cons_dur_table=None, cons_seconds=None,
                       mute_unvoiced_f0=True,
                       f0_portamento_frames=0, rvq_residual_alpha=0.2,
                       max_chunk_frames=2048, chunk_overlap_frames=96,
                       voicing_prior=False, consonant_gain=0.5, spectral_tilt_db=0.0):
        """从 MIDI + 拼音歌词合成

        lyrics 用空格分隔，数量与音符数不一致时会截断/补 "a"。
        cons_dur_table / cons_seconds 控制辅音时长，默认读 assets 里的统计表。
        返回 (T_samples,) 的 24kHz 波形。
        """
        # 解析 MIDI
        notes = self._parse_midi(midi_path)
        if not notes:
            raise ValueError(f"MIDI 文件中没有音符: {midi_path}")

        # 处理歌词
        if lyrics is not None:
            syllables = lyrics.strip().split()
            # 音节数与音符数对齐
            if len(syllables) != len(notes):
                warnings.warn(
                    f"歌词音节数 ({len(syllables)}) 与 MIDI 音符数 ({len(notes)}) 不匹配, "
                    f"将截断或填充"
                )
                while len(syllables) < len(notes):
                    syllables.append("a")
                syllables = syllables[:len(notes)]
        else:
            # 无歌词 → 全部用 "a"
            syllables = ["a"] * len(notes)

        # 构建音素序列、时长、F0
        if cons_dur_table is None:
            cons_dur_table = getattr(self, 'cons_dur_table', None)
        if cons_seconds is None:
            cons_seconds = DEFAULT_CONS_SECONDS
        phoneme_ids, durations, f0_curve, note_lengths, note_starts = self._build_score(
            notes, syllables, consonant_frames=consonant_frames,
            consonant_ratio=consonant_ratio,
            mute_unvoiced_f0=mute_unvoiced_f0,
            plosive_frames=plosive_frames,
            cons_dur_table=cons_dur_table,
            cons_seconds=cons_seconds,
        )

        # 音符边界滑音
        f0_curve = apply_portamento(f0_curve, note_starts, note_lengths,
                                    f0_portamento_frames)

        print(f"MIDI: {len(notes)} 音符, {len(phoneme_ids)} 音素, "
              f"F0 共 {len(f0_curve)} 帧 ({len(f0_curve)/self.frame_rate:.1f}s)")

        # 合成
        wav = self.sing(
            phoneme_ids, durations, f0_curve,
            note_lengths=note_lengths,
            predict_durations=predict_durations,
            use_predicted_f0=use_predicted_f0,
            f0_mode=f0_mode, vibrato_cents=vibrato_cents, vibrato_rate=vibrato_rate,
            n_steps=n_steps, irr_steps=irr_steps,
            rvq_residual_alpha=rvq_residual_alpha,
            max_chunk_frames=max_chunk_frames,
            chunk_overlap_frames=chunk_overlap_frames,
            voicing_prior=voicing_prior,
            consonant_gain=consonant_gain,
            spectral_tilt_db=spectral_tilt_db,
        )

        if output_path:
            self.save(output_path, wav)
        return wav

    @torch.no_grad()
    def sing_from_phonemes(self, phoneme_strs, durations, f0_curve,
                           note_lengths=None, predict_durations=False,
                           use_predicted_f0=True, n_steps=8, irr_steps=None,
                           rvq_residual_alpha=0.2, max_chunk_frames=2048,
                           chunk_overlap_frames=96, energy=None,
                           mu_refine=False, mu_f0=None,
                           voicing_prior=False, voicing_smooth=1):
        """从音素字符串合成，参数同 sing()。"""
        ph_ids = [self.p2i.get(p, 0) for p in phoneme_strs]
        return self.sing(
            ph_ids, durations, f0_curve,
            note_lengths=note_lengths,
            predict_durations=predict_durations,
            use_predicted_f0=use_predicted_f0,
            n_steps=n_steps, irr_steps=irr_steps,
            rvq_residual_alpha=rvq_residual_alpha,
            max_chunk_frames=max_chunk_frames,
            chunk_overlap_frames=chunk_overlap_frames,
            energy=energy,
            mu_refine=mu_refine,
            mu_f0=mu_f0,
            voicing_prior=voicing_prior,
            voicing_smooth=voicing_smooth,
        )

    def _parse_midi(self, midi_path):
        """解析 MIDI → [(pitch, start_sec, end_sec), ...]"""
        try:
            import pretty_midi
        except ImportError:
            raise ImportError("请安装 pretty_midi: pip install pretty_midi")

        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append((note.pitch, note.start, note.end))
        # 按开始时间排序
        notes.sort(key=lambda x: x[1])
        return notes

    def _build_score(self, notes, syllables, consonant_frames=12,
                     consonant_ratio=0.0, mute_unvoiced_f0=False,
                     plosive_frames=0, cons_dur_table=None, cons_seconds=None):
        """从 MIDI 音符 + 歌词构建音素序列/时长/F0/note_lengths

        Args:
            notes: list of (pitch, start_sec, end_sec)
            syllables: list of str — 拼音音节
            consonant_frames: int — 辅音分配的帧数 (consonant_ratio>0 时作下限)
            consonant_ratio: float — >0 时按音符长度比例分配辅音帧 (clamp 6~14)
            mute_unvoiced_f0: bool — 清辅音帧 F0 置 0 (与训练数据一致)

        Returns:
            phoneme_ids: np.ndarray (P,)
            durations:   np.ndarray (P,) — 每音素帧数
            f0_curve:    np.ndarray (T_total,) — 帧级 F0
            note_lengths:np.ndarray (N,) — 每音符帧数
            note_starts: list[int] — 每音符在 f0_curve 中的起始帧下标 (供滑音处理)
        """
        phoneme_ids_list = []
        durations_list = []
        note_lengths_list = []
        note_starts_list = []
        f0_segments = []
        cur_frames = 0  # f0_curve 中已累计的帧数 (含静音间隙)

        # 默认：统计表 + 爆破音固定 32ms（传 {} 可关闭爆破音覆盖）
        if cons_dur_table is None:
            cons_dur_table = getattr(self, 'cons_dur_table', None)
        if cons_seconds is None:
            cons_seconds = DEFAULT_CONS_SECONDS

        for i, ((pitch, start_sec, end_sec), syl) in enumerate(zip(notes, syllables)):
            # 获取音素
            ph_list = PINYIN_TO_PHONEMES.get(normalize_pinyin(syl), ["a"])
            # 静音间隙
            if i > 0:
                prev_end = notes[i - 1][2]
                gap = start_sec - prev_end
                if gap > 0.01:
                    gap_frames = max(1, int(gap * self.frame_rate))
                    phoneme_ids_list.extend([self.p2i.get("SP", 1)] * 1)
                    durations_list.append(gap_frames)
                    f0_segments.append(np.full(gap_frames, 0.0))
                    cur_frames += gap_frames

            # 音符总帧数
            note_dur_sec = end_sec - start_sec
            note_total_frames = max(1, int(note_dur_sec * self.frame_rate))
            note_lengths_list.append(note_total_frames)
            note_starts_list.append(cur_frames)

            # 分配各音素时长
            n_ph = len(ph_list)
            cons_frames = 0
            if n_ph == 1:
                # 单音素
                phoneme_ids_list.extend([self.p2i.get(ph_list[0], 0)] * 1)
                durations_list.append(note_total_frames)
            else:
                # 首音素为辅音，其余为韵母
                first_ph = ph_list[0]  # 辅音
                rest_phs = ph_list[1:]  # 韵母

                # 辅音帧数：秒数覆盖 > 数据统计表 > 'auto' | 固定值 | 比例
                if cons_seconds and first_ph in cons_seconds:
                    cons_frames = max(1, int(round(float(cons_seconds[first_ph]) * self.frame_rate)))
                elif cons_dur_table is not None and first_ph in cons_dur_table:
                    ent = cons_dur_table[first_ph]
                    target = int(round(note_total_frames * ent['ratio_p50']))
                    cap = int(min(round(note_total_frames * ent['ratio_p90']),
                                  ent.get('frames_p90', 99)))
                    cons_frames = max(1, min(target, max(1, cap)))
                elif consonant_ratio > 0:
                    cons_frames = int(round(note_total_frames * consonant_ratio))
                    cons_frames = max(6, min(cons_frames, 14))
                elif isinstance(consonant_frames, str) and consonant_frames == 'auto':
                    cons_frames = CONSONANT_FRAME_TABLE.get(first_ph, 10)
                    if plosive_frames > 0 and first_ph in PLOSIVES:
                        cons_frames = int(plosive_frames)
                else:
                    cons_frames = int(consonant_frames)
                cons_frames = min(cons_frames, note_total_frames // 2)
                vowel_frames = note_total_frames - cons_frames

                # 辅音
                phoneme_ids_list.extend([self.p2i.get(first_ph, 0)] * 1)
                durations_list.append(cons_frames)

                # 韵母: 平均分配剩余帧
                if len(rest_phs) == 1:
                    phoneme_ids_list.extend([self.p2i.get(rest_phs[0], 0)] * 1)
                    durations_list.append(vowel_frames)
                else:
                    per_vowel = vowel_frames // len(rest_phs)
                    rem = vowel_frames % len(rest_phs)
                    for j, rp in enumerate(rest_phs):
                        phoneme_ids_list.extend([self.p2i.get(rp, 0)] * 1)
                        d = per_vowel + (1 if j < rem else 0)
                        durations_list.append(max(1, d))

            # F0：音符内恒定，清辅音段置 0
            f0_note = np.full(note_total_frames, midi_to_f0(pitch))
            if (mute_unvoiced_f0 and n_ph > 1 and cons_frames > 0
                    and ph_list[0] in UNVOICED_INITIALS):
                f0_note[:cons_frames] = 0.0
            f0_segments.append(f0_note)
            cur_frames += note_total_frames

        # 拼接
        phoneme_ids = np.array(phoneme_ids_list, dtype=np.int64)
        durations = np.array(durations_list, dtype=np.int64)
        f0_curve = np.concatenate(f0_segments) if f0_segments else np.array([0.0])
        note_lengths = np.array(note_lengths_list, dtype=np.int64)

        return phoneme_ids, durations, f0_curve, note_lengths, note_starts_list

    @staticmethod
    def save(path, waveform, sample_rate=24000):
        """保存波形到 WAV 文件"""
        import soundfile as sf
        waveform = np.asarray(waveform, dtype=np.float32)
        # 防削波
        peak = np.abs(waveform).max()
        if peak > 1.0:
            waveform = waveform / peak * 0.95
        sf.write(path, waveform, sample_rate)
        duration = len(waveform) / sample_rate
        print(f"保存: {path} ({duration:.1f}s)")

    def pinyin_to_phonemes(self, pinyin):
        """拼音 → 音素字符串列表"""
        return PINYIN_TO_PHONEMES.get(pinyin.lower(), ["a"])

    def phoneme_str_to_id(self, phoneme_str):
        """音素字符串 → ID"""
        return self.p2i.get(phoneme_str, 0)

    def phoneme_id_to_str(self, phoneme_id):
        """音素 ID → 字符串"""
        return self.i2p.get(str(phoneme_id), "?")

    def lyrics_to_score(self, lyrics, note_duration_frames, f0_per_note,
                        consonant_frames=2):
        """歌词 + 音符信息 → 音素/时长序列 (方便编程式使用)

        Args:
            lyrics: str — 空格分隔的拼音, 如 "ni hao a"
            note_duration_frames: list[int] — 每音符帧数
            f0_per_note: list[float] — 每音符 F0 (Hz)
            consonant_frames: int

        Returns:
            phoneme_ids: np.ndarray (P,)
            durations:   np.ndarray (P,)
            f0_curve:    np.ndarray (T_total,)
            note_lengths:np.ndarray (N,)
        """
        syllables = lyrics.strip().split()
        assert len(syllables) == len(note_duration_frames) == len(f0_per_note), \
            f"音节({len(syllables)})/音符帧数({len(note_duration_frames)})/F0({len(f0_per_note)})数量不匹配"

        phoneme_ids_list = []
        durations_list = []
        f0_segments = []

        for syl, n_frames, f0_val in zip(syllables, note_duration_frames, f0_per_note):
            ph_list = PINYIN_TO_PHONEMES.get(syl.lower(), ["a"])
            n_ph = len(ph_list)

            if n_ph == 1:
                phoneme_ids_list.extend([self.p2i.get(ph_list[0], 0)] * 1)
                durations_list.append(n_frames)
            else:
                cons_frames = min(consonant_frames, n_frames // 2)
                vowel_frames = n_frames - cons_frames

                # 辅音
                phoneme_ids_list.extend([self.p2i.get(ph_list[0], 0)] * 1)
                durations_list.append(cons_frames)

                # 韵母
                rest = ph_list[1:]
                if len(rest) == 1:
                    phoneme_ids_list.extend([self.p2i.get(rest[0], 0)] * 1)
                    durations_list.append(vowel_frames)
                else:
                    per = vowel_frames // len(rest)
                    rem = vowel_frames % len(rest)
                    for j, rp in enumerate(rest):
                        phoneme_ids_list.extend([self.p2i.get(rp, 0)] * 1)
                        durations_list.append(max(1, per + (1 if j < rem else 0)))

            f0_segments.append(np.full(n_frames, f0_val))

        return (
            np.array(phoneme_ids_list, dtype=np.int64),
            np.array(durations_list, dtype=np.int64),
            np.concatenate(f0_segments),
            np.array(note_duration_frames, dtype=np.int64),
        )


