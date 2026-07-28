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

# ---------- 拼音 → 音素映射 ----------
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


# ---------- MIDI → F0 工具 ----------
def midi_to_f0(midi_pitch):
    """MIDI pitch → F0 (Hz). A4=440Hz, MIDI=69"""
    return 440.0 * (2 ** ((np.asarray(midi_pitch, dtype=np.float64) - 69) / 12))


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

    def __init__(self, checkpoint_path, config_dir=None, device="cpu"):
        self.device = device

        # 加载配置
        if config_dir is None:
            config_dir = _PACKAGE_ROOT / "yuaz_sgr" / "assets"
        config_dir = Path(config_dir)

        with open(config_dir / "base.yaml", "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        with open(config_dir / "model.yaml", "r", encoding="utf-8") as f:
            model_cfg_file = yaml.safe_load(f)

        # 合并配置 (model.yaml 覆盖 base.yaml, 并添加 encoder 中的字段)
        config = {**base_cfg, **model_cfg_file}
        # 确保 encoder 有完整字段 (model.yaml 已有, 但合并后也要保留)
        if "encoder" not in config:
            config["encoder"] = {}
        config["encoder"].setdefault("hidden_dim", 512)
        config["encoder"].setdefault("latent_dim", 128)
        config["encoder"].setdefault("n_mfcc", base_cfg.get("n_mfcc", 30))
        config["encoder"].setdefault("n_fft", base_cfg.get("n_fft", 1024))
        config["encoder"].setdefault("hop_length", base_cfg.get("hop_length", 256))
        config["encoder"].setdefault("sample_rate", base_cfg.get("sample_rate", 24000))

        self.config = config

        # 加载音素映射
        phoneme_map_path = config_dir / "phoneme_map.json"
        with open(phoneme_map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.p2i = mapping["p2i"]
        self.i2p = mapping["i2p"]

        # 帧率
        self.sample_rate = config.get("sample_rate", 24000)
        self.hop_length = config.get("hop_length", 256)
        self.frame_rate = self.sample_rate / self.hop_length  # 93.75 Hz

        # 加载模型
        from .models.svs_model import SVSModel

        self.model = SVSModel(config)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.to(device)

        step = ckpt.get("step", "unknown")
        print(f"YuazSinger 加载完成: step={step}, device={device}")

    # =================== 核心合成接口 ===================

    @torch.no_grad()
    def sing(self, phoneme_ids, durations, f0_curve,
             note_lengths=None, predict_durations=False,
             use_predicted_f0=True, n_steps=1):
        """核心合成: 音素 + 时长 + F0 → 音频

        Args:
            phoneme_ids: list[int] 或 np.ndarray, shape (P,) — 音素 ID 序列
            durations:   list[int] 或 np.ndarray, shape (P,) — 每音素帧数
            f0_curve:    list[float] 或 np.ndarray, shape (T_frames,) — 帧级 F0 (Hz)
            note_lengths:list[int] 或 np.ndarray, shape (N,) — 每音符帧数 (predict_durations 时需要)
            predict_durations: bool — 是否用 TemporalPredictor 预测音素时长
            use_predicted_f0: bool — True=用 F0Predictor 精炼 F0, False=直接用输入 F0
            n_steps: int — Flow Matching ODE 积分步数 (1 或 8)

        Returns:
            waveform: np.ndarray, shape (T_samples,), dtype=float32, 24kHz
        """
        # 转换输入
        ph  = torch.LongTensor(np.asarray(phoneme_ids)).unsqueeze(0).to(self.device)  # (1, P)
        dur = torch.LongTensor(np.asarray(durations)).unsqueeze(0).to(self.device)     # (1, P)
        f0  = torch.FloatTensor(np.asarray(f0_curve)).unsqueeze(0).to(self.device)     # (1, T_f)

        nl = None
        if note_lengths is not None:
            nl = torch.LongTensor(np.asarray(note_lengths)).unsqueeze(0).to(self.device)  # (1, N)

        wav, _, _, _, _, _ = self.model.forward_synthesis(
            ph, dur, f0,
            note_lengths=nl,
            predict_durations=predict_durations,
            use_predicted_f0=use_predicted_f0,
            n_steps=n_steps,
        )
        return wav[0, 0].cpu().numpy().astype(np.float32)

    # =================== MIDI 接口 ===================

    @torch.no_grad()
    def sing_from_midi(self, midi_path, lyrics=None, output_path=None,
                       predict_durations=False, use_predicted_f0=True,
                       n_steps=1, consonant_frames=2):
        """从 MIDI 文件 + 歌词字符串合成

        Args:
            midi_path: str — MIDI 文件路径
            lyrics: str — 歌词, 拼音用空格分隔, 如 "ni hao a wo shi na yi na"
                    为 None 时自动按 NOTE 数量均匀取拼音
            output_path: str | None — WAV 输出路径, None 则不保存
            predict_durations: bool
            use_predicted_f0: bool
            n_steps: int
            consonant_frames: int — 每个辅音分配的最小帧数

        Returns:
            waveform: np.ndarray, 24kHz
        """
        # 解析 MIDI
        notes = self._parse_midi(midi_path)
        if not notes:
            raise ValueError(f"MIDI 文件中没有音符: {midi_path}")

        # 处理歌词
        if lyrics is not None:
            syllables = lyrics.strip().split()
            # 确保音节数与音符数匹配
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
        phoneme_ids, durations, f0_curve, note_lengths = self._build_score(
            notes, syllables, consonant_frames=consonant_frames
        )

        print(f"MIDI: {len(notes)} 音符, {len(phoneme_ids)} 音素, "
              f"F0 共 {len(f0_curve)} 帧 ({len(f0_curve)/self.frame_rate:.1f}s)")

        # 合成
        wav = self.sing(
            phoneme_ids, durations, f0_curve,
            note_lengths=note_lengths,
            predict_durations=predict_durations,
            use_predicted_f0=use_predicted_f0,
            n_steps=n_steps,
        )

        if output_path:
            self.save(output_path, wav)
        return wav

    # =================== 便捷接口 ===================

    @torch.no_grad()
    def sing_from_phonemes(self, phoneme_strs, durations, f0_curve,
                           note_lengths=None, predict_durations=False,
                           use_predicted_f0=True, n_steps=1):
        """从音素字符串列表合成 (比 sing() 更友好)

        Args:
            phoneme_strs: list[str] — 如 ["n", "i", "h", "ao", "a"]
            durations:    list[int] — 每音素帧数
            f0_curve:     list[float] — 帧级 F0
            ... 其他参数同 sing()
        """
        ph_ids = [self.p2i.get(p, 0) for p in phoneme_strs]
        return self.sing(
            ph_ids, durations, f0_curve,
            note_lengths=note_lengths,
            predict_durations=predict_durations,
            use_predicted_f0=use_predicted_f0,
            n_steps=n_steps,
        )

    # =================== 工具方法 ===================

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

    def _build_score(self, notes, syllables, consonant_frames=2):
        """从 MIDI 音符 + 歌词构建音素序列/时长/F0/note_lengths

        Args:
            notes: list of (pitch, start_sec, end_sec)
            syllables: list of str — 拼音音节
            consonant_frames: int — 辅音保留的最小帧数

        Returns:
            phoneme_ids: np.ndarray (P,)
            durations:   np.ndarray (P,) — 每音素帧数
            f0_curve:    np.ndarray (T_total,) — 帧级 F0
            note_lengths:np.ndarray (N,) — 每音符帧数
        """
        phoneme_ids_list = []
        durations_list = []
        note_lengths_list = []
        f0_segments = []

        for i, ((pitch, start_sec, end_sec), syl) in enumerate(zip(notes, syllables)):
            # 获取音素
            ph_list = PINYIN_TO_PHONEMES.get(syl.lower(), ["a"])
            # 静音间隙
            if i > 0:
                prev_end = notes[i - 1][2]
                gap = start_sec - prev_end
                if gap > 0.01:
                    gap_frames = max(1, int(gap * self.frame_rate))
                    phoneme_ids_list.extend([self.p2i.get("SP", 1)] * 1)
                    durations_list.append(gap_frames)
                    f0_segments.append(np.full(gap_frames, 0.0))

            # 音符总帧数
            note_dur_sec = end_sec - start_sec
            note_total_frames = max(1, int(note_dur_sec * self.frame_rate))
            note_lengths_list.append(note_total_frames)

            # 分配各音素时长
            n_ph = len(ph_list)
            if n_ph == 1:
                # 单音素 → 全部时长
                phoneme_ids_list.extend([self.p2i.get(ph_list[0], 0)] * 1)
                durations_list.append(note_total_frames)
            else:
                # 多音素: 第一个为辅音, 其余为韵母
                first_ph = ph_list[0]  # 辅音
                rest_phs = ph_list[1:]  # 韵母

                # 辅音分配 consonant_frames 帧
                cons_frames = min(consonant_frames, note_total_frames // 2)
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

            # F0 曲线: 音符保持恒定 MIDI 音高
            f0_segments.append(np.full(note_total_frames, midi_to_f0(pitch)))

        # 拼接
        phoneme_ids = np.array(phoneme_ids_list, dtype=np.int64)
        durations = np.array(durations_list, dtype=np.int64)
        f0_curve = np.concatenate(f0_segments) if f0_segments else np.array([0.0])
        note_lengths = np.array(note_lengths_list, dtype=np.int64)

        return phoneme_ids, durations, f0_curve, note_lengths

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

    # =================== 辅助方法 ===================

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
