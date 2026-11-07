"""
example.py — YUAZ SGR 推理示例

使用前:
    1. 安装依赖: pip install -r requirements.txt
    2. 将训练好的检查点放在 yuaz_sgr_inference/ 目录下

运行:
    python example.py
"""

import os
import sys

# 确保能找到 yuaz_sgr 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yuaz_sgr import YuazSinger

def example_sing_from_score():
    """示例1: 用音素ID+时长+F0曲线直接合成"""
    print("=" * 50)
    print("示例1: sing() — 直接合成")
    print("=" * 50)

    # 加载模型 (请替换为你的检查点路径)
    checkpoint = "checkpoint_75k_pure.pt"
    if not os.path.exists(checkpoint):
        print(f"跳过: 检查点 {checkpoint} 不存在, 请先放置模型文件")
        return

    singer = YuazSinger(checkpoint, device="cpu")

    # 音素序列: "n i h ao a" (你好啊)
    # phoneme_map: n=33, i=19, h=18, ao=6, a=2
    phoneme_ids = [33, 19, 18, 6, 2]

    # 每个音素的帧数 (帧率 93.75Hz)
    durations = [3, 40, 3, 40, 10]

    # F0 曲线 (Hz) — 对应每帧, 共 3+40+3+40+10=96 帧
    import numpy as np
    f0_curve = np.concatenate([
        np.full(3, 262),   # C4  (n)
        np.full(40, 330),  # E4  (i)
        np.full(3, 262),   # C4  (h)
        np.full(40, 330),  # E4  (ao)
        np.full(10, 294),  # D4  (a)
    ])

    # 合成
    wav = singer.sing(
        phoneme_ids, durations, f0_curve,
        use_predicted_f0=True,
        irr_steps=4,
    )

    singer.save("output_example1.wav", wav)
    print("完成!\n")

def example_sing_from_phonemes():
    """示例2: 用音素字符串合成 (更友好)"""
    print("=" * 50)
    print("示例2: sing_from_phonemes() — 音素字符串")
    print("=" * 50)

    checkpoint = "checkpoint_75k_pure.pt"
    if not os.path.exists(checkpoint):
        print(f"跳过: 检查点 {checkpoint} 不存在")
        return

    singer = YuazSinger(checkpoint, device="cpu")
    import numpy as np

    # 音素: n i h ao (你好)
    # 时长分配: 辅音 12 帧 / 元音 36 帧
    phonemes = ["n", "i", "h", "ao"]
    durations = [12, 36, 12, 36]

    # F0: C4(262Hz) → E4(330Hz)
    f0_curve = np.concatenate([
        np.full(12, 262), np.full(36, 330),
        np.full(12, 262), np.full(36, 330),
    ])

    wav = singer.sing_from_phonemes(
        phonemes, durations, f0_curve,
        use_predicted_f0=True,
    )
    singer.save("output_example2.wav", wav)
    print("完成!\n")

def example_sing_from_midi():
    """示例3: 从 MIDI 文件合成"""
    print("=" * 50)
    print("示例3: sing_from_midi() — MIDI 文件")
    print("=" * 50)

    checkpoint = "checkpoint_75k_pure.pt"
    midi_path = "思华年.mid"

    if not os.path.exists(checkpoint):
        print(f"跳过: 检查点 {checkpoint} 不存在")
        return
    if not os.path.exists(midi_path):
        print(f"跳过: MIDI 文件 {midi_path} 不存在")
        return

    singer = YuazSinger(checkpoint, device="cpu")

    # 从 MIDI + 歌词合成
    wav = singer.sing_from_midi(
        midi_path,
        lyrics="si hua nian",  # 拼音用空格分隔
        output_path="output_example3.wav",
        predict_durations=False,
        use_predicted_f0=True,
        irr_steps=4,
    )
    print("完成!\n")

def example_three_modes():
    """示例4: 对比推理模式 (IRR 步数)"""
    print("=" * 50)
    print("示例4: 三种推理模式对比")
    print("=" * 50)

    checkpoint = "checkpoint_75k_pure.pt"
    if not os.path.exists(checkpoint):
        print(f"跳过: 检查点 {checkpoint} 不存在")
        return

    singer = YuazSinger(checkpoint, device="cpu")
    import numpy as np

    phonemes = ["w", "o", "sh", "i", "n", "a", "y", "i", "n", "a"]
    # 时长分配: 辅音 12 帧 / 元音 36 帧 (每音符 48 帧)
    durations = [12, 36, 12, 36, 12, 36, 12, 36, 12, 36]
    f0 = np.concatenate([
        np.full(12, 294), np.full(36, 294),
        np.full(12, 330), np.full(36, 330),
        np.full(12, 349), np.full(36, 349),
        np.full(12, 330), np.full(36, 330),
        np.full(12, 294), np.full(36, 294),
    ])
    note_lengths = [48, 48, 48, 48, 48]

    phoneme_ids = [singer.phoneme_str_to_id(p) for p in phonemes]

    # 模式1: 原版 F0 (直接用输入的)
    print("模式1: 原版 F0...")
    wav1 = singer.sing(phoneme_ids, durations, f0, use_predicted_f0=False)
    singer.save("output_mode1_original.wav", wav1)

    # 模式2: 预测 F0
    print("模式2: F0Predictor 精炼...")
    wav2 = singer.sing(phoneme_ids, durations, f0, use_predicted_f0=True)
    singer.save("output_mode2_predicted_f0.wav", wav2)

    # 模式3: 预测时长 + 预测 F0
    print("模式3: 预测时长 + 预测 F0...")
    wav3 = singer.sing(
        phoneme_ids, durations, f0,
        note_lengths=note_lengths,
        predict_durations=True,
        use_predicted_f0=True,
    )
    singer.save("output_mode3_predicted_all.wav", wav3)

    print("完成!\n")

def example_programmatic():
    """示例5: 编程式使用 lyrics_to_score"""
    print("=" * 50)
    print("示例5: lyrics_to_score() — 编程式")
    print("=" * 50)

    # 不需要模型加载, 只需要音素映射
    from yuaz_sgr.synthesizer import PINYIN_TO_PHONEMES

    # 查看某个拼音的音素分解
    for pinyin in ["ni", "hao", "shi", "jie", "zhong", "guo"]:
        phs = PINYIN_TO_PHONEMES.get(pinyin, ["?"])
        print(f"  {pinyin:8s} → {phs}")

    print("\n完成!\n")

if __name__ == "__main__":
    print("YUAZ SGR 推理示例\n")

    example_programmatic()  # 不需要模型, 直接运行
    example_sing_from_score()
    example_sing_from_phonemes()
    example_sing_from_midi()
    example_three_modes()

