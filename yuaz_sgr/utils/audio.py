# YUAZ SGR
"""utils/audio"""
import torch
import numpy as np
import librosa

def load_wav(path, sr=24000):
    """加载音频, 重采样到指定采样率"""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def save_wav(path, audio, sr=24000):
    """保存音频"""
    import soundfile as sf
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(path, audio, sr)

def normalize_db(audio, target_db=-26):
    """归一化到目标 dB"""
    rms = np.sqrt(np.mean(audio ** 2))
    return audio * (10 ** (target_db / 20)) / (rms + 1e-8)

def pad_or_trim(audio, target_length):
    """填充或截断到目标长度"""
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio[:target_length]

def mel_spectrogram(audio, sr=24000, n_fft=1024, hop_length=256, n_mels=128):
    """计算 mel-spectrogram"""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return np.log(np.clip(mel, a_min=1e-5, a_max=None))

def f0_to_midi(f0):
    """F0 (Hz) → MIDI pitch"""
    return 69 + 12 * np.log2(np.clip(f0, 1e-8, None) / 440.0)

def midi_to_f0(midi_pitch):
    """MIDI pitch → F0 (Hz)"""
    return 440.0 * (2 ** ((midi_pitch - 69) / 12))