import sys, json
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
from scripts.preprocess_opencpop import load_audio, load_phonemes, load_notes, align_phonemes_to_notes, compute_phone_durations, segment_audio
import numpy as np

wav = "/root/autodl-tmp/yuaz_sgr/opencpop/wavs/2007.wav"
tg  = "/root/autodl-tmp/yuaz_sgr/opencpop/textgrids/2007.TextGrid"
mid = "/root/autodl-tmp/yuaz_sgr/opencpop/midis/2007.midi"

audio = load_audio(wav)
print("audio:", audio.shape, audio.min(), audio.max(), audio.mean())

phonemes, intervals = load_phonemes(tg)
notes = load_notes(mid)
aligned = align_phonemes_to_notes(phonemes, intervals, notes)
print("aligned notes:", len(aligned))
for i, n in enumerate(aligned[:5]):
    print(f"  note {i}: dur={n['note_dur_sec']:.4f}s lag={n['lag']:.4f}s phones={len(n['phones'])}")

frame_phones, durations, f0_curve, lags, note_lengths = compute_phone_durations(aligned)
print("total frames:", len(f0_curve), "first f0:", f0_curve[:5])

segments = segment_audio(audio, frame_phones, durations, f0_curve, lags, note_lengths, aligned)
print("segments:", len(segments))
if segments:
    s = segments[0]
    print("seg0 audio:", s['audio'].shape, s['audio'].min(), s['audio'].max(), s['audio'].mean())
    print("seg0 f0:", s['f0_curve'].shape, s['f0_curve'][:5])
    print("seg0 durations:", s['durations'])
    print("seg0 note_lengths:", s['note_lengths'])
