import sys
sys.path.insert(0, "/root/autodl-tmp/yuaz_sgr")
from scripts.preprocess_opencpop import load_audio, load_phonemes, load_notes, align_phonemes_to_notes, compute_phone_durations, segment_audio
import numpy as np

wav = "/root/autodl-tmp/yuaz_sgr/opencpop/wavs/2007.wav"
tg  = "/root/autodl-tmp/yuaz_sgr/opencpop/textgrids/2007.TextGrid"
mid = "/root/autodl-tmp/yuaz_sgr/opencpop/midis/2007.midi"

audio = load_audio(wav)
print("audio[0:100]:", audio[:100].min(), audio[:100].max())

phonemes, intervals = load_phonemes(tg)
notes = load_notes(mid)
aligned = align_phonemes_to_notes(phonemes, intervals, notes)

print("note 0 start:", aligned[0]['note_start'], "end:", aligned[0]['note_end'])
print("note 0 dur:", aligned[0]['note_dur_sec'])
start_sample = int(aligned[0]['note_start'] * 24000)
print("start_sample:", start_sample)
print("audio at note_start:", audio[start_sample:start_sample+100].min(), audio[start_sample:start_sample+100].max())

frame_phones, durations, f0_curve, lags, note_lengths = compute_phone_durations(aligned)
segments = segment_audio(audio, frame_phones, durations, f0_curve, lags, note_lengths, aligned)
if segments:
    s = segments[0]
    print("seg0 audio[:100]:", s['audio'][:100].min(), s['audio'][:100].max())
    # Check if seg0 matches the expected audio range
    note_start_sample = 0
    note_end_sample = sum(max(1, int(aligned[n]['note_dur_sec'] * 24000)) for n in range(19))
    print(f"expected range: [{note_start_sample}:{note_end_sample}]")
    print(f"audio[{note_start_sample}:{note_end_sample}][:100]:", audio[note_start_sample:note_end_sample][:100].min(), audio[note_start_sample:note_end_sample][:100].max())
