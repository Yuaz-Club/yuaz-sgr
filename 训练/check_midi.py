"""检查黛玉葬花.mid 的内容"""
import pretty_midi
import sys

midi_path = r"d:\360MoveData\Users\洛洛\Desktop\yuaz_sgr own\黛玉葬花.mid"

midi = pretty_midi.PrettyMIDI(midi_path)
print(f"=== {midi_path} ===")
print(f"调号: {midi.key_signature_changes}")
print(f"速度: {midi.time_signature_changes}")

for i, inst in enumerate(midi.instruments):
    print(f"\n音轨 {i}: name={inst.name}, program={inst.program}, notes={len(inst.notes)}")
    for j, note in enumerate(inst.notes[:5]):
        dur = note.end - note.start
        f0 = 440.0 * (2 ** ((note.pitch - 69) / 12))
        print(f"  note[{j}]: pitch={note.pitch}, F0={f0:.0f}Hz, start={note.start:.2f}s, end={note.end:.2f}s, dur={dur:.2f}s")
    if len(inst.notes) > 5:
        print(f"  ... ({len(inst.notes)} notes total)")

# Check for lyrics
for i, inst in enumerate(midi.instruments):
    if inst.is_drum:
        continue
    # pretty_midi doesn't directly expose lyrics, let's try mido
    break

# Try mido for lyrics
import mido
mid = mido.MidiFile(midi_path)
print(f"\nMIDI format: {mid.type}, tracks: {len(mid.tracks)}")
for ti, track in enumerate(mid.tracks):
    lyrics = []
    for msg in track:
        if msg.type == 'lyrics' and msg.text.strip():
            lyrics.append(msg.text.strip())
    if lyrics:
        print(f"  歌词音轨 {ti}: {lyrics}")
