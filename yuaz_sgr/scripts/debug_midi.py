import mido
mid = mido.MidiFile("思华年.mid")
track = mid.tracks[1]
ticks_per_beat = mid.ticks_per_beat
tempo = 652174

# Parse notes with absolute timing
now_ticks = 0
active = {}
notes_raw = []
for msg in track:
    now_ticks += msg.time
    dt = mido.tick2second(now_ticks, ticks_per_beat, tempo)
    if msg.type == "note_on" and msg.velocity > 0:
        active[msg.note] = dt
    elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
        start = active.pop(msg.note, None)
        if start is not None:
            notes_raw.append((start, dt, msg.note))

notes_raw.sort(key=lambda x: x[0])
print("Total notes:", len(notes_raw))

# Show gaps between notes and proposed extended durations
for i, (s, e, n) in enumerate(notes_raw):
    if i < len(notes_raw) - 1:
        next_start = notes_raw[i+1][0]
    else:
        next_start = e  # last note: just use its own duration
    dur_on = e - s
    gap_to_next = next_start - s  # effective duration = from this note start to next note start
    print("note=%2d f0=%.0fHz start=%.2f on_dur=%.3f effective_dur=%.3f (gap_after=%.3f)" % (n, 440*(2**((n-69)/12)), s, dur_on, gap_to_next, gap_to_next-dur_on))
