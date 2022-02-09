def shorten_durations(generated_notes, notes_after):
    # shorten durations if necessary
    # (case when two notes overlap are not well-handled by Ableton)
    # (causes serious issues when generated notes overlap with the region after)

    # {pitch: index_in_the_seq_of_the_last_played_pitch}
    d = {}

    max_end_time = 0
    for k, note in enumerate(generated_notes):
        pitch = note["pitch"]

        if pitch in d:
            previous_note = generated_notes[d[note["pitch"]]]

            current_time = note["time"]
            previous_note_sounding_end = (
                previous_note["time"] + previous_note["duration"]
            )

            if previous_note_sounding_end > current_time:
                previous_note["duration"] = current_time - previous_note["time"] - 1e-2

        max_end_time = max(max_end_time, note["time"] + note["duration"])
        d[note["pitch"]] = k

    print(f"MAX END TIME: {max_end_time}")
    print("SHORTENING")
    for k, note in enumerate(notes_after):
        print(f"time: {note['time']}")
        time = note["time"]
        if time > max_end_time:
            break

        if len(d) == 0:
            break

        pitch = note["pitch"]

        if pitch in d:
            previous_note = generated_notes[d[pitch]]

            current_time = note["time"]
            previous_note_sounding_end = (
                previous_note["time"] + previous_note["duration"]
            )
            if previous_note_sounding_end > current_time:
                previous_note["duration"] = current_time - previous_note["time"] - 1e-2
                print(f"Note cut {previous_note}")
            del d[pitch]

    return generated_notes
