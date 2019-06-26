import midi_util


if __name__ == '__main__':
    # parse midi file to notes list
    midi_path = "midi_songs/*.mid"
    save_path = "midi_input/notes"

    notes = midi_util.parse_midi(midi_path, save_path)