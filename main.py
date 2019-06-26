import midi_util
import glob

if __name__ == '__main__':

    # parse midi songs to notes file
    midi_path = "midi_songs/*.mid"
    notes_path = "midi_input/notes"

    if glob.glob(notes_path):
        notes = midi_util.load_notes(notes_path)
    else:
        notes = midi_util.parse_midi(midi_path, notes_path)

