import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord

def parse_midi(path):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob(path):
        midi = converter.parse(file)
        print("Parsing %s" % file)
        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

if __name__ == '__main__':

    midi_path = "midi_songs/*.mid"
    notes = parse_midi(midi_path)

    with open('midi_input/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)