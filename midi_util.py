import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord

def parse_midi(path, save_path=None):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for midi_path in glob.glob(path):
        midi = converter.parse(midi_path)
        print("Parsing %s" % midi_path)
        notes_to_parse = None

        try: 
            # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: 
            # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(notes, file)

    return notes

def load_notes(path):

    with open(path, 'rb') as file:
        notes = pickle.load(file)
    
    return notes