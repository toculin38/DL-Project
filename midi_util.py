import glob
import pickle
import numpy
import music21
from music21 import converter, instrument, note, chord, interval, pitch

def parse_midi(path, save_path=None):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    data = []

    for midi_path in glob.glob(path):
        midi_file = converter.parse(midi_path)
        midi_file = to_c_major(midi_file)
        notes_to_parse = None

        try: 
            # file has instrument parts
            s2 = instrument.partitionByInstrument(midi_file)
            notes_to_parse = s2.parts[0].recurse() 
        except: 
            # file has notes in a flat structure
            notes_to_parse = midi_file.flat.notes

        notes = []

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                element = element.sortDiatonicAscending()
                notes.append(str(element.pitches[-1]))

        data.append(notes)

    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(data, file)

    return data

def to_c_major(midi_file):
    midi_scale = midi_file.analyze('key').getScale('major')
    interv = interval.Interval(midi_scale.tonic, pitch.Pitch('C'))
    midi_file = midi_file.transpose(interv)
    return midi_file


def load_data(path):

    with open(path, 'rb') as file:
        data = pickle.load(file)
    
    return data