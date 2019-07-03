import glob
import pickle
import numpy
import music21
from music21 import converter, instrument, note, chord, interval, pitch, stream

PitchMin = 33 #A1
PitchMax = 96 #C7
DurationMin = 0.25
DurationMax = 4.0
PitchTable = [0.0]
PitchTable.extend([float(x) for x in range(PitchMin, PitchMax + 1)])

def parse_midi(path, save_path=None):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    data = []

    for midi_path in glob.glob(path):
        print("Parsing {}".format(midi_path))
        midi_file = converter.parse(midi_path)
        midi_file = to_c_major(midi_file)

        if len(midi_file.parts) == 2:
            melody = midi_file.parts[0]
            accomp = midi_file.parts[1]
        else:
            print("Not Piano Structure {}".format(midi_path))
            continue

        try:
            measures = melody.measures(0, None, collect="Measure")
        except:
            print("Unavailable Midi {}".format(midi_path))
            continue

        notes = []

        for measure in measures:
            notes_to_parse = measure.recurse().notesAndRests
            offset_iter = stream.iterator.OffsetIterator(notes_to_parse)
            for element_group in offset_iter:
                element = element_group[0]
                if isinstance(element, note.Rest):
                    dt = clamp_duration(element.duration.quarterLength, DurationMin, DurationMax) 
                    notes.append([0, dt])
                elif isinstance(element, note.Note):
                    ps = clamp_pitch(element.pitch.ps, PitchMin, PitchMax)
                    dt = clamp_duration(element.duration.quarterLength, DurationMin, DurationMax) 
                    notes.append([ps, dt])
                elif isinstance(element, chord.Chord):
                    element = element.sortDiatonicAscending()
                    ps = clamp_pitch(element.pitches[-1].ps, PitchMin, PitchMax)
                    dt = clamp_duration(element.duration.quarterLength, DurationMin, DurationMax) 
                    notes.append([ps, dt])

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


def clamp_pitch(value, min, max):

    while value < min:
        value += 12.0
    while value > max:
        value -= 12.0

    return value

def clamp_duration(value, min, max):

    if value < min:
        value = min

    if value > max:
        value = max

    return value

def load_data(path):

    with open(path, 'rb') as file:
        data = pickle.load(file)
    
    return data