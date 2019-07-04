import glob
import pickle
import numpy as np
import music21
from music21 import converter, instrument, note, chord, interval, pitch, stream

PitchMin = 33 #A1
PitchMax = 96 #C7
DurationMin = 0.5
DurationMax = 4.0

PitchTokey = dict((float(pitch), number+1) for number, pitch in enumerate(range(PitchMin, PitchMax + 1)))
PitchTokey[0] = 0
KeyToPitch = dict((number+1, float(pitch)) for number, pitch in enumerate(range(PitchMin, PitchMax + 1)))
KeyToPitch[0] = 0

def parse_midi(path, save_path=None):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    data = []

    for midi_path in glob.glob(path):
        print("Parsing {}".format(midi_path))
        midi_file = converter.parse(midi_path)
        # midi_file = to_c_major(midi_file)

        parts = instrument.partitionByInstrument(midi_file)

        if parts: # file has instrument parts
            melody = parts[0]
        else: # file has notes in a flat structure
            melody = midi_file.flat

        try:
            measures = melody.measures(0, None, collect="Measure")
        except:
            print("Unavailable Midi {}".format(midi_path))
            continue

        # measure_len = int(DurationMax // DurationMin)
        key_numbers = len(PitchTokey)
        notes = []
        for measure in measures:
            offset_iter = stream.iterator.OffsetIterator(measure.recurse().notes)
            # measure_notes = np.full((measure_len, key_numbers), np.zeros(key_numbers))
            for element_group in offset_iter:
                # offset = element_group[0].offset

                # if offset % DurationMin != 0 or offset < DurationMin or offset >= DurationMax:
                #     continue
                
                # offset_index = int(offset // DurationMin)
                keyboard = to_keyboard(element_group, key_numbers)
                notes.append(keyboard)

        data.append(notes)

    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(data, file)

    return data 

def to_keyboard(element_group, key_numbers):
    keyboard = np.zeros(key_numbers)

    element = element_group[0]

    if isinstance(element, note.Note):
        pitch_index = element.pitch.ps
    elif isinstance(element, chord.Chord):
        pitch_index = max([pitch.ps for pitch in element.pitches])
    else:
        pitch_index = 0
        key_index = 0

    if PitchMax >= pitch_index >= PitchMin:
        key_index = PitchTokey[pitch_index]
        keyboard[key_index] = 1
    else:
        keyboard[0] = 1

    return keyboard

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

def clamp_chord(pitches, min, max):
    for pitch in pitches:
        pitch.ps = clamp_pitch(pitch.ps, min, max)
    return pitches

def clamp_duration(value, min , max):
    if value < min:
        value = min 
     
    if value > max:
        value = max 

    return value

def round_duration(value, min):
    return round(value / min) * min

def load_data(path):

    with open(path, 'rb') as file:
        data = pickle.load(file)
    
    return data