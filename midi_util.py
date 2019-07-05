import glob
import pickle
import numpy as np
import music21
from music21 import converter, instrument, note, chord, interval, pitch, stream

PitchMin = 33 #A1
PitchMax = 96 #C7
OffsetStep = 0.25
OffsetMax = 4.0

def parse_midi(path, save_path=None, part_index=0):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    data = []

    for midi_path in glob.glob(path):
        print("Parsing {}".format(midi_path))
        midi_file = converter.parse(midi_path)
        midi_file = to_major(midi_file, "C")

        if midi_file.parts: # file has multi-parts
            melody = midi_file.parts[part_index]
        else: # file has notes in a flat structure
            melody = midi_file.flat

        try:
            measures = melody.measures(0, None, collect="Measure")
        except:
            print("Unavailable Midi {}".format(midi_path))
            continue

        notes = []

        for measure in measures:
            offset_iter = stream.iterator.OffsetIterator(measure.recurse().notesAndRests)
            measure_len = int(OffsetMax / OffsetStep)
            measure_pitches = np.zeros(measure_len)
            measure_press = np.ones(measure_len)

            for element_group in offset_iter:
                offset = element_group[0].offset

                if offset % OffsetStep != 0:
                    continue
                offset_index = int(offset // OffsetStep)

                element = element_group[0]
                if isinstance(element, note.Note):
                    pitch_space = element.pitch.ps
                elif isinstance(element, chord.Chord):
                    pitch_space = max([pitch.ps for pitch in element.pitches])
                else: #Rest
                    pitch_space = 0

                if pitch_space < PitchMin or pitch_space > PitchMax:
                    pitch_space = 0

                # Represent Continuity
                measure_pitches[offset_index:] = pitch_space
                measure_press[offset_index:] = np.array(range(offset_index, measure_len))
                measure_press[offset_index:] = measure_press[offset_index:] - offset_index + 1

            measure_notes = np.stack((measure_pitches, measure_press), axis=-1)

            notes.extend(measure_notes)
  
        data.append(notes)

    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(data, file)

    return data 

def to_major(midi_file, scale_name):
    midi_scale = midi_file.analyze('key').getScale('major')
    interv = interval.Interval(midi_scale.tonic, pitch.Pitch(scale_name))
    return midi_file.transpose(interv)

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