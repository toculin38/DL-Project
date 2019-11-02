import glob
import pickle
import numpy as np
import music21
from music21 import converter, instrument, note, chord, interval, pitch, stream

PitchMin = 21 #A0
PitchMax = 108 #C8
PitchSize = PitchMax - PitchMin + 1
OffsetStep = 0.25
OffsetMax = 4.0

def element_to_note(element, max_flag):
    if isinstance(element, note.Note):
        pitch = element.pitch
        pitch_space = pitch.ps
        # pitch_space = (pitch.ps - PitchMin) % 12 + 1
        # octave = pitch.implicitOctave + 1
    elif isinstance(element, chord.Chord):
        if max_flag:
            pitch = max(element.pitches, key=lambda item: item.ps)
            pitch_space = pitch.ps
            # pitch_space = (pitch.ps - PitchMin) % 12 + 1
            # octave = pitch.implicitOctave + 1
        else:
            pitch = min(element.pitches, key=lambda item: item.ps)
            pitch_space = pitch.ps
            # pitch_space = (pitch.ps - PitchMin) % 12 + 1
            # octave = pitch.implicitOctave + 1
    else: #Rest
        pitch_space = 0
        # octave = 0

    if pitch_space < PitchMin or pitch_space > PitchMax:
        pitch_space = 0

    return pitch_space

def element_to_keys(element):

    keys_space = np.zeros(PitchSize)

    if isinstance(element, note.Note):
        if PitchMin <= element.pitch.ps <= PitchMax:
            idx = int(element.pitch.ps) - PitchMin
            keys_space[idx] = 1
    elif isinstance(element, chord.Chord):
        for pitch in element.pitches:
            if PitchMin <= pitch.ps <= PitchMax:
                idx = int(pitch.ps) - PitchMin
                keys_space[idx] = 1

    return keys_space

def parse_midi(midi_path, save_path=None):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    print("Parsing {}".format(midi_path))
    midi_file = converter.parse(midi_path)
    midi_file = to_major(midi_file, "C")

    try:
        melody = midi_file.parts[0].measures(0, None, collect="Measure")
        accomp = midi_file.parts[1].measures(0, None, collect="Measure")
    except:
        print("Unavailable Midi {}".format(midi_path))
        return False

    notes = []

    pre_key_pitch = 0
    pre_acp_pitch = 0

    for measure in zip(melody, accomp):

        melody_iter = stream.iterator.OffsetIterator(measure[0].recurse().notesAndRests)
        accomp_iter = stream.iterator.OffsetIterator(measure[1].recurse().notesAndRests)

        measure_len = int(OffsetMax / OffsetStep)
        melody_pitches = np.zeros((measure_len))
        accomp_pitches = np.zeros((measure_len))

        melody_pitches[:] = pre_key_pitch
        accomp_pitches[:] = pre_acp_pitch

        for melody_group in melody_iter:

            element1 = melody_group[0]

            # if element1.offset % OffsetMax == 0 and isinstance(element1, note.Rest):
            #     continue

            if element1.offset % OffsetStep == 0 and element1.offset < OffsetMax:
                offset_index = int(element1.offset // OffsetStep)
                pitch = element_to_note(element1, max_flag=True)
                melody_pitches[offset_index] = pitch
                melody_pitches[offset_index + 1:] = pitch

        for accomp_group in accomp_iter:
            
            element2 = accomp_group[0]

            # if element2.offset % OffsetMax == 0 and isinstance(element2, note.Rest):
            #     continue

            if element2.offset % OffsetStep == 0 and element2.offset < OffsetMax:
                offset_index = int(element2.offset // OffsetStep)
                pitch = element_to_note(element2, max_flag=False)
                accomp_pitches[offset_index] = pitch
                accomp_pitches[offset_index + 1:] = pitch

        pre_key_pitch = melody_pitches[-1]
        pre_acp_pitch = accomp_pitches[-1]

        # print(melody_pitches)
        # print(accomp_pitches)

        measure_notes = np.stack((melody_pitches, accomp_pitches), axis=-1)

        notes.extend(measure_notes)

    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(notes, file)

    return notes 

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