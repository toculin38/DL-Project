import numpy as np
from keras.utils import np_utils
from music21 import stream, note, chord, instrument, midi
import midi_util

PitchMin = midi_util.PitchMin
PitchMax = midi_util.PitchMax
PitchSize = midi_util.PitchSize

PitchTokey = dict()
PitchTokey[0] = 0

for number, pitch in enumerate(range(PitchMin, PitchMax + 1)):
    PitchTokey[pitch] = number + 1

KeyToPitch = dict((number, float(pitch)) for pitch, number in PitchTokey.items())

assert len(PitchTokey) == len(KeyToPitch)

KeySize = len(PitchTokey)
OffsetStep = midi_util.OffsetStep

# OffsetSize = int(midi_util.OffsetMax / midi_util.OffsetStep) # Number of offset in a bar
# CycleTimes = 16
# CycleLength = OffsetSize * CycleTimes # Represent a Cycle of music pattern
# OffsetBitSize = int(np.log2(CycleLength))

def prepare_song_sequences(data, sequence_length, modify_num=0):
    
    songs = []

    for notes in data:

        song_batches = []

        keys = normalize_to(np.array([PitchTokey[pitch[0]] for pitch in notes]))
        acps = normalize_to(np.array([PitchTokey[pitch[1]] for pitch in notes]))

        for i in range(0, len(notes), sequence_length):
            key_seq = keys[i: i + sequence_length]
            acp_seq = acps[i: i + sequence_length]

            k_dim = key_seq.shape[0]
            a_dim = acp_seq.shape[0]

            fix_key_seq = np.zeros((sequence_length))
            fix_acp_seq = np.zeros((sequence_length))

            fix_key_seq[:k_dim] = key_seq[:]
            fix_acp_seq[:a_dim] = acp_seq[:]

            seq_target = np.ones((1))

            if np.all(fix_key_seq == 0):
                seq_target = 0

            if np.all(fix_acp_seq == 0):
                seq_target = 0

            fix_key_seq = np.reshape(fix_key_seq, (sequence_length, 1))
            fix_acp_seq = np.reshape(fix_acp_seq, (sequence_length, 1))

            song_batches.append((fix_key_seq, fix_acp_seq, seq_target))

        songs.append(song_batches)

    return songs

def normalize_to(data):
    return data * 2 / (KeySize - 1) - 1 # tanh
    # return data / (KeySize - 1)

def normalize_back(data):
    return np.rint((data + 1) * (KeySize - 1) / 2).astype(int)

def to_onehot(array, num_classes):
    return np_utils.to_categorical(array, num_classes=num_classes) 

def random_modify(key_sequence):
    key_sequence = key_sequence.copy()
    random_index = np.random.randint(0, key_sequence.shape[0])
    random_key = np.clip(key_sequence[random_index] + np.random.randint(-2, 3) * 2, 0, KeySize - 1)
    key_sequence[random_index] = random_key
    return key_sequence

def random_pattern(data, frequency=0.5):
    key_pattern = np.zeros(data[0].shape[1])
    press_pattern = np.zeros(data[1].shape[1])
    accomp_pattern = np.zeros(data[3].shape[1])

    for idx, _ in enumerate(zip(key_pattern, press_pattern)):
        if idx != 0 and np.random.uniform(0, 1) > frequency:
            press_pattern[idx] = (press_pattern[idx - 1] + 1) % PressSize
            key_pattern[idx] = key_pattern[idx - 1]
        else:
            press_pattern[idx] = 0
            key_pattern[idx] = np.random.randint(0, KeySize)

        accomp_pattern[idx] = np.random.randint(0, KeySize)

    offset_pattern = data[2][0]
    return key_pattern, press_pattern, offset_pattern, accomp_pattern

def random_pattern_from_data(data):
    start = (np.random.randint(0, data[0].shape[0]) // data[0].shape[1]) * data[0].shape[1]
    key_pattern = data[0][start]
    ofs_pattern = data[1][start]
    acp_pattern = data[2][start]
    return key_pattern, acp_pattern, ofs_pattern

def create_midi(melody, accomp, midi_name=None, scale_name=None):

    melody = normalize_back(melody)
    accomp = normalize_back(accomp)

    output_part1 = []
    output_part2 = []

    offset = 0

    # create note and chord objects based on the values generated by the model
    for key in melody:

        pitch = KeyToPitch[key]
        
        if pitch != 0:
            new_note = note.Note(pitch)
        else:
            new_note = note.Rest()

        new_note.offset = offset
        new_note.duration.quarterLength = OffsetStep
        new_note.storedInstrument = instrument.Piano()
        output_part1.append(new_note)

        offset += OffsetStep

    offset = 0

    for key in accomp:

        pitch = KeyToPitch[key]

        if pitch != 0:
            new_note = note.Note(pitch)
        else:
            new_note = note.Rest()

        new_note.offset = offset
        new_note.duration.quarterLength = OffsetStep
        new_note.storedInstrument = instrument.Piano()
        output_part2.append(new_note)

        offset += OffsetStep

    p1 = stream.Part(output_part1)
    p2 = stream.Part(output_part2)
    midi_stream = stream.Stream()
    midi_stream.insert(0, p1)
    midi_stream.insert(0, p2)

    if scale_name:
        midi_stream = midi_util.to_major(midi_stream, scale_name)

    if midi_name:
        midi_stream.write("midi", fp= midi_name + ".mid")
    else:
        midi_stream.write("midi", fp="test_output.mid")