import numpy as np
from keras.utils import np_utils
from music21 import stream, note, chord, instrument, midi
import midi_util

PitchMin = midi_util.PitchMin
PitchSize = midi_util.PitchSize

OffsetSize = int(midi_util.OffsetMax / midi_util.OffsetStep) # Number of offset in a bar
CycleTimes = 16
CycleLength = OffsetSize * CycleTimes # Represent a Cycle of music pattern
OffsetStep = midi_util.OffsetStep
OffsetBitSize = int(np.log2(CycleLength))

def prepare_song_sequences(data, modify_num=0):
    keys_input = []
    
    # create input sequences and the corresponding outputs
    for notes in data:
        key_sequence = np.array([np.concatenate((note[:,0], note[:,1]), axis=0) for note in notes])
        keys_input.append(key_sequence)

    return keys_input

def prepare_accomp_sequences(data, sequence_length, modify_num=0):
    keys_input = []
    offset_input = []
    keys2_output = []
    accomp_input = []

    # create input sequences and the corresponding outputs
    for notes in data:
        key_sequence = np.array([note[:,0] for note in notes])
        key2_sequence = np.array([note[:,1] for note in notes])
        offset_sequence = np.unpackbits(np.array([[idx % CycleLength] for idx, note in enumerate(notes)], dtype=np.uint8), axis=-1)[:,-OffsetBitSize:]

        for i in range(sequence_length, len(notes) - sequence_length, 1):
            key_sequence_in = key_sequence[i:i + sequence_length]
            key2_sequence_out = key2_sequence[i:i + sequence_length]
            offset_sequence_in = offset_sequence[i:i + sequence_length]
            accomp_sequence_in = key2_sequence[i - sequence_length: i]

            for _ in range(modify_num):
                key_sequence_in = random_modify(key_sequence_in)

            for _ in range(modify_num):
                key2_sequence_out = random_modify(key2_sequence_out)

            keys_input.append(key_sequence_in)
            keys2_output.append(key2_sequence_out)
            offset_input.append(offset_sequence_in)
            accomp_input.append(accomp_sequence_in)

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(keys_input)

    keys_input = np.reshape(keys_input, (n_patterns, sequence_length, PitchSize))
    offset_input = np.reshape(offset_input, (n_patterns, sequence_length, OffsetBitSize))
    accomp_input = np.reshape(accomp_input, (n_patterns, sequence_length, PitchSize))
    keys2_output = np.reshape(keys2_output, (n_patterns, sequence_length, PitchSize))

    return [keys_input, offset_input, accomp_input], [keys2_output]

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

def generate_notes(generator, noise):
    # key_pattern , prs_pattern, ofs_pattern, acp_pattern = random_pattern(data)
    key_pattern, acp_pattern, ofs_pattern = random_pattern_from_data(data)
    melody_output = []
    accomp_output = []
    # generate 8 * SequenceLength notes
    for _ in range(8):
        # random modify the pattern to prevent looping
        # copy_key_pattern, copy_prs_pattern = random_modify(key_pattern, prs_pattern)
        # predict melody
        key_input = key_pattern
        key_input = np.reshape(key_input, (1, key_input.shape[0], key_input.shape[1]))
        ofs_input = np.reshape(ofs_pattern, (1, ofs_pattern.shape[0], ofs_pattern.shape[1]))
        prediction = melody_model.predict([key_input, ofs_input], verbose=0)
        key_prediction = np.reshape(prediction[0], (prediction[0].shape[0], prediction[0].shape[1]))

        for index, key_onehot in enumerate(key_prediction):
            key_onehot = np.round(key_onehot)
            key_pattern[index] = key_onehot
            melody_output.append(key_onehot)
            
        # adjust offset pattern
        shift_offset = len(ofs_pattern)
        new_offset_pattern = np.packbits(ofs_pattern, axis=-1) // (2 ** (8 - OffsetBitSize)) # pack to int
        new_offset_pattern = (new_offset_pattern + shift_offset) % CycleLength # shift
        new_offset_pattern = np.unpackbits(new_offset_pattern.astype(np.uint8), axis=-1) # unpack to 8 bits
        ofs_pattern = new_offset_pattern[:,-OffsetBitSize:] # select indices 1 ~ 7 bits (discard 0) as new pattern

        # predict accomp
        key_input = key_pattern
        key_input = np.reshape(key_input, (1, key_input.shape[0], key_input.shape[1]))
        acp_input = acp_pattern
        acp_input = np.reshape(acp_input, (1, acp_input.shape[0], acp_input.shape[1]))
        ofs_input = np.reshape(ofs_pattern, (1, ofs_pattern.shape[0], ofs_pattern.shape[1]))
        prediction = accomp_model.predict([key_input, ofs_input, acp_input], verbose=0)
        key_prediction = np.reshape(prediction[0], (prediction[0].shape[0], prediction[0].shape[1]))
        
        for index, key_onehot in enumerate(key_prediction):
            key_onehot = np.round(key_onehot)
            acp_pattern[index] = key_onehot
            accomp_output.append(key_onehot)

    return melody_output, accomp_output


def create_midi(melody, accomp, midi_name=None, scale_name=None):
    output_part1 = []
    output_part2 = []
    offset = 0

    # create note and chord objects based on the values generated by the model
    for onehot in melody:
        pitches = [i + PitchMin for i, x in enumerate(onehot) if x == 1]
        pitches_len = len(pitches)
        if len(output_part1) > 0 and pitches_len == 0:
            output_part1[-1].duration.quarterLength += OffsetStep
        else:
            if pitches_len == 1:
                new_note = note.Note(pitches[0])
            elif 1 < pitches_len < 10:
                new_note = chord.Chord(pitches)
            else:
                new_note = note.Rest()

            new_note.offset = offset
            new_note.duration.quarterLength = OffsetStep
            new_note.storedInstrument = instrument.Piano()
            output_part1.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += OffsetStep

    offset = 0
    for onehot in accomp:
        pitches = [i + PitchMin for i, x in enumerate(onehot) if x == 1]
        pitches_len = len(pitches)
        if len(output_part2) > 0 and pitches_len == 0:
            output_part2[-1].duration.quarterLength += OffsetStep
        else:
            if pitches_len == 1:
                new_note = note.Note(pitches[0])
            elif 1 < pitches_len < 10:
                new_note = chord.Chord(pitches)
            else:
                new_note = note.Rest()
            new_note.offset = offset
            new_note.duration.quarterLength = OffsetStep
            new_note.storedInstrument = instrument.Piano()
            output_part2.append(new_note)
        # increase offset each iteration so that notes do not stack
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

    # stream_player = midi.realtime.StreamPlayer(midi_stream)
    # stream_player.play()