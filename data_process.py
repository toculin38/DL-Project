import numpy as np
from keras.utils import np_utils
from music21 import stream, note, chord, instrument, midi
import midi_util

PitchMin = midi_util.PitchMin
PitchMax = midi_util.PitchMax

PitchTokey = dict((float(pitch), number+1) for number, pitch in enumerate(range(PitchMin, PitchMax + 1)))
PitchTokey[0] = 0

KeyToPitch = dict((number+1, float(pitch)) for number, pitch in enumerate(range(PitchMin, PitchMax + 1)))
KeyToPitch[0] = 0

KeySize = len(PitchTokey)
PressSize = int(midi_util.OffsetMax / midi_util.OffsetStep) # Number of offset in a bar
CycleTimes = 8
CycleLength = PressSize * CycleTimes # Represent a Cycle of music pattern
OffsetStep = midi_util.OffsetStep
OffsetBitSize = int(np.log2(CycleLength))
def prepare_sequences(data, sequence_length):
    """ Prepare the sequences used by the Neural Network """
    # create a dictionary to map pitches to integers
    keys_input = []
    keys_output = []
    press_input = []
    press_output = []

    keys2_input = []
    keys2_output = []
    press2_input = []
    press2_output = []

    offset_input = []
    # create input sequences and the corresponding outputs
    for notes in data:
        key_sequence = np_utils.to_categorical([PitchTokey[note[0]] for note in notes], num_classes=KeySize) 
        press_sequence = np_utils.to_categorical([note[1] - 1 for note in notes], num_classes=PressSize) 

        key2_sequence = np_utils.to_categorical([PitchTokey[note[2]] for note in notes], num_classes=KeySize) 
        press2_sequence = np_utils.to_categorical([note[3] - 1 for note in notes], num_classes=PressSize) 
        
        offset_sequence = np.unpackbits(np.array([[idx % CycleLength] for idx, note in enumerate(notes)], dtype=np.uint8), axis=-1)[:,-OffsetBitSize:]

        for i in range(0, len(notes) - sequence_length - 1, 1):
            key_sequence_in = key_sequence[i:i + sequence_length]
            key_sequence_out = key_sequence[i + 1: i + 1 + sequence_length]
            press_sequence_in = press_sequence[i:i + sequence_length]
            press_sequence_out = press_sequence[i + 1 : i + 1 + sequence_length]

            random_index = np.random.randint(1, key_sequence_in.shape[0])
            random_key = np.random.randint(1, key_sequence_in.shape[1])
            key_sequence_in[random_index] = np.zeros_like(key_sequence_in.shape[1])
            key_sequence_in[random_index][random_key] = 1

            random_index = np.random.randint(1, key_sequence_out.shape[0])
            random_key = np.random.randint(1, key_sequence_out.shape[1])
            key_sequence_out[random_index] = np.zeros_like(key_sequence_out.shape[1])
            key_sequence_out[random_index][random_key] = 1

            keys_input.append(key_sequence_in)
            keys_output.append(key_sequence_out)
            press_input.append(press_sequence_in)
            press_output.append(press_sequence_out)

            key2_sequence_in = key2_sequence[i:i + sequence_length]
            key2_sequence_out = key2_sequence[i + 1 : i + 1 + sequence_length]
            keys2_input.append(key2_sequence_in)
            keys2_output.append(key2_sequence_out)

            press2_sequence_in = press2_sequence[i:i + sequence_length]
            press2_sequence_out = press2_sequence[i + 1 : i + 1 + sequence_length]
            press2_input.append(press2_sequence_in)
            press2_output.append(press2_sequence_out)

            offset_sequence_in = offset_sequence[i:i + sequence_length]
            offset_input.append(offset_sequence_in)

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(keys_input)
    keys_input = np.reshape(keys_input, (n_patterns, sequence_length, KeySize))

    n_patterns = len(keys_output)
    keys_output = np.reshape(keys_output, (n_patterns, sequence_length, KeySize))

    n_patterns = len(press_input)
    press_input = np.reshape(press_input, (n_patterns, sequence_length, PressSize))

    n_patterns = len(press_output)
    press_output = np.reshape(press_output, (n_patterns , sequence_length, PressSize))

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(keys2_input)
    keys2_input = np.reshape(keys2_input, (n_patterns, sequence_length, KeySize))

    n_patterns = len(keys2_output)
    keys2_output = np.reshape(keys2_output, (n_patterns, sequence_length, KeySize))

    n_patterns = len(press2_input)
    press2_input = np.reshape(press2_input, (n_patterns, sequence_length, PressSize))

    n_patterns = len(press2_output)
    press2_output = np.reshape(press2_output, (n_patterns , sequence_length, PressSize))

    n_patterns = len(offset_input)
    offset_input = np.reshape(offset_input, (n_patterns, sequence_length, OffsetBitSize))

    return [keys_input, press_input, offset_input] , [keys_output, press_output, keys2_output, press2_output]

def generate_notes(model, data):
    """ Generate notes from the neural network based on a sequence of notes """

    # random pattern
    # pattern = np.vstack((random_seq_pitch, random_seq_duraion)).T
    random_pitch_indices = np.random.randint(data[0].shape[2], size=data[0].shape[1])
    random_pitch_indices = np.random.randint(data[1].shape[2], size=data[1].shape[1])
    key_pattern = np_utils.to_categorical(random_pitch_indices, num_classes=data[0].shape[2])
    press_pattern = np_utils.to_categorical(random_pitch_indices, num_classes=data[1].shape[2])
    offset_pattern = data[2][0]
    # key2_pattern = np_utils.to_categorical(random_pitch_indices, num_classes=data[2].shape[2])
    # press2_pattern = np_utils.to_categorical(random_pitch_indices, num_classes=data[3].shape[2])

    # random sequence in key_data
    # start = (np.random.randint(0, data[0].shape[0]-1))
    # key_pattern = data[0][start]
    # press_pattern = data[1][start]
    # offset_pattern = data[2][start]
    # key2_pattern = data[2][start]
    # press2_pattern = data[3][start]

    prediction_output = []
    print(key_pattern)

    # generate 512 offset
    for _ in range(128):
        # random modify the pattern to prevent looping
        random_offset_index = np.random.randint(0, key_pattern.shape[0]-1)
        random_pitch_index = np.random.randint(0, key_pattern.shape[1])
        copy_pattern = np.copy(key_pattern)
        copy_pattern[random_offset_index] = np_utils.to_categorical(random_pitch_index, num_classes=key_pattern.shape[1])

        key_input = np.reshape(copy_pattern, (1, key_pattern.shape[0], key_pattern.shape[1]))
        press_input = np.reshape(press_pattern, (1, press_pattern.shape[0], press_pattern.shape[1]))
        # key2_input = np.reshape(key2_pattern, (1, key2_pattern.shape[0], key2_pattern.shape[1]))
        # press2_input = np.reshape(press2_pattern, (1, press2_pattern.shape[0], press2_pattern.shape[1]))
        offset_input = np.reshape(offset_pattern, (1, offset_pattern.shape[0], offset_pattern.shape[1]))
        prediction_input = [key_input, press_input, offset_input]
        prediction = model.predict(prediction_input, verbose=0)

        key_prediction = np.reshape(prediction[0], (prediction[0].shape[1], prediction[0].shape[2]))
        press_prediction = np.reshape(prediction[1], (prediction[1].shape[1], prediction[1].shape[2]))
        key2_prediction = np.reshape(prediction[2], (prediction[2].shape[1], prediction[2].shape[2]))
        press2_prediction = np.reshape(prediction[3], (prediction[3].shape[1], prediction[3].shape[2]))

        # for index, (key_onehot, press_onehot, key2_onehot, press2_onehot) in enumerate(zip(key_prediction, press_prediction)):
        #     key_index = np.argmax(key_onehot)
        #     press_index = np.argmax(press_onehot)
        #     key2_index = np.argmax(key2_onehot)
        #     press2_index = np.argmax(press2_onehot)

        #     key_pattern[index] = np.zeros_like(key_onehot)
        #     key_pattern[index][key_index] = 1
        #     press_pattern[index] = np.zeros_like(press_onehot)
        #     press_pattern[index][press_index] = 1

        #     key2_pattern[index] = np.zeros_like(key2_onehot)
        #     key2_pattern[index][key2_index] = 1
        #     press2_pattern[index] = np.zeros_like(press2_onehot)
        #     press2_pattern[index][press2_index] = 1

        #     prediction_output.append((KeyToPitch[key_index], press_index + 1, KeyToPitch[key_index2], press_index2 + 1))

        key_onehot = key_prediction[-1]
        press_onehot = press_prediction[-1]
        key2_onehot = key2_prediction[-1]
        press2_onehot = press2_prediction[-1]

        key_index = np.argmax(key_onehot)
        press_index = np.argmax(press_onehot)
        key2_index = np.argmax(key2_onehot)
        press2_index = np.argmax(press2_onehot)

        prediction_output.append((KeyToPitch[key_index], press_index + 1, KeyToPitch[key2_index], press2_index + 1))

        key_pattern[0:-1] = key_pattern[1:]
        key_pattern[-1] = np.zeros_like(key_onehot)
        key_pattern[-1][key_index] = 1
        press_pattern[0:-1] = press_pattern[1:]
        press_pattern[-1] = np.zeros_like(press_onehot)
        press_pattern[-1][press_index] = 1

        # key2_pattern[0:-1] = key2_pattern[1:]
        # key2_pattern[-1] = np.zeros_like(key2_onehot)
        # key2_pattern[-1][key2_index] = 1
        # press2_pattern[0:-1] = press2_pattern[1:]
        # press2_pattern[-1] = np.zeros_like(press2_onehot)
        # press2_pattern[-1][press2_index] = 1

        shift_offset = 1
        new_offset_pattern = np.packbits(offset_pattern, axis=-1) // (2 ** (8 - OffsetBitSize)) # pack to int
        new_offset_pattern = (new_offset_pattern + shift_offset) % CycleLength # shift
        new_offset_pattern = np.unpackbits(new_offset_pattern, axis=-1) # unpack to 8 bits
        offset_pattern = new_offset_pattern[:,-OffsetBitSize:] # select indices 1 ~ 7 bits (discard 0) as new pattern

    return prediction_output

def create_midi(prediction_output, scale_name=None):
    offset = 0
    print(prediction_output)

    output_part1 = []
    output_part2 = []
    # create note and chord objects based on the values generated by the model
    for (pitch, press, pitch2, press2) in prediction_output:

        if len(output_part1) > 0 and press != 1:
            output_part1[-1].duration.quarterLength += OffsetStep
        else:
            if pitch != 0:
                new_note = note.Note(pitch)
            else:
                new_note = note.Rest()

            new_note.offset = offset
            new_note.duration.quarterLength = OffsetStep
            new_note.storedInstrument = instrument.Piano()
            output_part1.append(new_note)

        if len(output_part2) > 0 and press2 != 1:
            output_part2[-1].duration.quarterLength += OffsetStep
        else:
            if pitch2 != 0:
                new_note = note.Note(pitch2)
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

    midi_stream.write('midi', fp='test_output.mid')

    stream_player = midi.realtime.StreamPlayer(midi_stream)
    stream_player.play()