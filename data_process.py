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
def prepare_sequences(data, sequence_length, modify_num=0):
    keys_input = []
    press_input = []
    offset_input = []
    keys_output = []
    press_output = []
    keys2_output = []
    press2_output = []
    
    # create input sequences and the corresponding outputs
    for notes in data:
        key_sequence = np.array([PitchTokey[note[0]] for note in notes])
        press_sequence = np.array([note[1] - 1 for note in notes])
        key2_sequence = np.array([PitchTokey[note[2]] for note in notes])
        press2_sequence = np.array([note[3] - 1 for note in notes])
        offset_sequence = np.unpackbits(np.array([[idx % CycleLength] for idx, note in enumerate(notes)], dtype=np.uint8), axis=-1)[:,-OffsetBitSize:]

        for i in range(0, len(notes) - 2 * sequence_length, 1):
            key_sequence_in = key_sequence[i:i + sequence_length]
            key_sequence_out = key_sequence[i + sequence_length: i + 2 * sequence_length]
            key2_sequence_out = key2_sequence[i:i + sequence_length]
            press_sequence_in = press_sequence[i:i + sequence_length]
            press_sequence_out = press_sequence[i + sequence_length : i + 2 * sequence_length]
            press2_sequence_out = press2_sequence[i:i + sequence_length]
            offset_sequence_in = offset_sequence[i:i + sequence_length]

            for _ in range(modify_num):
                key_sequence_in, press_sequence_in = random_modify(key_sequence_in, press_sequence_in)

            for _ in range(modify_num):
                key_sequence_out, press_sequence_out = random_modify(key_sequence_out, press_sequence_out)

            keys_input.append(key_sequence_in)
            keys_output.append(key_sequence_out)
            keys2_output.append(key2_sequence_out)
            press_input.append(press_sequence_in)
            press_output.append(press_sequence_out)
            press2_output.append(press2_sequence_out)
            offset_input.append(offset_sequence_in)

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(keys_input)
    keys_input = to_onehot(np.array(keys_input), KeySize)
    keys_input = np.reshape(keys_input, (n_patterns, sequence_length, KeySize))
    press_input = to_onehot(np.array(press_input), PressSize)
    press_input = np.reshape(press_input, (n_patterns, sequence_length, PressSize))
    offset_input = np.reshape(offset_input, (n_patterns, sequence_length, OffsetBitSize))

    keys_output = to_onehot(np.array(keys_output), KeySize)
    keys_output = np.reshape(keys_output, (n_patterns, sequence_length, KeySize))
    press_output = to_onehot(np.array(press_output), PressSize)
    press_output = np.reshape(press_output, (n_patterns , sequence_length, PressSize))

    keys2_output = to_onehot(np.array(keys2_output), KeySize)
    keys2_output = np.reshape(keys2_output, (n_patterns, sequence_length, KeySize))
    press2_output = to_onehot(np.array(press2_output), PressSize)
    press2_output = np.reshape(press2_output, (n_patterns , sequence_length, PressSize))

    return [keys_input, press_input, offset_input], [keys_output, press_output], [keys2_output, press2_output]

def to_onehot(array, num_classes):
    return np_utils.to_categorical(array, num_classes=num_classes) 

def random_modify(key_sequence, press_sequence):
    key_sequence = key_sequence.copy()
    press_sequence = press_sequence.copy()
    random_index = np.random.randint(0, key_sequence.shape[0])
    random_key = np.clip(key_sequence[random_index] + np.random.randint(-4, 5), 0, KeySize - 1)
    key_sequence[random_index] = random_key

    press = press_sequence[random_index]
    press_sequence[random_index] = 0
    
    for idx in range(random_index + 1, len(press_sequence)):
        if press_sequence[idx] == 0:
            break
        else:
            press_sequence[idx] -= press

    return key_sequence, press_sequence

def random_pattern(data):
    key_pattern = np.zeros(data[0].shape[1])
    press_pattern = np.zeros(data[1].shape[1])

    for idx, _ in enumerate(zip(key_pattern, press_pattern)):
        if idx != 0 and np.random.uniform(0, 1) < 0.75:
            press_pattern[idx] = (press_pattern[idx - 1] + 1) % PressSize
            key_pattern[idx] = key_pattern[idx - 1]
        else:
            press_pattern[idx] = 0
            key_pattern[idx] = np.random.randint(13, KeySize - 12)

    offset_pattern = data[2][0]
    return key_pattern, press_pattern, offset_pattern

def generate_notes(melody_model, accomp_model, data):
    key_pattern , prs_pattern, offset_pattern = random_pattern(data)
    print(key_pattern)
    print(prs_pattern)

    melody_output = []
    accomp_output = []

    # generate 64 bars
    for _ in range(32):
        # random modify the pattern to prevent looping
        copy_key_pattern, copy_prs_pattern = random_modify(key_pattern, prs_pattern)

        # predict melody
        key_input = np_utils.to_categorical(key_pattern, num_classes=KeySize)
        key_input = np.reshape(key_input, (1, key_input.shape[0], key_input.shape[1]))
        prs_input = np_utils.to_categorical(prs_pattern, num_classes=PressSize)
        prs_input = np.reshape(prs_input, (1, prs_input.shape[0], prs_input.shape[1]))
        offset_input = np.reshape(offset_pattern, (1, offset_pattern.shape[0], offset_pattern.shape[1]))
        
        prediction_input = [key_input, prs_input, offset_input]
        prediction = melody_model.predict(prediction_input, verbose=0)
        key_prediction = np.reshape(prediction[0], (prediction[0].shape[1], prediction[0].shape[2]))
        prs_prediction = np.reshape(prediction[1], (prediction[1].shape[1], prediction[1].shape[2]))

        for index, (key_onehot, prs_onehot) in enumerate(zip(key_prediction, prs_prediction)):
            key_index = np.argmax(key_onehot)
            prs_index = np.argmax(prs_onehot)
            melody_output.append((KeyToPitch[key_index], prs_index))
            # adjust pattern
            key_pattern[index] = key_index
            prs_pattern[index] = prs_index

        # adjust offset pattern
        shift_offset = len(offset_pattern)
        new_offset_pattern = np.packbits(offset_pattern, axis=-1) // (2 ** (8 - OffsetBitSize)) # pack to int
        new_offset_pattern = (new_offset_pattern + shift_offset) % CycleLength # shift
        new_offset_pattern = np.unpackbits(new_offset_pattern, axis=-1) # unpack to 8 bits
        offset_pattern = new_offset_pattern[:,-OffsetBitSize:] # select indices 1 ~ 7 bits (discard 0) as new pattern

        # predict accomp
        key_input = np_utils.to_categorical(key_pattern, num_classes=KeySize)
        key_input = np.reshape(key_input, (1, key_input.shape[0], key_input.shape[1]))
        prs_input = np_utils.to_categorical(prs_pattern, num_classes=PressSize)
        prs_input = np.reshape(prs_input, (1, prs_input.shape[0], prs_input.shape[1]))
        offset_input = np.reshape(offset_pattern, (1, offset_pattern.shape[0], offset_pattern.shape[1]))

        prediction_input = [key_input, prs_input, offset_input]
        prediction = accomp_model.predict(prediction_input, verbose=0)
        key_prediction = np.reshape(prediction[0], (prediction[0].shape[1], prediction[0].shape[2]))
        prs_prediction = np.reshape(prediction[1], (prediction[1].shape[1], prediction[1].shape[2]))

        for index, (key_onehot, prs_onehot) in enumerate(zip(key_prediction, prs_prediction)):
            key_index = np.argmax(key_onehot)
            prs_index = np.argmax(prs_onehot)
            accomp_output.append((KeyToPitch[key_index], prs_index))

    return melody_output, accomp_output

def create_midi(melody_output, accomp_output, scale_name=None):
    output_part1 = []
    output_part2 = []

    print(melody_output)
    offset = 0
    # create note and chord objects based on the values generated by the model
    for (pitch, press) in melody_output:
        if len(output_part1) > 0 and press != 0:
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
        # increase offset each iteration so that notes do not stack
        offset += OffsetStep

    print(accomp_output)
    offset = 0
    for (pitch, press) in accomp_output:
        if len(output_part2) > 0 and press != 0:
            output_part2[-1].duration.quarterLength += OffsetStep
        else:
            if pitch != 0:
                new_note = note.Note(pitch)
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