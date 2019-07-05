import numpy as np
from keras.utils import np_utils
from music21 import stream, note, chord, instrument
import midi_util

PitchMin = midi_util.PitchMin
PitchMax = midi_util.PitchMax

PitchTokey = dict((float(pitch), number+1) for number, pitch in enumerate(range(PitchMin, PitchMax + 1)))
PitchTokey[0] = 0

KeyToPitch = dict((number+1, float(pitch)) for number, pitch in enumerate(range(PitchMin, PitchMax + 1)))
KeyToPitch[0] = 0

KeySize = len(PitchTokey)
PressSize = int(midi_util.OffsetMax / midi_util.OffsetStep) 
OffsetSize = int(np.log2(PressSize))
OffsetStep = midi_util.OffsetStep

def prepare_sequences(data, sequence_length):
    """ Prepare the sequences used by the Neural Network """
    # create a dictionary to map pitches to integers
    keys_input = []
    keys_output = []
    press_input = []
    press_output = []
    offset_input = []
    # create input sequences and the corresponding outputs
    for notes in data:
        key_sequence = np_utils.to_categorical([PitchTokey[note[0]] for note in notes], num_classes=KeySize) 
        press_sequence = np_utils.to_categorical([note[1] - 1 for note in notes], num_classes=PressSize) 
        offset_sequence = np.unpackbits(np.array([[note[2]] for note in notes], dtype=np.uint8), axis=-1)[:,-OffsetSize:]
        for i in range(0, len(notes) - 2 * sequence_length, 1):
            key_sequence_in = key_sequence[i:i + sequence_length]
            key_sequence_out = key_sequence[i + sequence_length: i + 2 * sequence_length]
            keys_input.append(key_sequence_in)
            keys_output.append(key_sequence_out)

            press_sequence_in = press_sequence[i:i + sequence_length]
            press_sequence_out = press_sequence[i + sequence_length: i + 2 * sequence_length]
            press_input.append(press_sequence_in)
            press_output.append(press_sequence_out)

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

    n_patterns = len(offset_input)
    offset_input = np.reshape(offset_input, (n_patterns, sequence_length, OffsetSize))

    return (keys_input, keys_output, offset_input, press_input, press_output)

# def sequence_to_onehot(notes_sequence):
#     key_index = PitchTokey[pitch_space]
#     onehot = np.zeros(len(PitchTokey))
#     onehot[key_index] = 1
#     return onehot

def generate_notes(model, key_data, press_data, offset_data):
    """ Generate notes from the neural network based on a sequence of notes """
    pitch_size = key_data.shape[2]

    # random pattern
    # pattern = np.vstack((random_seq_pitch, random_seq_duraion)).T
    # random_pitch_indices = np.random.randint(pitch_size, size=key_data.shape[1])
    # key_pattern = np_utils.to_categorical(random_pitch_indices, num_classes=pitch_size)
    # random sequence in key_data
    start = np.random.randint(0, len(key_data)-1)
    key_pattern = key_data[start]
    press_pattern = press_data[start]
    offset_pattern = offset_data[start]
    print(key_pattern.shape)
    print(press_pattern.shape)

    prediction_output = []
    
    # generate 16 measures
    for _ in range(16):

        # random modify the pattern to prevent looping
        # random_offset_index = np.random.randint(0, key_pattern.shape[0]-1)
        # random_pitch_index = np.random.randint(0, pitch_size)
        # copy_pattern = np.copy(key_pattern)
        # copy_pattern[random_offset_index] = np_utils.to_categorical(random_pitch_index, num_classes=pitch_size)

        prediction_key_input = np.reshape(key_pattern, (1, key_pattern.shape[0], key_pattern.shape[1]))
        prediction_press_input = np.reshape(press_pattern, (1, press_pattern.shape[0], press_pattern.shape[1]))
        prediction_offset_input = np.reshape(offset_pattern, (1, offset_pattern.shape[0], offset_pattern.shape[1]))
        prediction_input = [prediction_key_input, prediction_press_input, prediction_offset_input]
        prediction = model.predict(prediction_input, verbose=0)

        key_prediction = np.reshape(prediction[0], (prediction[0].shape[1], prediction[0].shape[2]))
        press_prediction = np.reshape(prediction[1], (prediction[1].shape[1], prediction[1].shape[2]))

        for index, (key_onehot, press_onehot) in enumerate(zip(key_prediction, press_prediction)):
            key_index = np.argmax(key_onehot)
            press_index = np.argmax(press_onehot)

            key_pattern[index] = np.zeros_like(key_onehot)
            key_pattern[index][key_index] = 1
            press_pattern[index] = np.zeros_like(press_onehot)
            press_pattern[index][press_index] = 1

            predict_pitch = KeyToPitch[key_index]
            predict_press = press_index + 1
            prediction_output.append((predict_pitch, predict_press))

            

        # press_index = np.argmax(prediction[1])

        # if press_index == 1 and np.random.uniform() < 0.15:
        #     key_index = np.argsort(-prediction[0][0])[1]
        # else:
        #     key_index = np.argmax(prediction[0])
            
        # predict_pitch = KeyToPitch[key_index]
        # predict_press = press_index + 1

        # prediction_output.append((predict_pitch, predict_press))

        # key_pattern[0:-1] = key_pattern[1:]
        # key_pattern[-1] = np.zeros_like(prediction[0])
        # key_pattern[-1][key_index] = 1

        # press_pattern[0:-1] = press_pattern[1:]
        # press_pattern[-1] = np.zeros_like(prediction[1])
        # press_pattern[-1][press_index] = 1

    return prediction_output

def create_midi(prediction_output, scale_name=None):
    offset = 0
    output_notes = []
    print(prediction_output)
    # create note and chord objects based on the values generated by the model
    for (pitch, press) in prediction_output:

        if len(output_notes) > 0 and press != 1:
            output_notes[-1].duration.quarterLength = OffsetStep * press
        else:
            if pitch != 0:
                new_note = note.Note(pitch)
            else:
                new_note = note.Rest()
            new_note.offset = offset
            new_note.duration.quarterLength = OffsetStep
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += OffsetStep

    midi_stream = stream.Stream(output_notes)

    if scale_name:
        midi_stream = midi_util.to_major(midi_stream, scale_name)

    midi_stream.write('midi', fp='test_output.mid')