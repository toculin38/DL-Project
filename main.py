import glob
import pickle
import numpy as np
from keras.utils import np_utils
from music21 import converter, instrument, note, chord

import network
import midi_util

def prepare_sequences(data):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 32
    duration_min = midi_util.DurationMin
    duration_max = midi_util.DurationMax
    # get all pitch names
    pitches = midi_util.PitchTable
    n_vocab = len(pitches)
    
    print(pitches)
    print(n_vocab)

    # create a dictionary to map pitches to integers
    ps_to_int = dict((pitch, number) for number, pitch in enumerate(pitches))
    
    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for notes in data:
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([np.array([ps_to_int[note[0]], (note[1] // duration_min - 1)]) for note in sequence_in])
            network_output.append(np.array([ps_to_int[sequence_out[0]], (sequence_out[1] // duration_min - 1)]))

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 2))

    n_patterns = len(network_output)
    network_output = np.reshape(network_output, (n_patterns, 2))

    # normalize input
    network_input[:,:,0] = network_input[:,:,0] / float(n_vocab)
    network_input[:,:,1] = network_input[:,:,1] / float(duration_max // duration_min)

    output_ps = np_utils.to_categorical(network_output[:,0],  num_classes=n_vocab)
    # output_dt = np_utils.to_categorical(network_output[:,1],  num_classes=int(duration_max // duration_min))

    # network_output = np.concatenate((output_ps, output_dt), axis=1)
    network_output = output_ps
    return (network_input, network_output, n_vocab)

if __name__ == '__main__':
    # parse midi songs to notes file
    midi_path = "midi_songs/*.mid"
    data_path = "midi_input/data"

    if glob.glob(data_path):
        data = midi_util.load_data(data_path)
    else:
        data = midi_util.parse_midi(midi_path, data_path)

    network_input, network_output, n_vocab = prepare_sequences(data)

    output_length = network_output.shape[1]
    # create model with/without weights file
    model = network.create(network_input, output_length, weights_path=None)

    network.train(model, network_input, network_output)

