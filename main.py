import glob
import pickle
import numpy as np
from keras.utils import np_utils
from music21 import converter, instrument, note, chord

import network
import midi_util

def prepare_sequences(data):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 64

    # get all pitch names
    pitchnames = sorted(set([note for notes in data for note in notes]))
    n_vocab = len(pitchnames)

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for notes in data:
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

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

    # network_input.shape: (sequnce_numbers, sequnce_length, 1)
    # network_output.shape: (sequnce_numbers, one-hot vector's length)

    # create model with/without weights file
    weights_path = "weights/weights-improvement-61-1.0923-bigger.hdf5"
    model = network.create(network_input, n_vocab, weights_path=weights_path)

    network.train(model, network_input, network_output)

