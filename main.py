import glob
import pickle
import numpy as np
from keras.utils import np_utils
from music21 import converter, instrument, note, chord

import network
import midi_util

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
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

    return (network_input, network_output)

if __name__ == '__main__':
    # parse midi songs to notes file
    midi_path = "midi_songs/*.mid"
    notes_path = "midi_input/notes"

    if glob.glob(notes_path):
        notes = midi_util.load_notes(notes_path)
    else:
        notes = midi_util.parse_midi(midi_path, notes_path)

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)
    # network_input.shape: (sequnce_numbers, sequnce_length, 1)
    # network_output.shape: (sequnce_numbers, one-hot vector's length)

    # create model with/without weights file
    weights_path = "weights/weights-improvement-01-4.7159-bigger.hdf5"
    model = network.create(network_input, n_vocab, weights_path=weights_path)

    network.train(model, network_input, network_output)

