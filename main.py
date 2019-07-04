import glob
import pickle
import numpy as np

from music21 import converter, instrument, note, chord, stream

import network
import midi_util
from data_process import *

if __name__ == '__main__':
    # parse midi songs to notes file
    midi_path = "midi_songs/4-4/*.mid"
    data_path = "midi_input/data"

    if glob.glob(data_path):
        data = midi_util.load_data(data_path)
    else:
        data = midi_util.parse_midi(midi_path, data_path)

    sequence_length = 64

    keyboard_table = midi_util.PitchTokey
    keyboard_size = len(keyboard_table)

    print(keyboard_table)
    print(keyboard_size)

    duration_min = midi_util.DurationMin
    duration_max = midi_util.DurationMax
    duration_size = int(duration_max // duration_min)

    network_input, network_output = prepare_sequences(data, sequence_length)
    print(network_input.shape)
    print(network_output.shape)
    # create model with/without weights file
    model = network.create(network_input, keyboard_size, weights_path="weights/weights-44-0.1837.hdf5")
    network.train(model, network_input, network_output)

    # generate midi
    # prediction_output = generate_notes(model, network_input)
    # create_midi(prediction_output, 'D')
