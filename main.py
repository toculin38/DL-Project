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

    key_data, key_target, press_data, press_target = prepare_sequences(data, sequence_length)

    key_size = key_target.shape[1]
    press_size = press_target.shape[1]

    print(key_data.shape)
    print(key_target.shape)
    print(press_target.shape)


    # create model with/without weights file
    model = network.create(key_data, key_size, press_data, press_size , weights_path=None)
    network.train(model, key_data, press_data, key_target, press_target)

    # generate midi
    # prediction_output = generate_notes(model, key_data, press_data)
    # create_midi(prediction_output)
