import argparse
import glob
import pickle
import numpy as np

from music21 import converter, instrument, note, chord, stream

import network
import midi_util
from data_process import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Music Generator by LSTM')
    parser.add_argument('--train', action='store_true', help='train the music generator')
    parser.add_argument('--generate', action='store_true', help='generate the music')
    parser.add_argument('--weights', type=str, help='weights path')
    args = parser.parse_args()

    # parse midi songs to notes file
    midi_path = "midi_songs/4-4/*.mid"
    molody_path = "midi_input/melody_data"
    accomp_path = "midi_input/accomp_data"

    if glob.glob(molody_path):
        molody_data = midi_util.load_data(molody_path)
    else:
        molody_data = midi_util.parse_midi(midi_path, molody_path, part_index=0)

    # if glob.glob(accomp_path):
    #     accomp_data = midi_util.load_data(accomp_path)
    # else:
    #     accomp_data = midi_util.parse_midi(midi_path, accomp_path, part_index=1)

    sequence_length = 16

    key_data, key_target, offset_data, press_data, press_target = prepare_sequences(molody_data, sequence_length)

    key_size = key_target.shape[1]
    press_size = press_target.shape[1]

    if args.weights:
        melody_model = network.create(key_data, press_data, offset_data, weights_path=args.weights)
    else:
        melody_model = network.create(key_data, press_data, offset_data, weights_path=None)
        if args.generate:
            print('Warning: generating music without trained weights')

    if args.train:
        print('training...')
        network.train(melody_model, key_data, press_data, offset_data, key_target, press_target)

    if args.generate:
        print('generating...')
        prediction_output = generate_notes(melody_model, key_data, press_data, offset_data)
        create_midi(prediction_output)
