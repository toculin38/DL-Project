import argparse
import glob
import pickle
import numpy as np
import os
from music21 import converter, instrument, note, chord, stream

import network
import midi_util
from data_process import *

def prepare_data(midi_folder="midi_songs/4-4/", save_folder="midi_input/"):
    melody_data = []

    for midi_path in glob.glob(midi_folder + "*.mid"):
        file_basename = os.path.basename(midi_path)
        file_name = os.path.splitext(file_basename)[0]
        file_path = save_folder + file_name

        if glob.glob(file_path):
            melody_data.append(midi_util.load_data(file_path))
        else:
            notes = midi_util.parse_midi(midi_path, file_path)
            if notes:
                melody_data.append(notes)
    return melody_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Music Generator by LSTM')
    parser.add_argument('--train', action='store_true', help='train the music generator')
    parser.add_argument('--generate', action='store_true', help='generate the music')
    parser.add_argument('--weights', type=str, help='weights path')
    args = parser.parse_args()

    # parse midi songs to notes file
    data = prepare_data()

    sequence_length = 64

    if args.weights:
        melody_model = network.create(sequence_length, KeySize, PressSize, OffsetBitSize, weights_path=args.weights)
    else:
        melody_model = network.create(sequence_length, KeySize, PressSize, OffsetBitSize, weights_path=None)
        if args.generate:
            print('Warning: generating music without trained weights')

    if args.train:
        for epoch in range(1000):
            print('preparing sequence...')
            train_data, train_target = prepare_sequences(data, sequence_length)
            print('training...')
            network.train(melody_model, train_data, train_target)

    if args.generate:
        print('preparing sequence...')
        train_data, train_target = prepare_sequences(data, sequence_length)
        print('generating...')
        prediction_output = generate_notes(melody_model, train_data)
        create_midi(prediction_output)
