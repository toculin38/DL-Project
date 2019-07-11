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
    parser.add_argument('--melody_weights', type=str, help='melody weights path')
    parser.add_argument('--accomp_weights', type=str, help='accomp weights path')
    args = parser.parse_args()

    # parse midi songs to notes file
    data = prepare_data()

    sequence_length = 64

    if args.melody_weights:
        melody_model = network.create_molody_model(sequence_length, KeySize, PressSize, OffsetBitSize, weights_path=args.melody_weights)
    else:
        melody_model = network.create_molody_model(sequence_length, KeySize, PressSize, OffsetBitSize, weights_path=None)
        if args.generate:
            print('Warning: generating music without trained weights')

    if args.accomp_weights:
        accomp_model = network.create_accomp_model(sequence_length, KeySize, PressSize, OffsetBitSize, weights_path=args.accomp_weights)
    else:
        accomp_model = network.create_accomp_model(sequence_length, KeySize, PressSize, OffsetBitSize, weights_path=None)
        if args.generate:
            print('Warning: generating music without trained weights')

    if args.train:
        for epoch in range(1000):
            print('preparing sequence...')
            train_data, melody_target, accomp_target = prepare_sequences(data, sequence_length, modify_num=0)
            print('Epoch: {} melody training...'.format(epoch))
            network.train(melody_model, epoch, train_data, melody_target, model_name="Melody")
            print('preparing sequence...')
            train_data, melody_target, accomp_target = prepare_sequences(data, sequence_length, modify_num=0)
            print('Epoch: {} Accomp training...'.format(epoch))
            network.train(accomp_model, epoch, train_data, accomp_target, model_name="Accomp")

    if args.generate:
        print('preparing sequence...')
        train_data, melody_target, accomp_target = prepare_sequences(data, sequence_length)
        print('generating...')
        melody_output, accomp_output = generate_notes(melody_model, accomp_model, train_data)
        create_midi(melody_output, accomp_output)
