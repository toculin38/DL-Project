import argparse
import glob
import pickle
import numpy as np
import os
from music21 import converter, instrument, note, chord, stream

import gan_network
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
    parser.add_argument('--weights', type=str, help='melody weights path')
    args = parser.parse_args()

    # parse midi songs to notes file
    data = prepare_data()

    sequence_length = 64

    if args.weights:
        g_model = gan_network.create_generate_model(PitchSize, weights_path=args.weights + "G.hdf5")
        d_model = gan_network.create_discrimi_model(PitchSize, weights_path=args.weights + "D.hdf5")
    else:
        g_model = gan_network.create_generate_model(PitchSize, weights_path=None)
        d_model = gan_network.create_discrimi_model(PitchSize, weights_path=None)
        if args.generate:
            print('Warning: generating music without trained weights')

    print("preparing sequences..")
    seq_data = prepare_song_sequences(data)

    if args.train:
        for epoch in range(1000):
            print('Epoch: {} GAN training...'.format(epoch))
            gan_network.train(g_model, d_model, epoch, seq_data, g_name="G", d_name="D")

            if epoch % 1 == 0:
                print('generating...')
                melody, accomp = gan_network.generate(g_model)
                create_midi(melody[0], accomp[0], "training midi-{}".format(epoch))

    if args.generate:
        print('generating...')
        melody, accomp = gan_network.generate(g_model)
        create_midi(melody[0], accomp[0])
