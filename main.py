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
    
    sequence_length = 32

    if args.weights:
        GAN_model, d_model , g_model = gan_network.build_GAN(KeySize, sequence_length, d_path=args.weights + "D.hdf5", g_path=args.weights + "G.hdf5")
    else:
        GAN_model, d_model , g_model = gan_network.build_GAN(KeySize, sequence_length)

        if args.generate:
            print('Warning: generating music without trained weights')

    print("preparing sequences..")
    songs = prepare_song_sequences(data, sequence_length)

    def test_song(index):
        song = songs[index]
        
        melody = np.concatenate(np.array([song_batch[0] for song_batch in song]))
        accomp = np.concatenate(np.array([song_batch[1] for song_batch in song]))

        melody = np.concatenate(melody) # (N, 1) to (N,)
        accomp = np.concatenate(accomp) # (N, 1) to (N,)

        create_midi(melody, accomp)

    test_song(5)

    if args.train:
        for epoch in range(1000):
            print('Epoch: {} GAN training...'.format(epoch))
            gan_network.train(GAN_model, g_model, d_model, epoch, songs, sequence_length, g_name="G", d_name="D")

            if epoch % 1 == 0:
                print('generating...')
                melody, accomp = gan_network.generate(g_model, sequence_length)
                create_midi(melody, accomp, "training midi-{}".format(epoch))

    if args.generate:
        melody, accomp = gan_network.generate(g_model, sequence_length)
        print(melody)
        create_midi(melody, accomp)
