

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, TimeDistributed, Dropout, Input, Activation, Concatenate, LeakyReLU, LSTM, BatchNormalization, Reshape, Bidirectional
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
from focal_losses import *

LatenSize = 32

def build_GAN(key_size, seq_len, g_path=None, d_path=None):

    noise_input = Input(batch_shape=(1, LatenSize))

    g_song_part = Sequential()

    g_song_part.add(Dense(256))
    g_song_part.add(LeakyReLU(alpha=0.2))
    g_song_part.add(BatchNormalization(momentum=0.8))
    g_song_part.add(Dense(512))
    g_song_part.add(LeakyReLU(alpha=0.2))
    g_song_part.add(BatchNormalization(momentum=0.8))
    g_song_part.add(Dense(1024))
    g_song_part.add(LeakyReLU(alpha=0.2))
    g_song_part.add(BatchNormalization(momentum=0.8))
    g_song_info = g_song_part(noise_input)

    song_out = Dense(np.prod(seq_len * 2), activation='tanh')(g_song_info)
    song_out = Reshape((seq_len, 2))(song_out)
    # song_out = LSTM(512, return_sequences=True)(song_out)
    # song_out = LSTM(512, return_sequences=True)(song_out)
    # song_out = Dense(2, activation='tanh')(song_out)

    generator = Model(inputs=[noise_input], outputs=[song_out])


    song_input = Input(batch_shape=(1, seq_len, 2))

    d_song_info = LSTM(512, return_sequences=True)(song_input)
    d_song_info = LSTM(512)(d_song_info)
    
    validate_out = Dense(512)(d_song_info)
    validate_out = LeakyReLU(alpha=0.2)(validate_out)
    validate_out = Dense(256)(validate_out)
    validate_out = LeakyReLU(alpha=0.2)(validate_out)
    validate_out = Dense(1, activation='sigmoid')(validate_out)

    optimizer = Adam(0.0002, 0.5)

    discrimnator = Model(inputs=[song_input], outputs=[validate_out])
    discrimnator.compile(loss='binary_crossentropy', optimizer=optimizer)
    discrimnator.summary()

    gan_out = discrimnator(song_out)

    discrimnator.trainable = False
    GAN_model = Model(inputs=[noise_input], outputs=[gan_out])
    GAN_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    GAN_model.summary()
    discrimnator.trainable = True

    if g_path:
        generator.load_weights(g_path)
    
    if d_path:
        discrimnator.load_weights(d_path)

    return GAN_model, discrimnator, generator

def train(gan_model, g_model, d_model, epoch, songs, seq_len, g_name="G", d_name="D"):

    random.shuffle(songs)

    for idx, real_batches in enumerate(songs):
        
        d_loss_real = 0
        d_loss_fake = 0
        g_loss = 0

        # Train real data
        for real_batch in real_batches:
            
            real_x = np.concatenate(([real_batch[0]], [real_batch[1]]), axis=-1)
            real_y = np.ones((1, 1))
            # real_y = np.array([real_batch[2]])

            # add noise to real data
            if random.uniform(0, 1) < 1.0:
                real_x = noise_to_data(real_x)

            d_loss_real += d_model.train_on_batch(real_x, real_y)
           
        d_model.reset_states()
        d_loss_real /= len(real_batches)

        z_batches = noise(seq_len)

        # Train fake data
        fake_batches = g_model.predict(z_batches)
 
        for fake_batch in fake_batches:
            fake_x = np.array([fake_batch])
            fake_y = np.zeros((1, 1))

            d_loss_fake += d_model.train_on_batch(fake_x, fake_y)

        d_model.reset_states()
        d_loss_fake /= len(z_batches)

        # Train generator 
        z_batches = noise(seq_len)
        for z in z_batches:
            z = np.array([z])
            real = np.ones((1, 1))
            g_loss += gan_model.train_on_batch(z, real)
            
        gan_model.reset_states()
        g_loss /= len(z_batches)

        print ("%d-%d [D Real loss: %f][D Fake loss: %f][G loss: %f]" % (epoch, idx, d_loss_real, d_loss_fake, g_loss))

    g_model.save("weights/"+ str(epoch) + "-"+ g_name +".hdf5")
    d_model.save("weights/"+ str(epoch) + "-"+ d_name +".hdf5")
    g_model.save("weights/"+ g_name +".hdf5")
    d_model.save("weights/"+ d_name +".hdf5")

def noise(sequence_length):
    
    batch_size = np.random.randint(8, 16)
    laten_z = np.random.normal(0, 1, (batch_size, LatenSize))
    noise = laten_z


    return noise

def noise_to_data(data_x):
    
    noise = np.random.normal(0, 0.005, data_x.shape)
    data_x = np.add(data_x, noise)
    return data_x

def generate(g_model, seq_len):

    z_batches = noise(seq_len)

    song = g_model.predict(z_batches)
    song = np.concatenate(song)
    
    melody = song[:,0]
    accomp = song[:,1]

    return melody, accomp
