from keras.models import Sequential, Model
from keras.layers import Activation, Concatenate, Dropout, LSTM, Dense, Input, TimeDistributed, RepeatVector, Bidirectional, Lambda
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
from focal_losses import *

def create_melody_model(sequence_length, key_size, offset_size, weights_path=None):
    key_input = Input(shape=(sequence_length, key_size))
    offset_input = Input(shape=(sequence_length, offset_size))

    key_layer_input = Concatenate(axis=-1)([key_input, offset_input])
    key_layer = Sequential()
    key_layer.add(LSTM(512, return_sequences=True))
    key_layer.add(Dropout(0.2))
    key_layer.add(LSTM(256, return_sequences=True))
    key_layer.add(TimeDistributed(Dense(128)))
    key_layer.add(Dropout(0.2))
    key_info = key_layer(key_layer_input)
    key_out = Dense(key_size)(key_info)
    key_out = Activation('sigmoid', name="key")(key_out)

    losses = {
        "key": binary_focal_loss(),
    }

    model = Model(inputs=[key_input, offset_input], outputs=[key_out])
    model.compile(loss=losses, optimizer='rmsprop')

    if weights_path:
        model.load_weights(weights_path)

    return model

def create_accomp_model(sequence_length, key_size, offset_size, weights_path=None):
    key_input = Input(shape=(sequence_length, key_size))
    offset_input = Input(shape=(sequence_length, offset_size))
    accomp_input = Input(shape=(sequence_length, key_size))

    key2_layer_input = Concatenate(axis=-1)([key_input, offset_input, accomp_input])
    key2_layer = Sequential()
    key2_layer.add(LSTM(512, return_sequences=True))
    key2_layer.add(Dropout(0.2))
    key2_layer.add(LSTM(256, return_sequences=True))
    key2_layer.add(TimeDistributed(Dense(128)))
    key2_layer.add(Dropout(0.2))
    key2_info = key2_layer(key2_layer_input)
    key2_out = Dense(key_size)(key2_info)
    key2_out = Activation('sigmoid', name="key2")(key2_out)

    losses = {
        "key2": binary_focal_loss(),
    }

    model = Model(inputs=[key_input, offset_input, accomp_input], outputs=[key2_out])
    model.compile(loss=losses, optimizer='rmsprop')

    if weights_path:
        model.load_weights(weights_path)

    return model

def train(model, epoch, data, target, model_name, batch_size=1024):
    filepath = "weights/"+ model_name + "-"+ str(epoch) +"-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(x=data, y=target, epochs=1, batch_size=batch_size, callbacks=callbacks_list)