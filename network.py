from keras.models import Sequential, Model
from keras.layers import Activation, Concatenate, Dropout, LSTM, Dense, Input, TimeDistributed, RepeatVector, Bidirectional, Lambda
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import numpy as np
from focal_losses import *


def create_melody_model(sequence_length, key_size, press_size, offset_size, weights_path=None):
    key_input = Input(shape=(sequence_length, key_size))
    press_input = Input(shape=(sequence_length, press_size))
    offset_input = Input(shape=(sequence_length, offset_size))

    key_layer_input = Concatenate(axis=-1)([key_input, press_input, offset_input])
    key_layer = Sequential()
    key_layer.add(LSTM(512, return_sequences=True))
    key_layer.add(Dropout(0.2))
    key_layer.add(LSTM(256, return_sequences=True))
    key_layer.add(TimeDistributed(Dense(128)))
    key_layer.add(Dropout(0.2))
    key_info = key_layer(key_layer_input)
    key_out = Dense(key_size)(key_info)
    key_out = Activation('softmax', name="key")(key_out)

    press_layer = Sequential()
    press_layer.add(LSTM(256, return_sequences=True))
    press_layer.add(Dropout(0.2))
    press_layer.add(LSTM(128, return_sequences=True))
    press_layer.add(TimeDistributed(Dense(64)))
    press_layer.add(Dropout(0.2))
    press_info = press_layer(key_info)
    press_out = Dense(press_size)(press_info)
    press_out = Activation('softmax', name="press")(press_out)

    losses = {
        "key": categorical_focal_loss(),
        "press" : categorical_focal_loss()
    }

    model = Model(inputs=[key_input, press_input, offset_input], outputs=[key_out, press_out])
    model.compile(loss=losses, optimizer='rmsprop')

    if weights_path:
        model.load_weights(weights_path)

    return model

def create_accomp_model(sequence_length, key_size, press_size, offset_size, weights_path=None):
    key_input = Input(shape=(sequence_length, key_size))
    press_input = Input(shape=(sequence_length, press_size))
    offset_input = Input(shape=(sequence_length, offset_size))
    accomp_input = Input(shape=(sequence_length, key_size))

    key2_layer_input = Concatenate(axis=-1)([key_input, press_input, offset_input, accomp_input])
    key2_layer = Sequential()
    key2_layer.add(LSTM(512, return_sequences=True))
    key2_layer.add(Dropout(0.2))
    key2_layer.add(LSTM(256, return_sequences=True))
    key2_layer.add(TimeDistributed(Dense(128)))
    key2_layer.add(Dropout(0.2))
    key2_info = key2_layer(key2_layer_input)
    key2_out = Dense(key_size)(key2_info)
    key2_out = Activation('softmax', name="key2")(key2_out)

    press2_layer = Sequential()
    press2_layer.add(LSTM(256, return_sequences=True))
    press2_layer.add(Dropout(0.2))
    press2_layer.add(LSTM(128, return_sequences=True))
    press2_layer.add(TimeDistributed(Dense(64)))
    press2_layer.add(Dropout(0.2))
    press2_info = press2_layer(key2_info)
    press2_out = Dense(press_size)(press2_info)
    press2_out = Activation('softmax', name="press2")(press2_out)

    losses = {
        "key2": categorical_focal_loss(),
        "press2" : categorical_focal_loss()
    }

    model = Model(inputs=[key_input, press_input, offset_input, accomp_input], outputs=[key2_out, press2_out])
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

def keys_binary_crossentropy(y_true, y_pred):
    press_true_mask = K.greater_equal(y_true, 0.5)
    press_pred_mask = K.greater_equal(y_pred, 0.5)
    press_mask = K.cast(K.any(K.stack([press_true_mask, press_pred_mask], axis=0), axis=0), K.floatx())

    press_true = tf.multiply(y_true, press_mask)
    press_pred = tf.multiply(y_pred, press_mask)

    # empty_true_mask = K.less(y_true, threshhold)
    # empty_pred_mask = K.less(y_pred, threshhold)
    # empty_mask = K.cast(K.any(K.stack([empty_true_mask, empty_pred_mask], axis=-1), axis=-1), K.floatx())

    # empty_true = tf.multiply(y_true, empty_mask)
    # empty_pred = tf.multiply(y_pred, empty_mask)

    # press_loss = K.mean(K.binary_crossentropy(press_true, press_pred), axis=-1)
    # empty_loss = K.mean(K.binary_crossentropy(empty_true, empty_pred), axis=-1)

    return K.sum(K.binary_crossentropy(press_true, press_pred), axis=-1) / K.sum(press_mask, -1)
