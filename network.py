from keras.models import Sequential, Model
from keras.layers import Activation, Concatenate, Dropout, LSTM, Dense, Input
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import numpy as np
def create(keys_data, keys_output_size, press_data, press_output_size, weights_path=None):
    """ create the structure of the neural network """
    key_input = Input(shape=(keys_data.shape[1], keys_data.shape[2]))
    press_input = Input(shape=(press_data.shape[1], press_data.shape[2]))

    key_layer = Sequential()
    key_layer.add(LSTM(512, return_sequences=True))
    key_layer.add(Dropout(0.3))
    key_layer.add(LSTM(512))
    key_layer.add(Dense(256))
    key_layer.add(Dropout(0.3))
    
    key_out = key_layer(key_input)
    key_out = Dense(keys_output_size)(key_out)
    key_out = Activation('softmax', name="keys_output")(key_out)

    press_layer = Sequential()
    press_layer.add(LSTM(512, return_sequences=True))
    press_layer.add(Dropout(0.3))
    press_layer.add(LSTM(512))
    press_layer.add(Dense(256))
    press_layer.add(Dropout(0.3))

    merged = Concatenate(axis=-1)([key_input, press_input])
    press_out = press_layer(merged)
    press_out = Dense(press_output_size)(press_out)
    press_out = Activation('softmax', name="press_output")(press_out)

    losses = {
        "keys_output": "categorical_crossentropy",
        "press_output" : "categorical_crossentropy"
    }

    model = Model(inputs=[key_input, press_input], outputs=[key_out, press_out])
    model.compile(loss=losses, optimizer='rmsprop')

    if weights_path:
        model.load_weights(weights_path)

    return model

def train(model, key_data, press_data, key_target, press_target):
    """ train the neural network """
    filepath = "weights/weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(x=[key_data, press_data], y=[key_target, press_target], epochs=1000, batch_size=512, callbacks=callbacks_list)

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
