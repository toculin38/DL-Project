from keras.models import Sequential, Model
from keras.layers import Activation, Concatenate, Dropout, LSTM, Dense, Input
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import numpy as np
def create(network_input, output_size, weights_path=None):
    """ create the structure of the neural network """
    input_shape = (network_input.shape[1], network_input.shape[2])
    first_input = Input(shape=input_shape)

    front = Sequential()
    front.add(LSTM(512, return_sequences=True))
    front.add(Dropout(0.3))
    front.add(LSTM(512, return_sequences=True))
    front.add(Dropout(0.3))
    front.add(LSTM(512))
    front.add(Dense(256))
    front.add(Dropout(0.3))

    pitch_out = front(first_input)
    pitch_out = Dense(output_size)(pitch_out)
    pitch_out = Activation('softmax', name="keys_output")(pitch_out)

    # duration_out = front(first_input)
    # duration_out = Dense(duration_length)(duration_out)
    # duration_out = Activation('softmax', name="duration_output")(duration_out)

    losses = {
        "keys_output": "categorical_crossentropy",
        # "duration_output" : "categorical_crossentropy"
    }

    model = Model(inputs=first_input, outputs=pitch_out)
    model.compile(loss=losses, optimizer='rmsprop')

    if weights_path:
        model.load_weights(weights_path)

    return model

def train(model, network_input, network_output):
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

    model.fit(network_input, network_output, epochs=1000, batch_size=512, callbacks=callbacks_list)

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
