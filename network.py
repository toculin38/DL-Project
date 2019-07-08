from keras.models import Sequential, Model
from keras.layers import Activation, Concatenate, Dropout, LSTM, Dense, Input, TimeDistributed, RepeatVector
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import numpy as np

def create_2(sequence_length, key_size, press_size, offset_size, weights_path=None):
    key_input = Input(shape=(sequence_length, key_size))
    press_input = Input(shape=(sequence_length, press_size))
    offset_input = Input(shape=(sequence_length, offset_size))

    encoder_input = Concatenate(axis=-1)([key_input, press_input, offset_input])
    encoder = Sequential()
    encoder.add(LSTM(512, return_sequences=True))
    encoder.add(Dropout(0.3))
    encoder.add(LSTM(256))
    encoder.add(Dropout(0.3))
    encoder.add(Dense(128))
    encoder.add(Activation("sigmoid"))

    latent_z = encoder(encoder_input)

    decoder_input = Input(shape=(128,))
    decoder = Sequential()
    decoder.add(RepeatVector(sequence_length))
    decoder.add(LSTM(256, return_sequences=True))
    decoder.add(Dropout(0.3))
    decoder.add(LSTM(512, return_sequences=True))
    decoder.add(Dropout(0.3))

    KeyTD = TimeDistributed(Dense(key_size))
    keySoftmax = Activation("softmax", name="key")
    key_out = keySoftmax(KeyTD(decoder(latent_z)))

    PressTD = TimeDistributed(Dense(press_size))
    PressSoftmax = Activation("softmax", name="press")
    press_out = PressSoftmax(PressTD(decoder(latent_z)))

    losses = {
        "key": "categorical_crossentropy",
        "press" : "categorical_crossentropy"
    }

    train_model = Model(inputs=[key_input, press_input, offset_input], outputs=[key_out, press_out])
    train_model.compile(loss=losses, optimizer='rmsprop')

    key_out = keySoftmax(KeyTD(decoder(decoder_input)))
    press_out = PressSoftmax(PressTD(decoder(decoder_input)))

    encoder_model = Model(inputs=[key_input, press_input, offset_input], outputs=[latent_z])
    decoder_model = Model(inputs=[decoder_input], outputs=[key_out, press_out])

    if weights_path:
        train_model.load_weights(weights_path)

    return train_model, encoder_model, decoder_model


def create(sequence_length, key_size, press_size, offset_size, weights_path=None):
    key_input = Input(shape=(sequence_length, key_size))
    press_input = Input(shape=(sequence_length, press_size))
    offset_input = Input(shape=(sequence_length, offset_size))

    key_layer = Sequential()
    key_layer.add(LSTM(512, return_sequences=True))
    key_layer.add(Dropout(0.3))
    key_layer.add(LSTM(512, return_sequences=True))
    key_layer.add(TimeDistributed(Dense(256)))
    key_layer.add(Dropout(0.3))
    
    key_layer_input = Concatenate(axis=-1)([key_input, press_input, offset_input])
    key_info = key_layer(key_layer_input)
    key_out = Dense(key_size)(key_info)
    key_out = Activation('softmax', name="key")(key_out)

    press_out = Dense(press_size)(key_info)
    press_out = Activation('softmax', name="press")(press_out)

    losses = {
        "key": "categorical_crossentropy",
        "press" : "categorical_crossentropy"
    }

    model = Model(inputs=[key_input, press_input, offset_input], outputs=[key_out, press_out])
    model.compile(loss=losses, optimizer='rmsprop')

    if weights_path:
        model.load_weights(weights_path)

    return model

def train(model, key_data, press_data, offset_data, key_target, press_target):
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

    model.fit(x=[key_data, press_data, offset_data], y=[key_target, press_target], epochs=1000, batch_size=512, callbacks=callbacks_list)

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
