from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

from keras.callbacks import ModelCheckpoint


def create(network_input, n_vocab, weights_path=None):
    """ create the structure of the neural network """
    input_shape = (network_input.shape[1], network_input.shape[2])

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=input_shape,
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if weights_path:
        model.load_weights(weights_path)

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=1000, batch_size=512, callbacks=callbacks_list)