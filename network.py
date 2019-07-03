from keras.models import Sequential, Model
from keras.layers import Activation, Concatenate, Dropout, LSTM, Dense, Input
from keras.callbacks import ModelCheckpoint

def create(network_input, pitch_length, duration_length, weights_path=None):
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
    pitch_out = Dense(pitch_length)(pitch_out)
    pitch_out = Activation('softmax', name="pitch_output")(pitch_out)

    duration_out = front(first_input)
    duration_out = Dense(duration_length)(duration_out)
    duration_out = Activation('softmax', name="duration_output")(duration_out)

    losses = {
        "pitch_output": "categorical_crossentropy",
        "duration_output" :  "categorical_crossentropy"
    }

    model = Model(inputs=first_input, outputs=[pitch_out, duration_out])
    model.compile(loss=losses, optimizer='rmsprop')

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