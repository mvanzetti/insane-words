from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import keras
import time
import os
import re


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class Model:

    def __init__(self, units, seq_len, dict_len, learning_rate, dropout_rate=0.2):
        self.layer_units = units
        self.dropout_rate = dropout_rate

        self.model = Sequential()
        self.model.add(
            LSTM(self.layer_units, return_sequences=True, input_shape=(seq_len, dict_len)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(self.layer_units, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(dict_len))
        self.model.add(Activation('softmax'))

        self.optimizer = RMSprop(lr=learning_rate)
        self.loss = 'categorical_crossentropy'

        self.callbacks_list = []
        self.saves_folder = None
        self.history = None

    def set_saves_folder(self, saves_folder):
        self.saves_folder = saves_folder

    def compile(self):
        checkpoint_filepath = self.saves_folder + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        time_callback = TimeHistory()
        self.callbacks_list = [checkpoint, time_callback]

        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def train(self, x_tensor, y_tensor, batch_size, epochs):
        self.history = self.model.fit(x_tensor, y_tensor, batch_size=batch_size, epochs=epochs,
                                      callbacks=self.callbacks_list)

        time_str = time.strftime("%Y%m%d-%H%M%S")
        model_file = self.saves_folder + '/model-' + time_str + '.hdf5'
        self.model.save(model_file, overwrite=True)

    def print_summary(self):
        print(self.model.summary())

    def remove_checkpoints(self):
        pattern = r"weights-improvement"
        path = os.path.dirname(os.path.realpath(__file__)) + "/" + self.saves_folder

        for f in os.listdir(path):
            if re.search(pattern, f):
                os.remove(os.path.join(path, f))
