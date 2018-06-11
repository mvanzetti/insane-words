from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
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
    """
    Stacked LSTM Model
    """

    def __init__(self, units, seq_len, dict_len, optimizer='rmsprop', learning_rate=0.001, dropout_rate=0.2):
        self.layer_units = units
        self.input_shape = (seq_len, dict_len)
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = self.define_model()
        #
        # plot_model(self.model, to_file='model.png')

        if optimizer == 'rmsprop':
            self.optimizer = RMSprop(lr=self.learning_rate)
        elif optimizer == 'adam':
            self.optimizer = Adam(lr=self.learning_rate)

        self.loss = 'categorical_crossentropy'

        self.callbacks_list = []
        self.saves_folder = None
        self.weights_folder = None
        self.history = None

    def define_model(self):
        dict_len = self.input_shape[1]

        model = Sequential()
        model.add(
            LSTM(self.layer_units, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(self.layer_units, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(dict_len))
        model.add(Activation('softmax'))

        return model

    def set_saves_folder(self, saves_folder):
        self.saves_folder = saves_folder
        self.weights_folder = os.path.join(saves_folder, "weights")

        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)

    def compile(self):
        checkpoint_filepath = self.weights_folder + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        time_callback = TimeHistory()
        self.callbacks_list = [checkpoint, time_callback]

        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def train(self, x_tensor, y_tensor, batch_size, epochs):
        print()
        print("Training Parameters")
        print()
        print("Batch Size       ", batch_size)
        print("Epochs           ", epochs)
        print()

        self.history = self.model.fit(x_tensor, y_tensor, batch_size=batch_size, epochs=epochs,
                                      callbacks=self.callbacks_list)

        time_str = time.strftime("%Y%m%d-%H%M%S")
        model_file = self.saves_folder + '/model-' + time_str + '.hdf5'
        self.model.save(model_file, overwrite=True)

        history_file = self.saves_folder + '/history-' + time_str + '.txt'
        np.savetxt(history_file, np.array(self.history.history['loss']), delimiter=';')

        # plt.plot(self.history.history['loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train'], loc='upper left')
        # plt.show()

    def print_info(self):
        print()
        print("Stacked LSTM")
        print()
        print("Layers           ", 2)
        print("Layer Units      ", self.layer_units)
        print("Input Shape      ", self.input_shape)
        print("Dropout Rate     ", self.dropout_rate)
        print("Optimizer        ", self.optimizer)
        print("Learning Rate    ", self.learning_rate)
        print("Loss             ", self.loss)
        print()

    def plot_model(self):
        plot_folder = os.path.join(self.saves_folder, "plots")

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plot_model(self.model, to_file=plot_folder + '/model.png')

    def print_summary(self):
        print(self.model.summary())

    def remove_checkpoints(self):
        pattern = r"weights-improvement"
        path = os.path.dirname(os.path.realpath(__file__)) + "/" + self.weights_folder

        for f in os.listdir(path):
            if re.search(pattern, f):
                os.remove(os.path.join(path, f))
