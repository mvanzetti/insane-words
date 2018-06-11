from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
from utils.custom_recurrents import AttentionDecoder
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import keras
import time
import os
import re


class AttentionModel(Model):
    """
    Stacked LSTM Model
    """

    def __init__(self, units, seq_len, dict_len, optimizer='rmsprop', learning_rate=0.001, dropout_rate=0.2):
        super(AttentionModel, self).__init__(units, seq_len, dict_len, optimizer, learning_rate, dropout_rate)

    def define_model(self):
        dict_len = self.input_shape[1]

        model = Sequential()
        model.add(
            LSTM(self.layer_units, return_sequences=True, input_shape=self.input_shape))
        model.add(AttentionDecoder(self.layer_units, output_dim=dict_len, name="AttentionLayer"))
        model.add(LSTM(self.layer_units, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(dict_len))
        model.add(Activation('softmax'))
        return model

    def print_info(self):
        print()
        print("LSTM with Attention")
        print()
        print("Layers           ", 3)
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
        plot_model(self.model, to_file=plot_folder + '/model_with_attention.png')
