from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from process import TextProcessor

import numpy as np
import random
import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='datasets/leopardi_short.txt',
                        help='data directory containing input dataset')

    parser.add_argument('--save_dir', type=str, default='saves',
                        help='directory to store checkpointed models')

    # parser.add_argument('--load_weights', type=bool, default=False,
    #                     help='True to train starting from latest saved weights, default is False')

    # parser.add_argument('--input_encoding', type=str, default=None,
    #                     help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    # parser.add_argument('--log_dir', type=str, default='logs',
    #                     help='directory containing tensorboard logs')
    # parser.add_argument('--save_dir', type=str, default='save',
    #                     help='directory to store checkpointed models')
    parser.add_argument('--rnn_units', type=int, default=512,
                        help='size of RNN hidden state')
    # parser.add_argument('--num_layers', type=int, default=2,
    #                     help='number of layers in the RNN')
    # parser.add_argument('--model', type=str, default='lstm',
    #                     help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    # parser.add_argument('--seq_length', type=int, default=25,
    #                     help='RNN sequence length')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    # parser.add_argument('--save_every', type=int, default=1000,
    #                     help='save frequency')
    # parser.add_argument('--grad_clip', type=float, default=5.,
    #                     help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # parser.add_argument('--decay_rate', type=float, default=0.97,
    #                    help='decay rate for rmsprop')
    # parser.add_argument('--gpu_mem', type=float, default=0.666,
    #                     help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    # parser.add_argument('--init_from', type=str, default=None,
    #                     help="""continue training from saved model at this path. Path must contain files saved by previous training process:
    #                             'config.pkl'        : configuration;
    #                             'words_vocab.pkl'   : vocabulary definitions;
    #                             'checkpoint'        : paths to model file(s) (created by tf).
    #                                                   Note: this file contains absolute paths, be careful when moving files around;
    #                             'model.ckpt-*'      : file(s) with model definition (created by tf)
    #                         """)
    args = parser.parse_args()
    train(args)


def train(args):
    input_file = args.input_file
    saves_folder = args.save_dir

    # text prep
    # try:
    #     text = open(input_file).read().lower()
    # except UnicodeDecodeError:
    #     import codecs
    #     text = codecs.open(input_file, encoding='utf-8').read().lower()

    text_processor = TextProcessor(input_file)

    # words = set(open(input_file).read().lower().split())

    text_processor.preprocess()
    text_processor.print_info()

    text_processor.prepare_sentences()
    text_processor.vectorize()

    # word_indices = dict((c, i) for i, c in enumerate(words))
    # indices_word = dict((i, c) for i, c in enumerate(words))

    # maxlen = 30
    # step = 3
    # print("maxlen:", maxlen, "step:", step)
    # sentences = []
    # next_words = []
    #
    # list_words = text_processor.processed.split()
    #
    # for i in range(0, len(list_words) - maxlen, step):
    #     sentences2 = ' '.join(list_words[i: i + maxlen])
    #     sentences.append(sentences2)
    #     next_words.append((list_words[i + maxlen]))
    # print('nb sequences(length of sentences):', len(sentences))
    # print("length of next_word", len(next_words))
    #
    # # vectorize words
    # print('Vectorization...')
    # X = np.zeros((len(sentences), maxlen, len(text_processor.words_set)), dtype=np.bool)
    # y = np.zeros((len(sentences), len(text_processor.words_set)), dtype=np.bool)
    # for i, sentence in enumerate(sentences):
    #     for t, word in enumerate(sentence.split()):
    #         # print(i, t, word)
    #         X[i, t, text_processor.word_indices[word]] = 1
    #     y[i, text_processor.word_indices[next_words[i]]] = 1

    # build the model: 2 stacked LSTM
    maxlen = text_processor.maxlen
    X = text_processor.X
    y = text_processor.y
    units = args.rnn_units

    print('Build model...')
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(maxlen, len(text_processor.words_set))))
    model.add(Dropout(0.2))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(text_processor.words_set)))
    # model.add(Dense(1000))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=args.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # train the model
    checkpoint_filepath = saves_folder + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model_file = saves_folder + '/model.hdf5'

    history = model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks_list)
    model.save(model_file, overwrite=True)

    print("Training completed.")


if __name__ == '__main__':
    main()
