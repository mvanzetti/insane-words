from process import TextProcessor
from model import Model
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='datasets/leopardi_short.txt',
                        help='data directory containing input dataset')

    parser.add_argument('--save_dir', type=str, default='saves',
                        help='directory to store checkpointed models')

    parser.add_argument('--vocab_name', type=str, default='vocab',
                        help='name of the vocabulary generated while processing the input dataset')

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
    parser.add_argument("optimizer",
                        help='optimizer to use for the training',
                        choices=['rmsprop', 'adam'], default='rmsprop')
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
    vocab_name = args.vocab_name

    text_processor = TextProcessor(input_file)
    text_processor.build_vocabulary(saves_folder, vocab_name)
    text_processor.print_vocabulary_info()
    text_processor.vectorize()

    X = text_processor.X
    y = text_processor.y
    units = args.rnn_units
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    seq_len = X.shape[1]
    dict_num = X.shape[2]

    print("X.shape", X.shape)
    print("y.shape", y.shape)

    model = Model(units, seq_len, dict_num, optimizer, learning_rate)
    model.print_info()
    model.set_saves_folder(saves_folder)
    model.compile()
    model.remove_checkpoints()
    model.train(X, y, batch_size, epochs)
    model.print_summary()

    # # train the model
    # checkpoint_filepath = saves_folder + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # time_callback = TimeHistory()
    # callbacks_list = [checkpoint, time_callback]
    #
    # time_str = time.strftime("%Y%m%d-%H%M%S")
    # model_file = saves_folder + '/model-' + time_str + '.hdf5'
    #
    # history = model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks_list)
    #
    # model.save(model_file, overwrite=True)

    print("Training completed.")
    # print(time_callback.times)


if __name__ == '__main__':
    main()
