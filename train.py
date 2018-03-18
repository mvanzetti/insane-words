import argparse

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/i_malavoglia_short',
                        help='data directory containing input.txt')
    parser.add_argument('--input_encoding', type=str, default=None,
                        help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                        help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                                'config.pkl'        : configuration;
                                'words_vocab.pkl'   : vocabulary definitions;
                                'checkpoint'        : paths to model file(s) (created by tf).
                                                      Note: this file contains absolute paths, be careful when moving files around;
                                'model.ckpt-*'      : file(s) with model definition (created by tf)
                            """)
    args = parser.parse_args()
    train(args)

def train(args):
    pass

if __name__ == '__main__':
    main()