import numpy as np
import argparse
import random
from keras.models import load_model
from process import TextProcessor
from utils.custom_recurrents import AttentionDecoder
from model import Model
from model_with_attention import AttentionModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default='datasets/leopardi_short.txt',
                        help='data directory containing input dataset')

    parser.add_argument('--vocab_name', type=str, default='vocab',
                        help='data directory containing input dataset')

    parser.add_argument('--load_dir', type=str, default='saves',
                        help='directory where to load checkpointed models and vocabulary')

    parser.add_argument('--maxlen_gen', type=int, default=100,
                        help='maximum length (in words) of the generated sample')

    parser.add_argument('--temperature', type=float, default=1.0,
                        help='tune the softmax temperature during sampling '
                             '(e.g.: < 1.0 more confident and more conservative, '
                             '> 1.0 more diversity at cost of spelling mistakes)')

    parser.add_argument('--model_filename', type=str, default='model.hdf5',
                        help='file that stores the trained model')

    parser.add_argument('--seed', type=str, default=None,
                        help='user provided seed')

    args = parser.parse_args()
    sample(args)


def sample_softmax(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# generate a sentence picked randomly in the text
def generate_seed_sentence(list_words, maxlen_seed):
    start_index = random.randint(0, len(list_words) - maxlen_seed - 1)
    sentence = list_words[start_index: start_index + maxlen_seed]
    # log.debug('Generating with seed: "%s"' , sentence)
    print(start_index)
    return sentence


# words: words in dict retrieved from training text
# sentence: seed sentence as a list of words
# temperature: parameter to tune for diversity of generated text
# maxlen_seed: max length of window to sample next words (seed sentences)
# maxlen_gen: max words to generate
def sample_words(model, word_indices, indices_word, words, sentence, temperature, maxlen_seed, maxlen_gen):
    generated = []
    for i in range(maxlen_gen):
        x = np.zeros((1, maxlen_seed, len(words)))
        for t, word in enumerate(sentence):
            x[0, t, word_indices[word]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample_softmax(preds, temperature)
        next_word = indices_word[next_index]

        del sentence[0]
        sentence.append(next_word)
        generated.append(next_word)

    return generated


def sample(args):
    input_file = args.input_file
    saves_folder = args.load_dir
    vocab_name = args.vocab_name
    maxlen_gen = args.maxlen_gen
    temperature = args.temperature
    model_filename = args.model_filename
    seed = args.seed

    model_file = saves_folder + '/' + model_filename

    model = load_model(model_file, custom_objects={'AttentionDecoder': AttentionDecoder})

    text_processor = TextProcessor(input_file)
    text_processor.load_vocabulary(saves_folder, vocab_name)
    # text_processor.preprocess()
    # text_processor.build_vocabulary()
    text_processor.print_vocabulary_info()
    # text_processor.vectorize()

    if seed is None:
        sentence = generate_seed_sentence(text_processor.processed.split(), 30)
    else:
        sentence = seed.split()

    print("seed:")
    print(' '.join(sentence))
    result = sample_words(model, text_processor.word_indices, text_processor.indices_word, text_processor.words_set,
                          sentence, temperature, 30, maxlen_gen)

    print()
    print("sampled:")
    print(' '.join(result))


if __name__ == '__main__':
    main()
