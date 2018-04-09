import re
import codecs
import numpy as np

tag_to_char = {
    '<EOL>': '\n',
    '<COMMA>': ',',
    '<DOT>': '.',
    '<APOSTR>': "'",
    '<EXCLAM>': '!',
    '<QUESTION>': '?',
    '<SEMICOL>': ';'
}

char_to_tag = {
    '\n': '<EOL>',
    ',': '<COMMA>',
    '.': '<DOT>',
    "'": '<APOSTR>',
    '!': '<EXCLAM>',
    '?': '<QUESTION>',
    ';': '<SEMICOL>'
}


class TextProcessor:
    """
    Processes a text from input file and provides data in the proper form for the modelization (X, y tensors)

    Attributes:
        data                raw data from input file
        processed           data after pre-processing (tagging of special character)
        words_set           set of words in the processed file
        word_indices        dictionary mapping words to int indexes
        indices_word        reverse dictionary of word_indices

        num_sentences       number of sampled sentences (examples)
        words_in_sentence   number of words in each sampled sentence example
        dict_len            number of distinct words in dictionaries or len(words_set)

        X                   prepared data, tensor of shape (num_sentences, words_in_sentence, dict_len)
                            in one-hot representation
        y                   prepared predictions of next words, tensor of shape (num_sentences, dict_len)
                            in one-hot representation


    """

    def __init__(self, input_file, encoding=None):
        self.data = None
        self.processed = None
        self.words_set = None
        self.word_indices = None
        self.indices_word = None

        self.num_sentences = None
        self.words_in_sentence = None
        self.dict_len = None

        self.X = None
        self.y = None

        with codecs.open(input_file, "r", encoding) as f:
            self.data = f.read()

        self.processed = self.data

    def preprocess(self):
        self.processed = self.replace_nonwords(self.data)

    def build_dicts(self):
        self.words_set = set(self.processed.split())
        self.word_indices = dict((c, i) for i, c in enumerate(self.words_set))
        self.indices_word = dict((i, c) for i, c in enumerate(self.words_set))

        self.dict_len = len(self.words_set)

    def vectorize(self, sampling_maxlen=30, sampling_step=3):
        """
        Provides vectorized form of sentences sampled from processed text data in one-hot representation,
        both for sampled sentences and next word predictions.

        :param sampling_maxlen: maximum length, in words, of each sampled sentence (sampling window length)
        :param sampling_step: step in words to jump while sampling sentences (sampling window step)
        """

        def sample_sentences(text, sample_len, sample_step):

            print("Sampling sentences with len (words):", sample_len, "with sampling step window:",
                  sample_step)
            sampled_sentences = []
            sampled_next_words = []

            list_words = text.split()

            for pos in range(0, len(list_words) - sample_len, sample_step):
                sentences2 = ' '.join(list_words[pos: pos + sample_len])
                sampled_sentences.append(sentences2)
                sampled_next_words.append((list_words[pos + sample_len]))
            print('nb sequences(length of sentences):', len(sampled_sentences))
            print("length of next_word", len(sampled_next_words))

            return sampled_sentences, sampled_next_words

        sentences, next_words = sample_sentences(sampling_maxlen, sampling_step)

        print('Vectorizing...')
        self.num_sentences = len(sentences)
        self.words_in_sentence = sampling_maxlen

        self.X = np.zeros((self.num_sentences, self.words_in_sentence, self.dict_len), dtype=np.bool)
        self.y = np.zeros((self.num_sentences, self.dict_len), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence.split()):
                # print(i, t, word)
                self.X[i, t, self.word_indices[word]] = 1
            self.y[i, self.word_indices[next_words[i]]] = 1

    @staticmethod
    def map_char_to_tag(data, char):
        return re.sub("[" + char + "]", " " + char_to_tag[char] + " ", data)

    @staticmethod
    def map_tag_to_char(data, tag):
        return re.sub(tag, char_to_tag[tag], data)

    def replace_nonwords(self, data):
        processed = data
        for char in char_to_tag.keys():
            processed = self.map_char_to_tag(processed, char)

        return processed

    def print_info(self):
        chars = set(self.data)
        print('corpus length:', len(self.data))
        print("chars:", type(chars))
        print("words", type(self.words_set))
        print("total number of unique words", len(self.words_set))
        print("total number of unique chars", len(chars))

        print("word_indices", type(self.word_indices), "length:", len(self.word_indices))
        print("indices_words", type(self.indices_word), "length", len(self.indices_word))
