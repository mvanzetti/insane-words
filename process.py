import re
import os
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

    def __init__(self, input_file, encoding=None):
        self.data = None
        self.processed = None
        self.vocab = None
        self.words_set = None
        self.word_indices = None
        self.indices_word = None

        # todo elevate to parametrization
        self.maxlen = 30
        self.step = 3

        self.list_words = None
        self.sentences = None
        self.next_words = None

        self.X = None
        self.Y = None

        with codecs.open(input_file, "r", encoding) as f:
            self.data = f.read()

    def preprocess(self):
        self.processed = self.replace_nonwords(self.data)

        self.words_set = set(self.processed.split())
        self.word_indices = dict((c, i) for i, c in enumerate(self.words_set))
        self.indices_word = dict((i, c) for i, c in enumerate(self.words_set))

    def prepare_sentences(self):

        print("maxlen:", self.maxlen, "step:", self.step)
        self.sentences = []
        self.next_words = []

        self.list_words = self.processed.split()

        for i in range(0, len(self.list_words) - self.maxlen, self.step):
            sentences2 = ' '.join(self.list_words[i: i + self.maxlen])
            self.sentences.append(sentences2)
            self.next_words.append((self.list_words[i + self.maxlen]))
        print('nb sequences(length of sentences):', len(self.sentences))
        print("length of next_word", len(self.next_words))

    def vectorize(self):

        # vectorize words
        print('Vectorization...')
        self.X = np.zeros((len(self.sentences), self.maxlen, len(self.words_set)), dtype=np.bool)
        self.y = np.zeros((len(self.sentences), len(self.words_set)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, word in enumerate(sentence.split()):
                # print(i, t, word)
                self.X[i, t, self.word_indices[word]] = 1
            self.y[i, self.word_indices[self.next_words[i]]] = 1

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

    # self.processed = re.sub('\n', " " + char_to_tag["\n"] + " ", data)
    # self.processed = re.sub("[,]", " " + char_to_tag[","] + " ", self.processed)
    # self.processed = re.sub("[.]", " " + char_to_tag["."] + " ", self.processed)
    # self.processed = re.sub("[']", " " + char_to_tag["'"] + " ", self.processed)
    # self.processed = re.sub("[!]", " " + char_to_tag["!"] + " ", self.processed)
    # self.processed = re.sub("[?]", " " + char_to_tag["?"] + " ", self.processed)
    # self.processed = re.sub("[;]", " " + char_to_tag[";"] + " ", self.processed)
