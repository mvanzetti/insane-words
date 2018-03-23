import re
import os
import codecs


class TextProcessor:

    def __init__(self, input_file, encoding=None):
        with codecs.open(input_file, "r", encoding=encoding) as f:
            self.data = f.read()

        self.processed = None
        self.mappings = {
            'forward': {
                '<EOL>': '\n',
                '<COMMA>': ',',
                '<DOT>': '.',
                '<APOSTR>': "'",
                '<EXCLAM>': '!',
                '<QUESTION>': '?',
                '<SEMICOL>': ';'
            }, 'reverse': {
                '\n': '<EOL>',
                ',': '<COMMA>',
                '.': '<DOT>',
                "'": '<APOSTR>',
                '!': '<EXCLAM>',
                '?': '<QUESTION>',
                ';': '<SEMICOL>'
            }
        }

    def replace_nonwords(self):
        reverse_map = self.mappings['reverse']

        self.processed = re.sub('\n', reverse_map["\n"], self.data)
        self.processed = re.sub("[,]", reverse_map[","], self.processed)
        self.processed = re.sub("[.]", reverse_map["."], self.processed)
        self.processed = re.sub("[']", reverse_map["'"], self.processed)
        self.processed = re.sub("[!]", reverse_map["!"], self.processed)
        self.processed = re.sub("[?]", reverse_map["?"], self.processed)
        self.processed = re.sub("[;]", reverse_map[";"], self.processed)

    def split_and_enrich(self, text_array, sep_array, sep_index=-1):
        sep_index += 1
        for token in text_array:
            self.split_array = re.split(sep_array[sep_index], token)
            # self.split_array.append(token)
            # self.split_array.append(sep_array[sep_index])

            if sep_index < len(sep_array):
                # sep_index += 1
                self.split_array = self.split_and_enrich(self.split_array, )
