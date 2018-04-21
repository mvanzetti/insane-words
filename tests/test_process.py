import unittest
from process import TextProcessor


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.text_processor = TextProcessor("datasets/input.txt")
        self.test_seps = [' ', '\n']

    def test_replace_nonwords(self):
        expected = "O Marcio <COMMA>  <EOL> sono io <COMMA>  ricordi <QUESTION>  <EOL>  <EOL> Quel che t <APOSTR> avea marcito <EOL> Un tempo lontano <DOT>  Ahimé <EXCLAM> "

        processed = self.text_processor.replace_nonwords(self.text_processor.data)
        self.assertEqual(processed, expected)

    def test_vocab_manipulation(self):
        self.text_processor.build_vocabulary("datasets", "vocab_tmp")

        words_set = self.text_processor.words_set
        word_indices = self.text_processor.word_indices
        indices_word = self.text_processor.indices_word

        self.text_processor.load_vocabulary("datasets", "vocab_tmp")

        self.assertEqual(words_set, self.text_processor.words_set)
        self.assertEqual(word_indices, self.text_processor.word_indices)
        self.assertEqual(indices_word, self.text_processor.indices_word)


if __name__ == '__main__':
    unittest.main()
