import unittest
from process import TextProcessor


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.text_processor = TextProcessor("datasets/input.txt")
        self.test_seps = [' ', '\n']

    @unittest.skip("in dev")
    def test_preprocess(self):
        self.test_preprocess()


    def test_replace_nonwords(self):
        expected = "O Marcio <COMMA>  <EOL> sono io <COMMA>  ricordi <QUESTION>  <EOL>  <EOL> Quel che t <APOSTR> avea marcito <EOL> Un tempo lontano <DOT>  Ahimé <EXCLAM> "

        processed = self.text_processor.replace_nonwords(self.text_processor.data)
        self.assertEqual(processed, expected)


if __name__ == '__main__':
    unittest.main()
