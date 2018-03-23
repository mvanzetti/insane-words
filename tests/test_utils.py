import unittest
from utils import TextProcessor


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.text_processor = TextProcessor("datasets/input.txt")
        self.test_seps = [' ', '\n']

    @unittest.skip("Maybe deprecated")
    def test_split_and_enrich(self):
        self.text_processor.split_and_enrich([self.test_string], self.test_seps)
        self.assertEqual(self.text_processor.split_array, ["Ciao,", "come", "va?", "\n", "Bene,", "grazie!"])

    def test_replace_newline(self):
        self.text_processor.replace_nonwords()
        self.assertEqual(self.text_processor.processed,
                         "Ciao<COMMA> come va<QUESTION><EOL>Bene<COMMA> grazie<EXCLAM><EOL><EOL>Evviva evviva<EXCLAM><EOL>C<APOSTR>è l<APOSTR>olio d<APOSTR>oliva<DOT>")


if __name__ == '__main__':
    unittest.main()
