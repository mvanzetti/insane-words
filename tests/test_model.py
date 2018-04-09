import unittest
from model import Model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.test_model = Model(512, 10, 20, 0.01)

    def test_remove_checkpoints(self):
        # todo complete test
        self.test_model.set_saves_folder("saves")
        self.test_model.remove_checkpoints()


if __name__ == '__main__':
    unittest.main()
