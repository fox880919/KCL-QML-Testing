import unittest
from qsvm import train_qsvm

class TestQSVM(unittest.TestCase):
    def test_accuracy(self):
        accuracy = train_qsvm()
        self.assertGreaterEqual(accuracy, 0.8, "QSVM accuracy is too low!")

if __name__ == "__main__":
    unittest.main()