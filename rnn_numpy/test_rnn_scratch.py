import numpy as np
import unittest
from .rnn_scratch import rnn_cell_forward, rnn_forward
np.random.seed(1)


class TestRNN_Fwd(unittest.TestCase):
    def test_cell_fwd(self):
        """
        Test the output of one single fwd pass in rnn_cell
        """
        xt = np.random.randn(3, 10)
        h_prev = np.random.randn(5, 10)
        W = np.random.randn(5, 5)
        U = np.random.randn(5, 3)
        V = np.random.randn(2, 5)
        b = np.random.randn(5, 1)
        c = np.random.randn(2, 1)
        parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}

        h_next, yt_hat, cache = rnn_cell_forward(xt, h_prev, parameters)

        self.assertEqual(yt_hat.shape, (2, 10))
        # self.assertEqual(h_next[4], [0.59584544 0.18141802 0.61311866 0.99808218 0.85016201 0.99980978 -0.18887155 0.99815551 0.6531151 0.82872037])
        self.assertEqual(h_next.shape, (5, 10))
        # print("yt_hat[1] =", yt_hat[1])
    
    def test_rnn_rwd(self):
        """
        Test fwd prop of the rnn
        """
        x = np.random.randn(3,10,4)
        h0 = np.random.randn(5,10)
        W = np.random.randn(5,5)
        U = np.random.randn(5,3)
        V = np.random.randn(2,5)
        b = np.random.randn(5,1)
        c = np.random.randn(2,1)
        parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}
        h, y_hat, caches = rnn_forward(x, h0, parameters)
        self.assertEqual(h.shape, (5, 10, 4))
        self.assertEqual(y_hat.shape, (2, 10, 4))
        self.assertEqual(len(caches), 2)

if __name__ == '__main__':
    unittest.main()