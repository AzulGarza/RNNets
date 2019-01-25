import numpy as np
from .rnn_scratch import (rnn_cell_forward, rnn_forward,
                          rnn_cell_backward, rnn_backward)
np.random.seed(1)


class TestRNN_Fwd(object):
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

        assert yt_hat.shape == (2, 10)
        assert h_next.shape == (5, 10)

    def test_rnn_fwd(self):
        """
        Test fwd prop of the rnn
        """
        x = np.random.randn(3, 10, 4)
        h0 = np.random.randn(5, 10)
        W = np.random.randn(5, 5)
        U = np.random.randn(5, 3)
        V = np.random.randn(2, 5)
        b = np.random.randn(5, 1)
        c = np.random.randn(2, 1)
        parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}
        h, y_hat, caches = rnn_forward(x, h0, parameters)
        assert h.shape == (5, 10, 4)
        assert y_hat.shape == (2, 10, 4)
        assert len(caches) == 2
        assert np.all([h[4][1], [-0.99999375, 0.77911235, -0.99861469, -0.99833267]])
        assert np.all([caches[1][1][3], [-1.1425182, -0.34934272, -0.20889423, 0.58662319]])

    def test_rnn_cell_bwd(self):
        """
        Test single cell of backwd pass
        """
        np.random.seed(1)
        xt = np.random.randn(3, 10)
        h_prev = np.random.randn(5, 10)
        U = np.random.randn(5, 3)
        W = np.random.randn(5, 5)
        V = np.random.randn(2, 5)
        b = np.random.randn(5, 1)
        c = np.random.randn(2, 1)
        parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}

        h_next, yt_hat, cache = rnn_cell_forward(xt, h_prev, parameters)

        dh_next = np.random.randn(5, 10)

        gradients = rnn_cell_backward(dh_next, cache)
        assert gradients["dxt"].shape == (3, 10)

    def test_rnn_bwd(self):
        np.random.seed(1)
        x = np.random.randn(3, 10, 4)
        h0 = np.random.randn(5, 10)
        U = np.random.randn(5, 3)
        W = np.random.randn(5, 5)
        V = np.random.randn(2, 5)
        b = np.random.randn(5, 1)
        c = np.random.randn(2, 1)
        parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}
        h, y, caches = rnn_forward(x, h0, parameters)
        dh = np.random.randn(5, 10, 4)
        gradients = rnn_backward(dh, caches)
        assert gradients["dx"].shape == (3, 10, 4)
        assert gradients["dU"][3][1] == 11.264104496527779
