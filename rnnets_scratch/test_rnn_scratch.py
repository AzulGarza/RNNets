import numpy as np
from rnn_scratch import (initialize_parameters, rnn_cell_forward, rnn_forward,
                          rnn_cell_backward, rnn_backward,
                          clip, sample)
np.random.seed(1)


class TestRNN_Fwd(object):
    def test_init_params(self):
        """
        Test the init of params to proper shape
        """
        vocab_size = 29
        parameters = initialize_parameters(70, vocab_size, vocab_size)
        assert parameters["U"].shape == (70, 29)
        assert parameters["W"].shape == (70, 70)

    def test_cell_fwd(self):
        """
        Test the output of one single fwd pass in rnn_cell
        """
        x = np.random.randn(3, 10)
        h_prev = np.random.randn(5, 10)
        W = np.random.randn(5, 5)
        U = np.random.randn(5, 3)
        V = np.random.randn(2, 5)
        b = np.random.randn(5, 1)
        c = np.random.randn(2, 1)
        parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}
        h_next, yt_hat = rnn_cell_forward(parameters, h_prev, x)

        assert yt_hat.shape == (2, 10)
        assert h_next.shape == (5, 10)

    #def test_rnn_fwd(self):
        """
        Test fwd prop of the rnn
        """
       # vocab_size = 29
       # X = [None] + list(range(1, 20, 2))
       # Y = X[1:] + [0]
       # h0 = np.random.randn(70, 1)
       # W = np.random.randn(70, vocab_size)
       # U = np.random.randn(1, 29)
       # V = np.random.randn(2, 5)
       # b = np.random.randn(5, 1)
       # c = np.random.randn(2, 1)
       # parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}
       # loss, caches = rnn_forward(X, Y, h0, parameters, vocab_size=29)
        #assert h.shape == (5, 10, 4)
        #assert y_hat.shape == (2, 10, 4)
    #    assert len(caches) == 2
        #assert np.all([h[4][1], [-0.99999375, 0.77911235, -0.99861469, -0.99833267]])
        #assert np.all([caches[1][1][3], [-1.1425182, -0.34934272, -0.20889423, 0.58662319]])
    # def test_rnn_cell_bwd(self):
    #     """
    #     Test single cell of backwd pass
    #     """
    #     np.random.seed(1)
    #     xt = np.random.randn(3, 10)
    #     h_prev = np.random.randn(5, 10)
    #     U = np.random.randn(5, 3)
    #     W = np.random.randn(5, 5)
    #     V = np.random.randn(2, 5)
    #     b = np.random.randn(5, 1)
    #     c = np.random.randn(2, 1)
    #     parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}

    #     h_next, yt_hat, cache = rnn_cell_forward(xt, h_prev, parameters)

    #     dh_next = np.random.randn(5, 10)

    #     gradients = rnn_cell_backward(dh_next, cache)
    #     assert gradients["dxt"].shape == (3, 10)

    #def test_rnn_bwd(self):
    #    """
    #    Test backprop
    #    """
    #    np.random.seed(1)
    #    x = np.random.randn(3, 10, 4)
    #    h0 = np.random.randn(5, 10)
    #    U = np.random.randn(5, 3)
    #    W = np.random.randn(5, 5)
    #    V = np.random.randn(2, 5)
    #    b = np.random.randn(5, 1)
    #    c = np.random.randn(2, 1)
    #    parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}
    #    h, y, caches = rnn_forward(x, h0, parameters)
    #    dh = np.random.randn(5, 10, 4)
    #    gradients = rnn_backward(dh, caches)
    #    assert gradients["dx"].shape == (3, 10, 4)
        # Por qué la precisión cambia?
    #    assert gradients["dU"][3][1] == 11.264104496527779
        # assert gradients["dU"][3][1] == 11.264104496527777

    #def test_clipped_gradients(self):
    #    """
    #    Test the values of the clipped gradients
    #    """
    #    np.random.seed(3)
    #    dU = np.random.randn(5, 3)*10
    #    dW = np.random.randn(5, 5)*10
    #    dV = np.random.randn(2, 5)*10
    #    db = np.random.randn(5, 1)*10
    #    dc = np.random.randn(2, 1)*10
    #    gradients = {"dU": dU, "dW": dW, "dV": dV, "db": db, "dc": dc}
    #    gradients = clip(gradients, 10)
    #    assert gradients["dW"][1][2] == 10.0
    #    assert gradients["dU"][3][1] == -10.0
    #    assert gradients["dV"][1][2] == 0.2971381536101662
    #    assert gradients["db"][4] == 10.

    #def test_sample_seq_chars(self):
    #    """ Test the sample function of rnn_scratch """
    #    np.random.seed(2)
    #    _, n_h = 20, 100
    #    b = np.random.randn(n_h, 1)
    #    vocab_size = b.shape[0]
    #    U, W, V = np.random.randn(n_h, vocab_size),\
    #        np.random.randn(n_h, n_h), np.random.rand(vocab_size, n_h)
    #    c = np.random.randn(vocab_size, 1)
    #    parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}

    #    indices = sample(parameters, char_to_ix, 0)
