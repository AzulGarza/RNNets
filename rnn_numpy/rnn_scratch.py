import numpy as np
from .rnn_utils import *


def rnn_cell_forward(xt, h_prev, parameters):
    """
    Implements one forward step of the RNN-cell.

    Parameters:
    xt : input data at time 't', shape (n_x, m).
    h_prev : Hidden state at time 't-1', shape (n_a, m)
    parameters : python dictionary with the following:
        U : Weight matrix multiplying the input, shape (n_a, n_x)
        W : Weight matrix multiplying the hidden state, shape (n_a, n_a)
        V : Weght matrix relating the hidden-state to the output, shape (n_y, n_a)
        b : Bias, shape (n_a, 1)
        c : Bias relating the hidden-state to the output, shape (n_y, 1)

    Returns
    -------
    h_next : next hidden state, shape (n_a, m)
    yt_hat : prediction at time 't', shape (n_y, m)
    cache : tuple of values needed for backward, contains (a_next, a_prev, xt, parameters)
    """

    U = parameters["U"]
    W = parameters["W"]
    V = parameters["V"]
    b = parameters["b"]
    c = parameters["c"]

    h_next = np.tanh(b + (W @ h_prev) + (U @ xt))
    yt_hat = softmax(c + (V @ h_next))

    cache = (h_next, h_prev, xt, parameters)

    return h_next, yt_hat, cache
