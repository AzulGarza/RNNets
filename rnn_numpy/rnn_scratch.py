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


def rnn_forward(x, h0, parameters):
    """
    Implements the fwd prop of the rnn

    Parameters:
    -----------
    x : Input data for every time-step 't', shape (n_x, m, T_x).
    h0 : Initial hidden state, shape (n_h, m)
    parameters : python dictionary containing:
        U : Weight matrix multiplying the input, shape (n_a, n_x)
        W : Weight matrix multiplying the hidden state, shape (n_a, n_a)
        V : Weight matrix relating the hidden-state to the output, shape (n_y, n_a)
        b : Bias, shape (n_a, 1)
        c : Bias relating the hidden-state to the output, shape (n_y, 1)
    Returns:
    h : Hidden states for every time-step, shape (n_a, m, T_x)
    y_hat : Predictions for every time-step, shape (n_y, m, T_x)
    caches : tuple needed for backward pass, contains (list of caches, x)
    """
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_h = parameters["V"].shape

    h = np.zeros((n_h, m, T_x))
    y_hat = np.zeros((n_y, m, T_x))
    h_next = h0

    for t in range(T_x):
        h_next, yt_hat, cache = rnn_cell_forward(x[:,:,t], h_next, parameters)
        h[:,:,t] = h_next
        y_hat[:,:,t] = yt_hat
        caches.append(cache)

    caches = (caches, x)

    return h, y_hat, caches