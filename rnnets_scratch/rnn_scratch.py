import numpy as np
from .rnn_utils import *


def rnn_cell_forward(xt, h_prev, parameters):
    """ Implements one forward step of the RNN-cell.

    Parameters:
    -----------
    xt : input data at time 't', shape (n_x, m).
    h_prev : Hidden state at time 't-1', shape (n_a, m)
    parameters : python dictionary with the following:
        U : Weight matrix multiplying the input, shape (n_a, n_x)
        W : Weight matrix multiplying the hidden state, shape (n_a, n_a)
        V : Weght matrix relating the hidden-state to the output, shape (n_y, n_a)
        b : Bias, shape (n_a, 1)
        c : Bias relating the hidden-state to the output, shape (n_y, 1)

    Returns:
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
    """ Implements the fwd prop of the rnn

    Parameters:
    -----------
    x : Input data for every time-step 't', shape (n_x, m, T_x).
    h0 : Initial hidden state, shape (n_h, m)
    parameters : python dictionary containing:
        U : Weight matrix multiplying the input, shape (n_a, n_x)
        W : Weight matrix multiplying the hidden state, shape (n_a, n_a)
        V : Weight matrix relating the hidden-state to the output, shape (n_y, n_a) b : Bias, shape (n_a, 1) c : Bias relating the hidden-state to the output, shape (n_y, 1) Returns: --------

    Returns:
    --------
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
        h_next, yt_hat, cache = rnn_cell_forward(x[:, :, t], h_next, parameters)
        h[:, :, t] = h_next
        y_hat[:, :, t] = yt_hat
        caches.append(cache)

    caches = (caches, x)

    return h, y_hat, caches


def rnn_cell_backward(dh_next, cache):
    (h_next, h_prev, xt, parameters) = cache

    U = parameters["U"]
    W = parameters["W"]
    V = parameters["V"]
    b = parameters["b"]
    c = parameters["c"]

    dtanh = (1 - h_next**2) * dh_next

    # Gradient of loss with respect to U
    dxt = U.T @ dtanh
    dU = dtanh @ xt.T

    # Gradient with respect to W
    dh_prev = W.T @ dtanh
    dW = dtanh @ h_prev.T

    # Gradient with respecto to b
    db = np.sum(dtanh, 1, keepdims=True)

    gradients = {"dxt": dxt, "dh_prev": dh_prev, "dU": dU, "dW": dW, "db": db}

    return gradients


def rnn_backward(dh, caches):
    """ Implements the back prop for a RNN
    Parameters:
    -----------
    dh : Gradients of all hidden states, shape (n_a, m, T_X)
    caches : tuple with info form the fwd pass (rnn_forward)

    Returns:
    --------
    gradients: Python dict with:
        dx : Gradient w.r.t. the input, shape (n_x, m, T_x)
        dh0 : Gradient w.r.t. the initial hidden state, shape (n_a, m)
        dU : Gradient w.r.t the input's weight matrix, shape (n_a, n_x)
        dW : Gradient w.r.t. the hidden state's weight matrix, shape (n_a, n_a)
        db : Gradient w.r.t. the bias, shape (n_a, 1)

    """
    (caches, x) = caches
    (h1, h0, x1, parameters) = caches[0]

    n_a, m, T_x = dh.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    dU = np.zeros((n_a, n_x))
    dW = np.zeros((n_a, n_a))
    db = np.zeros((n_a, 1))
    dh0 = np.zeros((n_a, m))
    dh_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(dh[:, :, t] + dh_prevt, caches[t])
        dxt, dh_prevt, dUt, dWt, dbt = gradients["dxt"], gradients["dh_prev"],\
            gradients["dU"], gradients["dW"], gradients["db"]

        dx[:, :, t] = dxt
        dU += dUt
        dW += dWt
        db += dbt

    dh0 = dh_prevt

    gradients = {"dx": dx, "dh0": dh0, "dU": dU, "dW": dW, "db": db}

    return gradients
