import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    # capitalize first character
    txt = txt[0].upper() + txt[1:]
    print('%s' % (txt, ), end='')


def initialize_parameters(n_h, n_x, n_y):
    """
    Initialize parameters with small random values

    Returns:
    --------
    parameters : python dictionary containing:
        U -- Weight matrix multiplying the input, shape (n_h, n_x)
        W -- Weight matrix multiplying the hidden state, shape (n_h, n_h)
        V -- Weight matrix of hidden-state to the output, shape (n_y, n_h)
        b -- Bias, numpy array of shape (n_h, 1)
        c -- Bias relating the hidden-state to the output, shape (n_y, 1)
    """
    np.random.seed(1)
    U = np.random.randn(n_h, n_x)*0.01  # input to hidden
    W = np.random.randn(n_h, n_h)*0.01  # hidden to hidden
    V = np.random.randn(n_y, n_h)*0.01  # hidden to output
    b = np.zeros((n_h, 1))  # hidden bias
    c = np.zeros((n_y, 1))  # output bias

    parameters = {"U": U, "W": W, "V": V, "b": b, "c": c}

    return parameters


def rnn_cell_forward(parameters, h_prev, x):
    """ Implements one forward step of the RNN-cell.

    Parameters:
    -----------
    parameters : python dictionary with the following:
    x : input data at time 't', shape (n_x, m).
    h_prev : Hidden state at time 't-1', shape (n_h, m)
        U : Weight matrix multiplying the input, shape (n_h, n_x)
        W : Weight matrix multiplying the hidden state, shape (n_h, n_h)
        V : Weight matrix of hidden-state to the output, shape (n_y, n_h)
        b : Bias, shape (n_h, 1)
        c : Bias relating the hidden-state to the output, shape (n_y, 1)

    Returns:
    -------
    h_next : next hidden state, shape (n_h, m)
    yt_hat : prediction at time 't', shape (n_y, m)
    """

    U, W, V, b, c = parameters["U"], parameters["W"], parameters["V"], \
        parameters["b"], parameters["c"]

    h_next = np.tanh(np.dot(U, x) + np.dot(W, h_prev) + b)
    yt_hat = softmax(np.dot(V, h_next) + c)

    return h_next, yt_hat


def rnn_forward(X, Y, h0, parameters, vocab_size=29):
    """ Implements the fwd prop of the rnn

    Parameters:
    -----------
    X : Input data for every time-step 't', shape (n_x, m, T_x).
    Y : Output for every time-step 't'
    h0 : Initial hidden state, shape (n_h, m)
    vocab_size : Number of unique characters in your vocabulary
    parameters : python dictionary containing:
        U : Weight matrix multiplying the input, shape (n_h, n_x)
        W : Weight matrix multiplying the hidden state, shape (n_h, n_h)
        V : Weight matrix of hidden-state to the output, shape (n_y, n_h)
        b : Bias, shape (n_h, 1)
        c : Bias relating the hidden-state to the output, shape (n_y, 1)
    Returns:
    --------
    loss : Value of Negative Log-Likelihood
    cache : Tuple with y_hat, h, and x
    """
    x, h, y_hat = {}, {}, {}

    h[-1] = np.copy(h0)

    loss = 0

    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))

        if (X[t] is not None):
            x[t][X[t]] = 1

        h[t], y_hat[t] = rnn_cell_forward(parameters, h[t-1], x[t])

        loss -= np.log(y_hat[t][Y[t], 0])

    cache = (y_hat, h, x)

    return loss, cache


def rnn_cell_backward(dy, gradients, parameters, x, h, h_prev):
    """ Implements the bwd pass for RNN-cell
    Parameters:
    -----------
    dy :
    gradients : python dict with gradients of every matrix and bias.
    parameters : python dict with matrices and bias.
    x : Character input at every time t.
    h : Current hidden state.
    h_prev : Previous hidden state.
    Returns:
    --------
    gradients : python dict with gradients.
    """
    gradients['dV'] += np.dot(dy, h.T)
    gradients['dc'] += dy
    dh = np.dot(parameters['V'].T, dy) + gradients['dh_next']
    dtanh = (1 - h * h) * dh
    gradients['db'] += dtanh
    gradients['dU'] += np.dot(dtanh, x.T)
    gradients['dW'] += np.dot(dtanh, h_prev.T)
    gradients['dh_next'] = np.dot(parameters['W'].T, dtanh)
    return gradients


def rnn_backward(X, Y, parameters, cache):
    """ Implements the back prop for a RNN
    Parameters:
    -----------
    X :
    Y :
    parameters : python dict with matrices and bias.
    cache : Tuple with y_hat, h, and x

    Returns:
    --------
    gradients : Python dict with the gradients
    h : Hidden state
    """
    gradients = {}
    (y_hat, h, x) = cache
    W, U, V, c, b = parameters['W'], parameters['U'], parameters['V'], \
        parameters['c'], parameters['b']

    gradients['dW'], gradients['dU'] = np.zeros_like(W), np.zeros_like(U)
    gradients['dV'] = np.zeros_like(V)

    gradients['db'], gradients['dc'] = np.zeros_like(b), np.zeros_like(c)

    gradients['dh_next'] = np.zeros_like(h[0])

    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_cell_backward(dy, gradients, parameters, x[t], h[t], h[t-1])

    return gradients, h


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length


def update_parameters(parameters, gradients, lr):
    """ Update the parameters according to gradient descent """
    parameters['U'] += -lr * gradients['dU']
    parameters['W'] += -lr * gradients['dW']
    parameters['V'] += -lr * gradients['dV']
    parameters['b'] += -lr * gradients['db']
    parameters['c'] += -lr * gradients['dc']
    return parameters


def clip(gradients, maxValue):
    """
    Clips the gradients' values between minimum and maximum

    Parameters:
    -----------
    gradients : a dict containing the gradients "dU", "dW", "dV", "db", "dc"
    maxValue : Set the value of gradients between [-maxValue, maxValue]

    Returns:
    --------
    gradients : a dictionary with the clipped gradients
    """
    dW, dU, dV, db, dc = gradients["dW"], gradients["dU"], gradients["dV"], \
        gradients["db"], gradients["dc"]

    for gradient in [dU, dW, dV, db, dc]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dW": dW, "dU": dU, "dV": dV, "db": db, "dc": dc}

    return gradients


def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability
    distributions output of the RNN
    Parameters:
    -----------
    parameters : python dictionary containing W, U, V, b, c.
    char_to_ix : python dictionary mapping each character to an index.
    Returns:
    --------
    indices : a list of length n containign the indices of the samples chars.
    """
    W, U, V, b, c = parameters["W"], parameters["U"], parameters["V"], \
        parameters["b"], parameters["c"]
    vocab_size = c.shape[0]
    n_h = W.shape[1]

    x = np.zeros((vocab_size, 1))
    h_prev = np.zeros((n_h, 1))
    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):
        h = np.tanh(np.matmul(U, x) + np.matmul(W, h_prev) + b)
        o = np.matmul(V, h) + c
        y = softmax(o)

        np.random.seed(counter+seed)

        idx = np.random.choice(range(vocab_size), p=y.ravel())

        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        h_prev = h

        seed += 1
        counter += 1

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices


def optimize(X, Y, h_prev, parameters, learning_rate=0.01):
    """
    Execute one step of the optimization to train the model.

    Parameters:
    -----------
    X : list of integers, each int is a number mappint to a char in vocab.
    Y : list of integers, same as X but shifted one index to the left.
    h_prev : previous hidden state.
    parameters -- python dictionary containing:
        U : Weight matrix multiplying the input, shape (n_h, n_x)
        W : Weight matrix multiplying the hidden state, shape (n_h, n_h)
        V : Weight matrix of hidden-state to the output, shape (n_y, n_h)
        b :  Bias, numpy array of shape (n_h, 1)
        c :  Bias relating the hidden-state to the output, shape (n_y, 1)
        learning_rate : learning rate for the model.

    Returns:
    --------
    loss : value of the loss function (cross-entropy)
    gradients : python dictionary containing:
        dU -- Gradients of input-to-hidden weights, of shape (n_h, n_x)
        dW -- Gradients of hidden-to-hidden weights, of shape (n_h, n_h)
        dV -- Gradients of hidden-to-output weights, of shape (n_y, n_h)
        db -- Gradients of bias vector, of shape (n_h, 1)
        dc -- Gradients of output bias vector, of shape (n_y, 1)
    h[len(X)-1] -- the last hidden state, of shape (n_h, 1)
    """

    loss, cache = rnn_forward(X, Y, h_prev, parameters, vocab_size=29)
    gradients, h = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, h[len(X)-1]


def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_h=70, names=9, vocab_size=29):
    """
    Trains the model and generates names.

    Parameters:
    -----------
    data : text corpus
    ix_to_char : dictionary that maps the index to a character
    char_to_ix : dictionary that maps a character to an index
    num_iterations : number of iterations to train the model for
    n_h : number of units of the RNN cell
    names : number of names you want to sample at each iteration.
    vocab_size : number of unique characters found in the text.

    Returns:
    --------
    parameters : learned parameters
    """
    n_x, n_y = vocab_size, vocab_size

    parameters = initialize_parameters(n_h, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, names)

    # Build list of all names (training examples).
    with open("../data/names/Chinese.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state
    h_prev = np.zeros((n_h, 1))

    # Optimization loop
    for j in range(num_iterations):

        # Define one training example (X,Y)
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # Perform one optimization step:
        # Forward-prop -> Backward-prop -> Clip -> Update parameters
        curr_loss, gradients, h_prev = optimize(X, Y, h_prev, parameters, learning_rate=0.01)

        # Use a latency trick to keep the loss smooth.
        # It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample()
        # to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of names to print
            seed = 0
            for name in range(names):

                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                # To get the same result, increment the seed by one.
                seed += 1

            print('\n')

    return parameters


if __name__ == '__main__':
    data = open('../data/names/Chinese.txt', 'r').read()
    data = data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print(f'Existen {data_size} caracteres en total \
          y {vocab_size} caracteres únicos en el vocabulario')
    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    print(f'Diccionario índice to caracter: {ix_to_char}')
    print('Generando nombres del país seleccionado')
    parameters = model(model, ix_to_char, char_to_ix)
