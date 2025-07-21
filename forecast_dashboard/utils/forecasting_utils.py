import numpy as np

def create_sequences(data, n_input, n_output):
    """
    Create sequences for CNN-LSTM input.

    Parameters:
        data (ndarray): 2D array of shape (time, features)
        n_input (int): Number of past time steps to use as input
        n_output (int): Number of time steps to predict (default 6)

    Returns:
        X (ndarray): shape (samples, n_input, features)
        y (ndarray): shape (samples, n_output, features)
    """
    X, y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        X.append(data[i:i+n_input])
        y.append(data[i+n_input:i+n_input+n_output])
    return np.array(X), np.array(y)

