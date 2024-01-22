import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def purge_data(data, years):
    """ Remove data older than a specified number of years. """
    purge_date = pd.Timestamp('today') - pd.DateOffset(years=years)
    return data[data.index >= purge_date]

def embargo_data(data, days):
    """ Exclude the most recent specified number of days. """
    return data[:-days]

def scale_data(data):
    """ Scale data using MinMaxScaler. """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler

def create_sequences(data, seq_length):
    """ Create sequences from the data. """
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_test_split(X, y, train_split):
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, y_train, X_test, y_test, train_size

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def moving_average(data, window_size):
    """ Returns the moving average of the given data. """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size