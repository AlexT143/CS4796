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

def train_test_split(X, y, train_size):
    train_samples = int(train_size * len(X))
    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]
    return X_train, y_train, X_test, y_test, train_samples