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

# def scale_data(data, online, window_size=100):
#     """ Scale data using MinMaxScaler or Online Normalization and return parameters for unscaling. """
    
#     def online_normalize(data, window_size):
#         # Initialize arrays to store normalized data and parameters for unscaling
#         normalized_data = np.zeros_like(data, dtype=float)
#         parameters = []
        
#         # Iterate through the data
#         for i in range(len(data)):
#             # Determine the window
#             start = max(0, i - window_size + 1)
#             end = i + 1
            
#             # Calculate mean and std for the window
#             window_mean = np.mean(data[start:end])
#             window_std = np.std(data[start:end])
            
#             # Normalize the current data point
#             if window_std != 0:
#                 normalized_data[i] = (data[i] - window_mean) / window_std
#                 parameters.append((window_mean, window_std))
#             else:
#                 normalized_data[i] = 0  # Avoid division by zero
#                 parameters.append((window_mean, 1))  # To avoid division by zero later
        
#         return normalized_data, parameters

#     if online:
#         # Online normalization
#         normalized_data, params = online_normalize(data, window_size)
#         return normalized_data, params
#     else:
#         # Traditional MinMaxScaler normalization
#         scaler = MinMaxScaler()
#         normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
#         return normalized_data, scaler
    
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

def calculate_log_returns(data):
    log_returns = np.log(data / data.shift(1))
    log_returns.drop(log_returns.index[0], inplace=True)
    return log_returns

def zero_mean_normalize(data):
    return (data - data.mean()) / data.std()

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

# def periodic_positional_encoding(position, d_model, periodicity, num_periods):
#     angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[:, np.newaxis] // 2)) / np.float32(d_model))
#     angle_rates_periodic = angle_rates * periodicity

#     angle_rads = np.outer(position % periodicity, angle_rates_periodic[:, 0])
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

#     season_encoding = np.zeros((num_periods, d_model), dtype=np.float32)
#     for i in range(num_periods):
#         season_encoding[i, :] = angle_rads[i * periodicity, :]

#     pos_encoding = angle_rads[np.newaxis, ...]
#     season_index = (position // periodicity) % num_periods
#     pos_encoding += season_encoding[season_index, np.newaxis, :]

#     return tf.cast(pos_encoding, dtype=tf.float32)

def moving_average(data, window_size):
    """ Returns the moving average of the given data. """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size