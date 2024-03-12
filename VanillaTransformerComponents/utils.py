import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from hyperparameters import SEQ_LENGTH


def purge_data(data, years):
    """Remove data older than a specified number of years."""
    purge_date = pd.Timestamp("today") - pd.DateOffset(years=years)
    return data[data.index >= purge_date]


def embargo_data(data, days):
    """Exclude the most recent specified number of days."""
    return data[:-days]


def scale_data(data):
    """Scale data using MinMaxScaler."""
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
    """Create sequences from the data."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_test_split(X, y, train_size):
    train_samples = int(train_size * len(X))
    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]
    return X_train, y_train, X_test, y_test, train_samples


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def calculate_log_returns(data):
    log_returns = np.log(data / data.shift(1))
    log_returns.drop(log_returns.index[0], inplace=True)
    return log_returns


def zero_mean_normalize(data):
    return (data - data.mean()) / data.std()


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def exponential_smoothing_positional_encoding(position, d_model, alpha=0.1):
    # Standard sinusoidal positional encoding
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Compute exponential smoothing weights
    pos = np.arange(position)
    weights = alpha * (1 - alpha) ** (pos - 1)
    weights = weights[:, np.newaxis]

    # Multiply positional encoding with exponential smoothing weights
    espe = angle_rads * weights

    return tf.cast(espe[np.newaxis, ...], dtype=tf.float32)


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from pmdarima import auto_arima


def train_arima(data):
    models = []
    num_features = data.shape[1] // SEQ_LENGTH
    for i in range(num_features):
        feature_data = data[:, i * SEQ_LENGTH : (i + 1) * SEQ_LENGTH].flatten()
        model = auto_arima(feature_data, seasonal=False, suppress_warnings=True)
        models.append(model)
    return models


def train_exponential_smoothing(data, trend="add", seasonal="add", freq=None):
    models = []
    for i in range(data.shape[1]):
        # Convert data to pandas Series with DatetimeIndex
        series = pd.Series(
            data[:, i],
            index=pd.date_range(start="2010-01-01", periods=len(data), freq=freq),
        )
        model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal)
        results = model.fit()
        models.append(results)
    return models


def get_arima_forecast(models, steps, data):
    data_reshaped = data.reshape(data.shape[0], -1)
    forecasts = []
    for i, model in enumerate(models):
        try:
            feature_data = data_reshaped[:, i * SEQ_LENGTH : (i + 1) * SEQ_LENGTH].flatten()
            forecast = model.forecast(steps=steps)
            forecasts.append(forecast)
        except Exception as e:
            print(
                f"ARIMA forecast failed for feature {i}: {str(e)}. Using last observed value as forecast."
            )
            last_value = feature_data[-1]
            forecast = np.full(steps, last_value)
            forecasts.append(forecast)
    return np.array(forecasts).T.reshape(-1, SEQ_LENGTH, data.shape[-1])


def get_es_forecast(models, steps):
    forecasts = []
    for model in models:
        forecast = model.forecast(steps=steps)
        forecasts.append(forecast)
    return np.array(forecasts).T
