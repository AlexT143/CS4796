import os
import sys
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf

def train_arima(data):
    models = []
    num_features = data.shape[2]  # Get the number of features from the third dimension
    for i in range(num_features):
        print(f"Training ARIMA model for feature {i+1}/{num_features}")
        
        feature_data = data[:, :, i][0]
        print(feature_data[:5])
        print(feature_data.shape)
        model = auto_arima(feature_data, seasonal=False, suppress_warnings=False)
        print("ARIMA model trained successfully")
        models.append(model)
    return models

def train_exponential_smoothing(data, freq=None):
    models = []
    num_features = data.shape[2]  # Get the number of features from the third dimension
    for i in range(num_features):
        print(f"Training Exponential Smoothing model for feature {i+1}/{num_features}")
        series = pd.Series(
            feature_data = data[:, :, i][:, 3],
            index=pd.date_range(start='2010-01-01', periods=len(data), freq=freq),
        )
        model = ExponentialSmoothing(series, trend='add', seasonal='add')
        results = model.fit()
        models.append(results)
    return models

def get_arima_forecast(models, steps, data):
    forecasts = []
    num_features = data.shape[2]  # Get the number of features from the third dimension
    for i, model in enumerate(models):
        print(f"Generating ARIMA forecast for feature {i+1}/{num_features}")
        feature_data = data[:, :, i][:, 3]
        forecast = model.predict(n_periods=steps)
        forecasts.append(forecast)
    return np.array(forecasts).T.reshape(-1, steps, num_features)

def get_es_forecast(models, steps):
    forecasts = []
    num_features = len(models)
    for i, model in enumerate(models):
        print(f"Generating Exponential Smoothing forecast for feature {i+1}/{num_features}")
        forecast = model.forecast(steps=steps)
        forecasts.append(forecast)
    return np.array(forecasts).T.reshape(-1, steps, num_features)