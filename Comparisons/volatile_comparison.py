import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from data_preprocessing import preprocess_data

def load_model_and_data(model_dir):
    model = tf.keras.models.load_model(f"{model_dir}/model")
    with open(f"{model_dir}/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    with open(f"{model_dir}/variables.pkl", "rb") as file:
        X_test, _, _ = pickle.load(file)
    return model, scaler, X_test

def evaluate_model(model, scaler, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test[:, 3], scaler.inverse_transform(y_pred)[:, 3])
    return mse

def main():
    volatile_models = [
        "models/volatile_residual_time_series_arima_False_es_True",
        "models/volatile_arima_transformer",
        "models/volatile_volatility_transformer",
        "models/volatile_exponential_smoothing_alpha_0.8"
    ]

    results = []

    for model_dir in volatile_models:
        model, scaler, X_test = load_model_and_data(model_dir)
        _, _, _, y_test, _, _, _ = preprocess_data()  # Load the original y_test data
        mse = evaluate_model(model, scaler, X_test, y_test)
        results.append({"Model": model_dir, "MSE": mse})

    print("Volatile Models Performance Comparison:")
    for result in results:
        print(f"Model: {result['Model']}, MSE: {result['MSE']:.4f}")

    best_model = min(results, key=lambda x: x["MSE"])
    print(f"\nBest Volatile Model: {best_model['Model']}, MSE: {best_model['MSE']:.4f}")

if __name__ == "__main__":
    main()