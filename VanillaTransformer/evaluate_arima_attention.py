import os
import sys
import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.hyperparameters import SEQ_LENGTH
from BaseTransformerModules.utils import create_sequences, train_test_split


sys.path.insert(0, root_dir)

from Baselines.TrainARIMABaseline import evaluate_arima
from Baselines.LSTMBaseline import evaluate_lstm

def evaluate_model(model_path, scaler_path, variables_path):
    # Load the saved model
    transformer = tf.keras.models.load_model(model_path)

    # Load the scaler
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    # Load the necessary variables
    with open(variables_path, "rb") as file:
        X_test, scaled_data, train_size = pickle.load(file)

    # Make predictions on the test set
    y_pred = transformer.predict(X_test)

    # Inverse scale the predictions and test data
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(
        scaled_data[train_size : train_size + len(X_test)]
    )

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse)

    return mse, mae, mape


def main():
    attention_mechanisms = ["vanilla", "volatility"]
    ar_orders = [1, 2]
    ma_orders = [1, 2]

    # Evaluate ARIMA baseline
    with open("models/base/variables.pkl", "rb") as file:
        _, scaled_data, train_size = pickle.load(file)

    arima_predictions = evaluate_arima(scaled_data, train_size)
    arima_predictions_inverse = scaler.inverse_transform(arima_predictions)
    mse = mean_squared_error(y_test_inverse, arima_predictions_inverse)
    mae = mean_absolute_error(y_test_inverse, arima_predictions_inverse)
    mape = mean_absolute_percentage_error(y_test_inverse, arima_predictions_inverse)
    print("Evaluation Metrics for ARIMA baseline:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print()

    # Evaluate LSTM baseline
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    X_train, y_train, X_test, y_test, _ = train_test_split(X, y, train_size)
    lstm_predictions = evaluate_lstm(X_train, y_train, X_test)
    lstm_predictions_inverse = scaler.inverse_transform(lstm_predictions)
    mse = mean_squared_error(y_test_inverse, lstm_predictions_inverse)
    mae = mean_absolute_error(y_test_inverse, lstm_predictions_inverse)
    mape = mean_absolute_percentage_error(y_test_inverse, lstm_predictions_inverse)
    print("Evaluation Metrics for LSTM baseline:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print()
    
    for attention_mechanism in attention_mechanisms:
        if attention_mechanism == "vanilla":
            model_path = f"models/{attention_mechanism}/model"
            scaler_path = f"models/{attention_mechanism}/scaler.pkl"
            variables_path = f"models/{attention_mechanism}/variables.pkl"

            mse, mae, mape = evaluate_model(model_path, scaler_path, variables_path)
            print(f"Evaluation Metrics for {attention_mechanism} attention:")
            print(f"Mean Squared Error (MSE): {mse}")
            print(f"Mean Absolute Error (MAE): {mae}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape}")
            print()
        elif attention_mechanism == "volatility":
            for garch_order in [(1, 1)]:
                model_path = f"models/{attention_mechanism}/garch{garch_order[0]}{garch_order[1]}/model"
                scaler_path = f"models/{attention_mechanism}/garch{garch_order[0]}{garch_order[1]}/scaler.pkl"
                variables_path = f"models/{attention_mechanism}/garch{garch_order[0]}{garch_order[1]}/variables.pkl"

                mse, mae, mape = evaluate_model(model_path, scaler_path, variables_path)
                print(
                    f"Evaluation Metrics for {attention_mechanism} attention with GARCH order {garch_order[0]}{garch_order[1]}:"
                )
                print(f"Mean Squared Error (MSE): {mse}")
                print(f"Mean Absolute Error (MAE): {mae}")
                print(f"Mean Absolute Percentage Error (MAPE): {mape}")
                print()
        else:
            for ar_order in ar_orders:
                for ma_order in ma_orders:
                    model_path = (
                        f"models/{attention_mechanism}/ar{ar_order}_ma{ma_order}/model"
                    )
                    scaler_path = f"models/{attention_mechanism}/ar{ar_order}_ma{ma_order}/scaler.pkl"
                    variables_path = f"models/{attention_mechanism}/ar{ar_order}_ma{ma_order}/variables.pkl"

                    mse, mae, mape = evaluate_model(
                        model_path, scaler_path, variables_path
                    )
                    print(
                        f"Evaluation Metrics for {attention_mechanism} attention with AR order {ar_order} and MA order {ma_order}:"
                    )
                    print(f"Mean Squared Error (MSE): {mse}")
                    print(f"Mean Absolute Error (MAE): {mae}")
                    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
                    print()


if __name__ == "__main__":
    main()
