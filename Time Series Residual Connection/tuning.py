import os
import sys
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH
from time_series_utils import train_arima, train_exponential_smoothing, get_arima_forecast, get_es_forecast

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data


def train_and_evaluate_model(
    epochs, batch_size, residual_connection_type, use_arima, use_es
):
    print("Preprocessing data...")
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = (
        preprocess_data()
    )

    print("Training ARIMA models...")
    arima_models = train_arima(X_train)
    print("Training Exponential Smoothing models...")
    es_models = train_exponential_smoothing(
        X_train, freq="D"
    )  # Assuming daily frequency

    print("Getting ARIMA forecasts...")
    arima_forecast = (
        get_arima_forecast(arima_models, SEQ_LENGTH, X_train) if use_arima else None
    )
    print("Getting Exponential Smoothing forecasts...")
    es_forecast = get_es_forecast(es_models, SEQ_LENGTH) if use_es else None
    
    # Model parameters
    d_model = (
        12  # The dimensionality of the output space of the Dense layers/embeddings
    )
    num_heads = 4  # Number of attention heads
    dff = 6  # Dimensionality of the inner layers of the feedforward networks
    dropout_rate = 0.1  # Dropout rate
    num_decoder_layers = 1  # Number of decoder layers

    # Split the training data into training and validation sets
    val_size = int(0.2 * train_size)  # 20% of the training data for validation
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    transformer = build_transformer(
        input_shape=(SEQ_LENGTH, X_train.shape[-1]),
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        num_encoder_layers=1,
        num_decoder_layers=1,
        rate=dropout_rate,
        attention_mechanism="vanilla",
        positional_encoding_type="vanilla",
        residual_connection_type=residual_connection_type,
        arima_forecast=arima_forecast,
        es_forecast=es_forecast,
    )

    transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss="mse")
    transformer.summary()

    history = transformer.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )

    y_pred = transformer.predict(X_test)
    # mse, mae, mape = calculate_metrics(y_test, y_pred)

    # print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}")

    # return mse, mae, mape


def main():
    epochs = 20
    batch_size = 32
    residual_connection_types = ["vanilla", "time_series"]
    use_arima_options = [True, False]
    use_es_options = [True, False]
    num_runs = (
        5  # Number of times to run the training and evaluation for each combination
    )

    results = []

    for residual_connection_type in residual_connection_types:
        for use_arima in use_arima_options:
            for use_es in use_es_options:
                print(
                    f"Training with Residual Connection Type: {residual_connection_type}, Use ARIMA: {use_arima}, Use ES: {use_es}"
                )
                mse_list, mae_list, mape_list = [], [], []
                for run in range(num_runs):
                    print(f"Run {run + 1}/{num_runs}")
                    mse, mae, mape = train_and_evaluate_model(
                        epochs, batch_size, residual_connection_type, use_arima, use_es
                    )
                    mse_list.append(mse)
                    mae_list.append(mae)
                    mape_list.append(mape)
                avg_mse = np.mean(mse_list)
                avg_mae = np.mean(mae_list)
                avg_mape = np.mean(mape_list)
                results.append(
                    {
                        "Residual Connection Type": residual_connection_type,
                        "Use ARIMA": use_arima,
                        "Use ES": use_es,
                        "Average MSE": avg_mse,
                        "Average MAE": avg_mae,
                        "Average MAPE": avg_mape,
                    }
                )

    for result in results:
        print(result)

    best_result = min(results, key=lambda x: x["Average MSE"])
    print(f"Best Configuration: {best_result}")


if __name__ == "__main__":
    main()
