import os
import sys
import pickle
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH
from VanillaTransformerComponents.utils import (
    train_arima,
    train_exponential_smoothing,
    get_arima_forecast,
    get_es_forecast,
)

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data


def train_and_save_model(
    epochs, batch_size, residual_connection_type, use_arima, use_es, model_type
):
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = (
        preprocess_data()
    )

    # Train traditional time series models
    arima_models = train_arima(X_train)
    es_models = train_exponential_smoothing(
        X_train, freq="D"
    )  # Assuming daily frequency

    # Get forecasts from traditional models
    arima_forecast = (
        get_arima_forecast(arima_models, SEQ_LENGTH, X_train) if use_arima else None
    )
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
    mse, mae, mape = calculate_metrics(y_test, y_pred)

    print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}")

    # Save the model
    model_dir = f"models/{model_type}_residual_{residual_connection_type}_arima_{use_arima}_es_{use_es}"
    os.makedirs(model_dir, exist_ok=True)
    tf.keras.models.save_model(transformer, f"{model_dir}/model")

    # Save the scaler
    with open(f"{model_dir}/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Save the necessary variables
    with open(f"{model_dir}/variables.pkl", "wb") as file:
        pickle.dump((X_test, scaled_data, train_size), file)

    return mse, mae, mape


def main():
    epochs = 20
    batch_size = 32
    model_configs = {
        "involatile": {"residual_connection_type": "vanilla", "use_arima": True, "use_es": False},
        "volatile": {"residual_connection_type": "time_series", "use_arima": False, "use_es": True},
        "inline": {"residual_connection_type": "time_series", "use_arima": True, "use_es": True},
    }

    for model_type, config in model_configs.items():
        print(f"Training {model_type} model")
        residual_connection_type = config["residual_connection_type"]
        use_arima = config["use_arima"]
        use_es = config["use_es"]

        mse, mae, mape = train_and_save_model(
            epochs, batch_size, residual_connection_type, use_arima, use_es, model_type
        )
        print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}")


if __name__ == "__main__":
    main()