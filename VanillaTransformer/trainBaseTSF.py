import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pickle
from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.utils import train_arima, train_exponential_smoothing, get_arima_forecast, get_es_forecast
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH
import tensorflow as tf

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_save_model(
    attention_mechanism,
    positional_encoding_type="vanilla",
    attention_ar_order=1,
    attention_ma_order=1,
    garch_order=(1, 1),
    residual_connection_type="vanilla",
):
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data()

    # Model parameters
    d_model = 12  # The dimensionality of the output space of the Dense layers/embeddings
    num_heads = 4  # Number of attention heads
    dff = 6  # Dimensionality of the inner layers of the feedforward networks
    dropout_rate = 0.1  # Dropout rate
    num_decoder_layers = 1  # Number of decoder layers

    arima_models = None
    es_models = None
    arima_forecast = None
    es_forecast = None
    
    if residual_connection_type == 'time_series':
        # Train ARIMA and Exponential Smoothing models
        print("Training ARIMA")
        arima_models = train_arima(scaled_data)
        print("Training Exponential Smoothing")
        es_models = train_exponential_smoothing(scaled_data, freq='D')  # Assuming daily frequency

        # Get forecasts from ARIMA and Exponential Smoothing models
        arima_forecast = get_arima_forecast(arima_models, SEQ_LENGTH, scaled_data)
        es_forecast = get_es_forecast(es_models, SEQ_LENGTH)
    
    if residual_connection_type == 'time_series':
    # Train ARIMA and Exponential Smoothing models
        print("Training ARIMA")
        arima_models = train_arima(scaled_data)
        print("Training Exponential Smoothing")
        es_models = train_exponential_smoothing(scaled_data, freq='D')  # Assuming daily frequency

        # Get forecasts from ARIMA and Exponential Smoothing models
        arima_forecast = get_arima_forecast(arima_models, SEQ_LENGTH, scaled_data)
        es_forecast = get_es_forecast(es_models, SEQ_LENGTH)

    transformer = build_transformer(
        input_shape=(SEQ_LENGTH, X_train.shape[-1]),
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        num_encoder_layers=0,
        num_decoder_layers=1,
        rate=dropout_rate,
        attention_ar_order=attention_ar_order,
        attention_ma_order=attention_ma_order,
        garch_order=garch_order,
        attention_mechanism=attention_mechanism,
        positional_encoding_type=positional_encoding_type,
        arima_forecast=arima_forecast,
        es_forecast=es_forecast,
        residual_connection_type=residual_connection_type,
    )

    transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss="mse")
    transformer.summary()

    history = transformer.fit(
        X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test)
    )

    test_loss = transformer.evaluate(X_test, y_test)
    print(
        f"Test Loss for {attention_mechanism} attention, {positional_encoding_type} positional encoding, and {residual_connection_type} residual connection: {test_loss}"
    )

    # Save the model
    model_dir = f"models/{attention_mechanism}/{positional_encoding_type}/{residual_connection_type}"
    if attention_mechanism == "volatility":
        model_dir += f"/garch{garch_order[0]}{garch_order[1]}"
    elif attention_mechanism != "vanilla":
        model_dir += f"/ar{attention_ar_order}_ma{attention_ma_order}"
    os.makedirs(model_dir, exist_ok=True)
    tf.keras.models.save_model(transformer, f"{model_dir}/model")

    # Save the scaler
    with open(f"{model_dir}/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Save the necessary variables
    with open(f"{model_dir}/variables.pkl", "wb") as file:
        pickle.dump((X_test, scaled_data, train_size), file)


def main():
    attention_mechanisms = ["vanilla"]
    positional_encoding_types = ["vanilla"]
    residual_connection_types = ["vanilla", 'time_series']
    ar_orders = [1, 2]
    ma_orders = [1, 2]

    for attention_mechanism in attention_mechanisms:
        for positional_encoding_type in positional_encoding_types:
            for residual_connection_type in residual_connection_types:
                if attention_mechanism == "vanilla":
                    train_and_save_model(
                        attention_mechanism,
                        positional_encoding_type,
                        residual_connection_type=residual_connection_type,
                    )
                elif attention_mechanism == "volatility":
                    train_and_save_model(
                        attention_mechanism,
                        positional_encoding_type,
                        garch_order=(1, 1),
                        residual_connection_type=residual_connection_type,
                    )
                else:
                    for ar_order in ar_orders:
                        for ma_order in ma_orders:
                            train_and_save_model(
                                attention_mechanism,
                                positional_encoding_type,
                                ar_order,
                                ma_order,
                                residual_connection_type=residual_connection_type,
                            )


if __name__ == "__main__":
    main()