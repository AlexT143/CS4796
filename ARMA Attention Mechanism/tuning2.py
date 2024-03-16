import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_evaluate_model(epochs, batch_size, ar_order, ma_order, volatility_type, num_runs):
    results = []

    for run in range(num_runs):
        try:
            # Preprocess the data
            X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data(volatility_type=volatility_type)

            # Split the training data into training and validation sets
            val_size = int(0.2 * train_size)  # 20% of the training data for validation
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]

            if volatility_type == "volatile":
                num_decoder_layers = 8
                d_model = 36
                num_heads = 4
                dff = 12
                
            elif volatility_type == "involatile":
                num_decoder_layers = 5
                d_model = 24
                num_heads = 4
                dff = 6
            elif volatility_type == "inline":
                num_decoder_layers = 7
                d_model = 24
                num_heads = 2
                dff = 12
                
                
            transformer = build_transformer(
                input_shape=(SEQ_LENGTH, X_train.shape[-1]),
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                num_encoder_layers=0,
                num_decoder_layers=num_decoder_layers,
                rate=0.1,
                attention_mechanism="arma",
                positional_encoding_type="vanilla",
                residual_connection_type="vanilla",
                attention_ar_order=ar_order,
                attention_ma_order=ma_order
            )

            transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss="mse")
            transformer.summary()

            history = transformer.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val)
            )

            test_loss = transformer.evaluate(X_test, y_test)
            print(f"Test Loss for ARIMA-Inspired Transformer with AR order {ar_order}, MA order {ma_order}, Volatility {volatility_type}: {test_loss}")

            results.append({
                "run": run + 1,
                "ar_order": ar_order,
                "ma_order": ma_order,
                "volatility_type": volatility_type,
                "loss": test_loss
            })

        except tf.errors.InvalidArgumentError as e:
            print(f"Caught InvalidArgumentError for AR order {ar_order}, MA order {ma_order}, Volatility {volatility_type}: {str(e)}")
            continue

    return pd.DataFrame(results)

def main():
    epochs = 20
    batch_size = 32
    ar_orders = [1, 2, 3, 4]
    ma_orders = [ 1, 2, 3, 4]
    volatility_types = ["involatile", "volatile", "inline"]

    models_data_folder = "models_data"
    if not os.path.exists(models_data_folder):
        os.makedirs(models_data_folder)

    csv_file = os.path.join(models_data_folder, "arma_transformer_results.csv")
    file_exists = os.path.isfile(csv_file)

    fieldnames = ["run", "ar_order", "ma_order", "volatility_type", "loss"]

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        for ar_order in ar_orders:
            for ma_order in ma_orders:
                for volatility_type in volatility_types:
                    if file_exists:
                        data = pd.read_csv(csv_file)
                        model_count = len(data[(data["ar_order"] == ar_order) & (data["ma_order"] == ma_order) & (data["volatility_type"] == volatility_type)])
                        num_additional_runs = 30 - model_count
                    else:
                        num_additional_runs = 30

                    if num_additional_runs > 0:
                        print(f"Training {num_additional_runs} additional models for AR order: {ar_order}, MA order: {ma_order}, Volatility: {volatility_type}")
                        results = train_and_evaluate_model(epochs, batch_size, ar_order, ma_order, volatility_type, num_additional_runs)

                        for _, result_row in results.iterrows():
                            writer.writerow(result_row.to_dict())


if __name__ == "__main__":
    main()