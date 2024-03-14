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

def train_and_evaluate_model(epochs, batch_size, ar_order, ma_order):
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data()

    # Model parameters
    d_model = 12  # The dimensionality of the output space of the Dense layers/embeddings
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
        attention_mechanism="arima_inspired",
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
    print(f"Test Loss for ARIMA-Inspired Transformer with AR order {ar_order} and MA order {ma_order}: {test_loss}")

    return test_loss

def main():
    epochs = 20
    batch_size = 32
    ar_orders = [1, 2, 3]
    ma_orders = [1, 2, 3]
    num_runs = 5  # Number of times to run the training and evaluation for each combination

    results = {}

    models_data_folder = "models_data"
    if not os.path.exists(models_data_folder):
        os.makedirs(models_data_folder)

    csv_file = os.path.join(models_data_folder, "arima_transformer_results.csv")
    file_exists = os.path.isfile(csv_file)

    fieldnames = ["run", "ar_order", "ma_order", "loss"]

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for ar_order in ar_orders:
            for ma_order in ma_orders:
                print(f"Training with AR order: {ar_order}, MA order: {ma_order}")
                losses = []

                for run in range(num_runs):
                    print(f"Run {run + 1}/{num_runs}")
                    loss = train_and_evaluate_model(epochs, batch_size, ar_order, ma_order)
                    losses.append(loss)

                    row = {
                        "run": run + 1,
                        "ar_order": ar_order,
                        "ma_order": ma_order,
                        "loss": loss
                    }
                    writer.writerow(row)

                avg_loss = np.mean(losses)
                results[(ar_order, ma_order)] = avg_loss
                print(f"Average Test Loss for AR order {ar_order} and MA order {ma_order}: {avg_loss}")

    best_params = min(results, key=results.get)
    best_ar_order, best_ma_order = best_params
    best_loss = results[best_params]

    print(f"Best AR order: {best_ar_order}, Best MA order: {best_ma_order}, Best Average Loss: {best_loss}")

if __name__ == "__main__":
    main()