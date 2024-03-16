import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import csv
from datetime import datetime 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def calculate_permutation_feature_importance(model, X_test, y_test, scaler, n_iterations=10):
    baseline_mse = mean_squared_error(y_test[:, 3], scaler.inverse_transform(model.predict(X_test))[:, 3])
    feature_importance = np.zeros((X_test.shape[2], n_iterations))

    for i in range(X_test.shape[2]):
        for j in range(n_iterations):
            X_test_permuted = X_test.numpy().copy()  # Convert to NumPy array and create a copy
            np.random.shuffle(X_test_permuted[:, :, i])
            X_test_permuted = tf.convert_to_tensor(X_test_permuted)  # Convert back to tensor
            mse = mean_squared_error(y_test[:, 3], scaler.inverse_transform(model.predict(X_test_permuted))[:, 3])
            feature_importance[i, j] = mse - baseline_mse

    return np.mean(feature_importance, axis=1)

def train_and_evaluate_model(epochs, batch_size, num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, dropout_rate, volatility_type):
    
    if d_model % num_heads != 0:
        print(f"Skipping combination: d_model={d_model}, num_heads={num_heads} (not divisible)")
        return None, None, None, None, None
    
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data(volatility_type=volatility_type)

    # Cast the input tensors to the expected data type
    X_train = tf.cast(X_train, tf.float32)
    y_train = tf.cast(y_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)
    y_test = tf.cast(y_test, tf.float32)
    
    # Split the training data into training and validation sets
    val_size = int(0.2 * train_size)  # 20% of the training data for validation
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # Cast X_val and y_val after splitting
    X_val = tf.cast(X_val, tf.float32)
    y_val = tf.cast(y_val, tf.float32)

    transformer = build_transformer(
        input_shape=(SEQ_LENGTH, X_train.shape[-1]),
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        rate=dropout_rate,
        attention_mechanism="vanilla",
        positional_encoding_type="vanilla",
        residual_connection_type="vanilla",
    )

    transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss="mse")
    transformer.summary()

    history = transformer.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val)
    )

    test_loss = transformer.evaluate(X_test, y_test)
    print(f"Test Loss for base Transformer with {num_encoder_layers} encoder layers and {num_decoder_layers} decoder layers: {test_loss}")

    return test_loss, transformer, X_test, y_test, scaler

def main():
    epochs = 10
    batch_size = 32
    num_encoder_layers = 0
    num_decoder_layers = 8
    d_model_list = [12, 24, 36, 48]
    num_heads_list = [4, 6, 8, 12]
    dff_list = [6, 12, 18, 24]
    dropout_rate_list = [0.1, 0.2, 0.3, 0.4]
    num_runs = 10  # Number of times to run the training and evaluation for each combination

    results = {}

    feature_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    models_data_folder = "models_data"
    if not os.path.exists(models_data_folder):
        os.makedirs(models_data_folder)

    csv_file = os.path.join(models_data_folder, "vanilla_transformer_hyperparameters_results.csv")
    file_exists = os.path.isfile(csv_file)

    fieldnames = ["run", "num_encoder_layers", "num_decoder_layers", "d_model", "num_heads", "dff", "dropout_rate", "loss", "volatility_type"] + feature_names

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for volatility_type in ["involatile", "volatile", "inline"]:
            for d_model in d_model_list:
                for num_heads in num_heads_list:
                    for dff in dff_list:
                        for dropout_rate in dropout_rate_list:
                            print(f"Training with {num_encoder_layers} encoder layers, {num_decoder_layers} decoder layers, d_model={d_model}, num_heads={num_heads}, dff={dff}, dropout_rate={dropout_rate}, volatility_type={volatility_type}")
                            losses = []
                            feature_importances = []

                            for run in range(num_runs):
                                print(f"Run {run + 1}/{num_runs}")
                                loss, transformer, X_test, y_test, scaler = train_and_evaluate_model(epochs, batch_size, num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, dropout_rate, volatility_type)
                                
                                if transformer is None:
                                    print(f"Skipping combination: d_model={d_model}, num_heads={num_heads} (not divisible)")
                                    break
                                
                                losses.append(loss)

                                # Calculate permutation feature importance
                                feature_importance = calculate_permutation_feature_importance(transformer, X_test, y_test, scaler)
                                feature_importances.append(feature_importance)

                                row = {
                                    "run": run + 1,
                                    "num_encoder_layers": num_encoder_layers,
                                    "num_decoder_layers": num_decoder_layers,
                                    "d_model": d_model,
                                    "num_heads": num_heads,
                                    "dff": dff,
                                    "dropout_rate": dropout_rate,
                                    "loss": loss,
                                    "volatility_type": volatility_type
                                }
                                for i, importance in enumerate(feature_importance):
                                    row[feature_names[i]] = importance
                                writer.writerow(row)

                            if transformer is None:
                                continue

                            avg_loss = np.mean(losses)
                            avg_feature_importance = np.mean(feature_importances, axis=0)
                            results[(num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, dropout_rate, volatility_type)] = (avg_loss, avg_feature_importance)
                            print(f"Average Test Loss for {num_encoder_layers} encoder layers, {num_decoder_layers} decoder layers, d_model={d_model}, num_heads={num_heads}, dff={dff}, dropout_rate={dropout_rate}, volatility_type={volatility_type}: {avg_loss}")

if __name__ == "__main__":
    main()
    
    