import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from datetime import datetime
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data



def calculate_permutation_feature_importance(
    model, X_test, y_test, scaler, n_iterations=10
):
    baseline_mse = mean_squared_error(
        y_test[:, 3], scaler.inverse_transform(model.predict(X_test))[:, 3]
    )
    feature_importance = np.zeros((X_test.shape[2], n_iterations))

    for i in range(X_test.shape[2]):
        for j in range(n_iterations):
            X_test_permuted = X_test.copy()
            np.random.shuffle(X_test_permuted[:, :, i])
            mse = mean_squared_error(
                y_test[:, 3],
                scaler.inverse_transform(model.predict(X_test_permuted))[:, 3],
            )
            feature_importance[i, j] = mse - baseline_mse

    return np.mean(feature_importance, axis=1)


def train_and_evaluate_model(
    epochs, batch_size, num_encoder_layers, num_decoder_layers, volatility_type
):
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = (
        preprocess_data(volatility_type=volatility_type)
    )

    # Model parameters
    d_model = (
        12  # The dimensionality of the output space of the Dense layers/embeddings
    )
    num_heads = 4  # Number of attention heads
    dff = 6  # Dimensionality of the inner layers of the feedforward networks
    dropout_rate = 0.1  # Dropout rate

    # Split the training data into training and validation sets
    val_size = int(0.2 * train_size)  # 20% of the training data for validation
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

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

    start_time = datetime.now()
    history = transformer.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()

    test_loss = transformer.evaluate(X_test, y_test)
    print(
        f"Test Loss for base Transformer with {num_encoder_layers} encoder layers and {num_decoder_layers} decoder layers: {test_loss}"
    )

    return test_loss, transformer, X_test, y_test, scaler, training_time, volatility_type


def main():
    feature_names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    models_data_folder = "models_data"
    if not os.path.exists(models_data_folder):
        os.makedirs(models_data_folder)

    csv_file = os.path.join(models_data_folder, "vanilla_transformer_structure_results.csv")
    file_exists = os.path.isfile(csv_file)

    fieldnames = ["run", "num_encoder_layers", "num_decoder_layers", "loss", "training_time", "volatility_type"] + feature_names

    epochs = 20
    batch_size = 32
    num_encoder_layers_list = [0]
    num_decoder_layers_list = [5, 6, 7, 8, 9, 10]
    num_runs = 10  # Number of times to run the training and evaluation for each combination

    results = {
        (num_encoder_layers, num_decoder_layers): (float("inf"), None)
        for num_encoder_layers in num_encoder_layers_list
        for num_decoder_layers in num_decoder_layers_list
    }

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for volatility_type in ["involatile", "inline", "volatile"]:
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs} for {volatility_type}")
                for num_encoder_layers in num_encoder_layers_list:
                    for num_decoder_layers in num_decoder_layers_list:
                        loss, transformer, X_test, y_test, scaler, training_time, volatility_type = train_and_evaluate_model(
                            epochs, batch_size, num_encoder_layers, num_decoder_layers, volatility_type
                        )

                        # Calculate permutation feature importance
                        feature_importance = calculate_permutation_feature_importance(
                            transformer, X_test, y_test, scaler
                        )

                        key = (num_encoder_layers, num_decoder_layers)
                        if loss < results[key][0]:
                            results[key] = (loss, feature_importance)

                        row = {
                            "run": run + 1,
                            "num_encoder_layers": num_encoder_layers,
                            "num_decoder_layers": num_decoder_layers,
                            "loss": loss,
                            "training_time": training_time,
                            "volatility_type": volatility_type,
                        }
                        for i, importance in enumerate(feature_importance):
                            row[feature_names[i]] = importance
                        writer.writerow(row)

    best_params = min(results, key=lambda x: results[x][0])
    best_num_encoder_layers, best_num_decoder_layers = best_params
    best_loss, best_feature_importance = results[best_params]

    print(
        f"Best Number of Encoder Layers: {best_num_encoder_layers}, Best Number of Decoder Layers: {best_num_decoder_layers}, Best Average Loss: {best_loss}"
    )

    # Print feature importance for the best model
    for i, importance in enumerate(best_feature_importance):
        print(f"{feature_names[i]}: {importance}")



if __name__ == "__main__":
    main()