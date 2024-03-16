import os
import sys
import numpy as np
import tensorflow as tf
import csv
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_evaluate_model(epochs, batch_size, alpha, volatility_type):
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data(volatility_type=volatility_type)

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
        attention_mechanism="vanilla",
        positional_encoding_type="exponential_smoothing",
        residual_connection_type="vanilla",
        alpha=alpha
    )

    transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss="mse")
    transformer.summary()

    history = transformer.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val)
    )

    test_loss = transformer.evaluate(X_test, y_test)
    print(f"Test Loss for Exponential Smoothing Positional Encoding with alpha {alpha}, Volatility {volatility_type}: {test_loss}")

    return test_loss

def main():
    epochs = 20
    batch_size = 32
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    volatility_types = ["involatile", "volatile", "inline"]
    min_model_count = 5

    models_data_folder = "models_data"
    if not os.path.exists(models_data_folder):
        os.makedirs(models_data_folder)

    csv_file = os.path.join(models_data_folder, "exponential_smoothing_results.csv")
    file_exists = os.path.isfile(csv_file)

    fieldnames = ["run", "alpha", "volatility_type", "loss"]

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for alpha in alphas:
            for volatility_type in volatility_types:
                if file_exists:
                    data = pd.read_csv(csv_file)
                    model_count = len(data[(data["alpha"] == alpha) & (data["volatility_type"] == volatility_type)])
                    num_additional_runs = min_model_count - model_count
                else:
                    num_additional_runs = min_model_count

                if num_additional_runs > 0:
                    print(f"Training {num_additional_runs} additional models for alpha: {alpha}, Volatility: {volatility_type}")
                    for run in range(num_additional_runs):
                        print(f"Run {run + 1}/{num_additional_runs}")
                        loss = train_and_evaluate_model(epochs, batch_size, alpha, volatility_type)

                        row = {
                            "run": run + 1,
                            "alpha": alpha,
                            "volatility_type": volatility_type,
                            "loss": loss
                        }
                        writer.writerow(row)

    results = {}
    data = pd.read_csv(csv_file)
    for alpha in alphas:
        for volatility_type in volatility_types:
            losses = data[(data["alpha"] == alpha) & (data["volatility_type"] == volatility_type)]["loss"].values
            avg_loss = np.mean(losses)
            results[(alpha, volatility_type)] = avg_loss
            print(f"Average Test Loss for alpha {alpha}, Volatility {volatility_type}: {avg_loss}")

    best_combination = min(results, key=results.get)
    best_alpha, best_volatility_type = best_combination
    best_loss = results[best_combination]
    print(f"Best alpha: {best_alpha}, Best Volatility Type: {best_volatility_type}, Best Average Loss: {best_loss}")

if __name__ == "__main__":
    main()