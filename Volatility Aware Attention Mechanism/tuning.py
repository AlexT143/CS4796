import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_evaluate_model(epochs, batch_size, garch_order):
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
        attention_mechanism="volatility",
        positional_encoding_type="vanilla",
        residual_connection_type="vanilla",
        garch_order=garch_order
    )

    transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss="mse")
    transformer.summary()

    history = transformer.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val)
    )

    test_loss = transformer.evaluate(X_test, y_test)
    print(f"Test Loss for Volatility-Aware Transformer with GARCH order {garch_order}: {test_loss}")

    return test_loss

def main():
    epochs = 20
    batch_size = 32
    garch_orders = [(1, 1), (2, 2), (3, 3)]
    num_runs = 5  # Number of times to run the training and evaluation for each combination

    results = {}

    for garch_order in garch_orders:
        print(f"Training with GARCH order: {garch_order}")
        losses = []

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            loss = train_and_evaluate_model(epochs, batch_size, garch_order)
            losses.append(loss)

        avg_loss = np.mean(losses)
        results[garch_order] = avg_loss
        print(f"Average Test Loss for GARCH order {garch_order}: {avg_loss}")

    best_params = min(results, key=results.get)
    best_garch_order = best_params
    best_loss = results[best_params]

    print(f"Best GARCH order: {best_garch_order}, Best Average Loss: {best_loss}")

if __name__ == "__main__":
    main()