import os
import sys
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.transformer import build_transformer
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_evaluate_model(epochs, batch_size, alpha):
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
    print(f"Test Loss for Exponential Smoothing Positional Encoding with alpha {alpha}: {test_loss}")

    return test_loss

def main():
    epochs = 5
    batch_size = 32
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    num_runs = 4  # Number of times to run the training and evaluation for each combination

    results = {}

    for alpha in alphas:
        print(f"Training with alpha: {alpha}")
        losses = []
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            loss = train_and_evaluate_model(epochs, batch_size, alpha)
            losses.append(loss)
        avg_loss = np.mean(losses)
        results[alpha] = avg_loss
        print(f"Average Test Loss for alpha {alpha}: {avg_loss}")

    best_alpha = min(results, key=results.get)
    best_loss = results[best_alpha]
    print(f"Best alpha: {best_alpha}, Best Average Loss: {best_loss}")

if __name__ == "__main__":
    main()