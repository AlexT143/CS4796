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

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_save_model(epochs, batch_size, alpha, model_type):
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
    print(f"Test Loss for {model_type} Exponential Smoothing Positional Encoding with alpha {alpha}: {test_loss}")

    # Save the model
    model_dir = f"models/{model_type}_exponential_smoothing_alpha_{alpha}"
    os.makedirs(model_dir, exist_ok=True)
    tf.keras.models.save_model(transformer, f"{model_dir}/model")

    # Save the scaler
    with open(f"{model_dir}/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Save the necessary variables
    with open(f"{model_dir}/variables.pkl", "wb") as file:
        pickle.dump((X_test, scaled_data, train_size), file)

    return test_loss

def main():
    epochs = 20
    batch_size = 32
    alphas = {
        "involatile": 0.2,
        "volatile": 0.8,
        "inline": 0.5
    }

    for model_type, alpha in alphas.items():
        print(f"Training {model_type} model with alpha {alpha}")
        test_loss = train_and_save_model(epochs, batch_size, alpha, model_type)
        print(f"Test Loss for {model_type} model: {test_loss}")

if __name__ == "__main__":
    main()