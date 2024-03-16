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

def calculate_permutation_feature_importance(model, X_test, y_test, scaler, n_iterations=10):
    baseline_mse = mean_squared_error(y_test[:, 3], scaler.inverse_transform(model.predict(X_test))[:, 3])
    feature_importance = np.zeros((X_test.shape[2], n_iterations))

    for i in range(X_test.shape[2]):
        for j in range(n_iterations):
            X_test_permuted = X_test.copy()
            np.random.shuffle(X_test_permuted[:, :, i])
            mse = mean_squared_error(y_test[:, 3], scaler.inverse_transform(model.predict(X_test_permuted))[:, 3])
            feature_importance[i, j] = mse - baseline_mse

    return np.mean(feature_importance, axis=1)

def train_and_save_model(epochs, batch_size, ar_order, ma_order, model_type):
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
    print(f"Test Loss for {model_type} ARIMA-Inspired Transformer: {test_loss}")

    # Save the model
    model_dir = f"models/{model_type}_arima_transformer"
    os.makedirs(model_dir, exist_ok=True)
    tf.keras.models.save_model(transformer, f"{model_dir}/model")

    # Save the scaler
    with open(f"{model_dir}/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Save the necessary variables
    with open(f"{model_dir}/variables.pkl", "wb") as file:
        pickle.dump((X_test, scaled_data, train_size), file)

    return transformer, X_test, y_test, scaler

def main():
    epochs = 20
    batch_size = 32

    # Train and save models for different market volatility levels
    volatile_params = (3, 1)  # AR order 3, MA order 1
    involatile_params = (1, 3)  # AR order 1, MA order 3
    inline_params = (2, 2)  # AR order 2, MA order 2

    volatile_transformer, volatile_X_test, volatile_y_test, volatile_scaler = train_and_save_model(epochs, batch_size, *volatile_params, "volatile")
    involatile_transformer, involatile_X_test, involatile_y_test, involatile_scaler = train_and_save_model(epochs, batch_size, *involatile_params, "involatile")
    inline_transformer, inline_X_test, inline_y_test, inline_scaler = train_and_save_model(epochs, batch_size, *inline_params, "inline")

    # Calculate permutation feature importance for each model
    volatile_feature_importance = calculate_permutation_feature_importance(volatile_transformer, volatile_X_test, volatile_y_test, volatile_scaler)
    involatile_feature_importance = calculate_permutation_feature_importance(involatile_transformer, involatile_X_test, involatile_y_test, involatile_scaler)
    inline_feature_importance = calculate_permutation_feature_importance(inline_transformer, inline_X_test, inline_y_test, inline_scaler)

    # Print feature importance for each model
    feature_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    print("Feature Importance for Volatile ARIMA-Inspired Transformer:")
    for i, importance in enumerate(volatile_feature_importance):
        print(f"{feature_names[i]}: {importance}")

    print("\nFeature Importance for Involatile ARIMA-Inspired Transformer:")
    for i, importance in enumerate(involatile_feature_importance):
        print(f"{feature_names[i]}: {importance}")

    print("\nFeature Importance for Inline ARIMA-Inspired Transformer:")
    for i, importance in enumerate(inline_feature_importance):
        print(f"{feature_names[i]}: {importance}")

if __name__ == "__main__":
    main()