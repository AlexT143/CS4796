import os

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)

import pickle

from BaseTransformerModules.transformer import build_transformer

from BaseTransformerModules.utils import embargo_data, purge_data, scale_data, create_sequences, train_test_split

from BaseTransformerModules.hyperparameters import SEQ_LENGTH

import tensorflow as tf

import yfinance as yf

import joblib

def main():
# Download stock data
    data = yf.download("MCD", start="2010-01-01")
    # Apply purging and embargoing
    #data = purge_data(data, 2) # Purge data older than 2 years

    data = embargo_data(data, SEQ_LENGTH) # Embargo the most recent 5 days
    # Scale data

    scaled_data, scaler = scale_data(data)
    
    # Create sequences
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # Split into training and testing sets

    X_train, y_train, X_test, y_test, train_size = train_test_split(X, y, 0.8)

    # Model parameters
    d_model = 12 # The dimensionality of the output space of the Dense layers/embeddings
    num_heads = 4 # Number of attention heads
    dff = 6 # Dimensionality of the inner layers of the feedforward networks
    dropout_rate = 0.1 # Dropout rate
    num_decoder_layers = 1 # Number of decoder layers
    
    transformer = build_transformer(
        input_shape=(SEQ_LENGTH, X_train.shape[-1]),
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        num_encoder_layers=1,
        num_decoder_layers=1,
        rate=dropout_rate,
        attention_ar_order=1,
        attention_ma_order=1,
        attention_mechanism='base'
    )
    transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')

    transformer.summary()

    history = transformer.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    test_loss = transformer.evaluate(X_test, y_test)

    print(f"Test Loss: {test_loss}")

    # Save the model
    tf.keras.models.save_model(transformer, "models/baseTSF/model")

    # Save the scaler
    with open("models/baseTSF/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Save the necessary variables
    with open("models/baseTSF/variables.pkl", "wb") as file:
        pickle.dump((X_test, scaled_data, train_size), file)

if __name__ == "__main__":
    main()