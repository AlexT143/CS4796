import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.transformer import build_transformer
from BaseTransformerModules.utils import embargo_data, purge_data, scale_data, create_sequences, train_test_split
from BaseTransformerModules.hyperparameters import SEQ_LENGTH

import tensorflow as tf
import yfinance as yf

import joblib

# Download stock data
data = yf.download("BA")

# Apply purging and embargoing
data = purge_data(data, 2) # Purge data older than 2 years
data = embargo_data(data, 5) # Embargo the most recent 5 days

# Scale data
scaled_data, scaler = scale_data(data)

# Create sequences
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split into training and testing sets
X_train, y_train, X_test, y_test, train_size = train_test_split(X, y, 0.8)

# Model parameters
d_model = 12 # The dimensionality of the output space of the Dense layers/embeddings
num_heads = 4  # Number of attention heads
dff = 48  # Dimensionality of the inner layers of the feedforward networks
dropout_rate = 0.1  # Dropout rate
num_decoder_layers = 2  # Number of decoder layers

transformer = build_transformer(
    input_shape=(SEQ_LENGTH, X_train.shape[-1]), 
    d_model=d_model, 
    num_heads=num_heads, 
    dff=dff, 
    num_decoder_layers=num_decoder_layers, 
    rate=dropout_rate)

transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')

transformer.summary()

history = transformer.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

test_loss = transformer.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Using a relative path
transformer.save("models/baseTSF")
joblib.dump(scaler, "models/baseTSF/scaler.save")