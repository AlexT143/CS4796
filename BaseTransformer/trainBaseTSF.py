import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import os
import sys
import tensorflow as tf

import joblib

#print("Setting up directories...")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

#print("Importing Transformer model...")
from BaseTransformerModules.transformer import Transformer  # Adjust the import path accordingly

#print("Fetching and preprocessing data...")
stock_data = yf.download("BAC", end="2019-05-01")
#print("Data fetched successfully.")

features = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)
#print("Data scaled successfully.")

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]  # Sequence of features
        y = data[i + seq_length]      # Next set of features
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 16
X, y = create_sequences(scaled_data, seq_length)
#print(f"Sequences created. Total number of sequences: {len(X)}")

#print(X, y)


# Model and training setup
#print("Setting up the model...")
num_layers = 2
d_model = 6
num_heads = 6
dff = 24
pe_input = seq_length
pe_target = 1
rate = 0.25

transformer = Transformer(num_layers, d_model, num_heads, dff, pe_input, pe_target, rate)
#print("Model setup complete.")

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.legacy.Adam()

epochs = 10  # Define the number of epochs
batch_size = 32  # Define the batch size


# Assuming 'scaled_data' is your full dataset
train_size = int(len(scaled_data) * 0.8)  # 80% for training
train_data = scaled_data[:train_size]
validation_data = scaled_data[train_size:]

# Create sequences for training and validation
X_train, y_train = create_sequences(train_data, seq_length)
X_val, y_val = create_sequences(validation_data, seq_length)


# Training loop
print("Starting training...")
for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}/{epochs}")

    # Training Loop
    for batch in range(len(X) // batch_size):
        inp = X[batch * batch_size : (batch + 1) * batch_size]
        tar = y[batch * batch_size : (batch + 1) * batch_size]

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar, True, None, None, None)

            print("Training Shapes - inp:", inp.shape, "tar:", tar.shape, "predictions:", predictions.shape)
            loss = loss_object(tar, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        print(f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{len(X)//batch_size}, Loss: {loss.numpy()}")

    # Validation Loop
    val_loss = 0
    for batch in range(len(X_val) // batch_size):
        inp = X_val[batch * batch_size : (batch + 1) * batch_size]
        tar = y_val[batch * batch_size : (batch + 1) * batch_size]

        predictions, _ = transformer(inp, np.zeros_like(inp), False, None, None, None)

        print("Validation Shapes - inp:", inp.shape, "tar:", tar.shape, "predictions:", predictions.shape)
        val_loss += loss_object(tar, predictions).numpy()
        
    val_loss /= max(len(X_val) // batch_size, 1)  # Avoid division by zero
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")

# Using a relative path
transformer.save("models/baseTSF")
joblib.dump(scaler, "models/baseTSF/scaler.save")
