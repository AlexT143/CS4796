import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers.legacy import Adam
from sklearn.metrics import mean_squared_error

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def create_rnn_model(input_shape, n_features):
    model = Sequential()
    model.add(SimpleRNN(64, activation='relu', input_shape=(input_shape[0], n_features)))
    model.add(Dense(n_features))  # Update the output layer to have the same number of units as features
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_and_save_rnn(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    
    print("Validation Loss:")
    for epoch, val_loss in enumerate(history.history['val_loss'], start=1):
        print(f"Epoch {epoch}: {val_loss}")
    
    model_dir = "models/RNN"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/rnn_model.h5")
    
    return model

def make_predictions(model, X_test):
    rnn_predictions = model.predict(X_test)
    return rnn_predictions

def reshape_data(X_train, X_test):
    n_features = X_train.shape[2]  # Get the number of features from the input data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)
    return X_train, X_test

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

if __name__ == "__main__":
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data()

    # Split the training data into training and validation sets
    val_size = int(0.2 * train_size)  # 20% of the training data for validation
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # Reshape the data for RNN input
    X_train, X_test = reshape_data(X_train, X_test)
    X_val, _ = reshape_data(X_val, X_val)

    n_features = X_train.shape[2]
    rnn_model = create_rnn_model(input_shape=(X_train.shape[1], n_features), n_features=n_features)

    # Train and save the RNN model
    epochs = 25
    batch_size = 32
    rnn_model = train_and_save_rnn(rnn_model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # Make predictions using the trained RNN model
    rnn_predictions = make_predictions(rnn_model, X_test)

    # Calculate permutation feature importance
    feature_importance = calculate_permutation_feature_importance(rnn_model, X_test, y_test, scaler)

    # Print feature importance
    feature_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for i, importance in enumerate(feature_importance):
        print(f"{feature_names[i]}: {importance}")