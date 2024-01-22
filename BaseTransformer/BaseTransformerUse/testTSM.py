import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerUse.trainTSM import X_train, X_test, y_train, y_test, train_size, scaled_data
from BaseTransformerModules.hyperparameters import SEQ_LENGTH
from BaseTransformerModules.utils import moving_average
from BaseTransformerModules.attention_mechanism import BaseMultiHeadAttention

import tensorflow as tf
import numpy as np

import joblib

custom_objects = {
    'MultiHeadAttention': BaseMultiHeadAttention
}

transformer = tf.keras.models.load_model("models/baseTSF", custom_objects=custom_objects)

# Load the MinMaxScaler
scaler = joblib.load("models/baseTSF/scaler.save")

# After training, make predictions
predictions = transformer.predict(X_test)

# Directly apply inverse scaling
inverse_scaled_predictions = scaler.inverse_transform(predictions)

print(inverse_scaled_predictions[:5])

actual = scaler.inverse_transform(scaled_data)

# Plot the predictions for closing vs the actual values for closing
import matplotlib.pyplot as plt

plt.plot(inverse_scaled_predictions[:, 3], label='Predicted')
plt.plot(actual[train_size + SEQ_LENGTH:, 3], label='Actual')
plt.legend()
plt.show()


# Prepare the most recent sequence
latest_sequence = scaled_data[-SEQ_LENGTH:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)  # Reshape for the model

# Make the prediction
next_day_prediction = transformer.predict(latest_sequence)

# Inverse scale the prediction
next_day_prediction = scaler.inverse_transform(next_day_prediction)

# Extract the closing price
predicted_closing_price = next_day_prediction[0, 3]
print(f"Predicted Closing Price for Tomorrow: {predicted_closing_price}")

# Calculate moving averages
short_term_window = 5
long_term_window = 20

# Calculate moving averages on predicted prices
short_term_ma_pred = moving_average(inverse_scaled_predictions[:, 3], short_term_window)
long_term_ma_pred = moving_average(inverse_scaled_predictions[:, 3], long_term_window)

# Adjust the length of short_term_ma_pred to match long_term_ma_pred
short_term_ma_pred = short_term_ma_pred[-len(long_term_ma_pred):]

# Generate signals based on predicted prices
buy_signals_pred = (short_term_ma_pred[:-1] < long_term_ma_pred[:-1]) & (short_term_ma_pred[1:] > long_term_ma_pred[1:])
sell_signals_pred = (short_term_ma_pred[:-1] > long_term_ma_pred[:-1]) & (short_term_ma_pred[1:] < long_term_ma_pred[1:])

# Adjust the length of the signal arrays
buy_signals_pred_adjusted = np.append([False], buy_signals_pred)  # Add False at the beginning
sell_signals_pred_adjusted = np.append([False], sell_signals_pred)  # Add False at the beginning

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(actual[train_size + SEQ_LENGTH:, 3], label='Actual Closing Price', alpha=0.5)
plt.plot(range(short_term_window-1, len(short_term_ma_pred)+short_term_window-1), short_term_ma_pred, label='Predicted Short Term MA', alpha=0.7)
plt.plot(range(long_term_window-1, len(long_term_ma_pred)+long_term_window-1), long_term_ma_pred, label='Predicted Long Term MA', alpha=0.7)

# Plot buy and sell signals based on predictions
plt.scatter(np.where(buy_signals_pred_adjusted)[0], short_term_ma_pred[buy_signals_pred_adjusted], marker='^', color='g', label='Buy Signal on Prediction', alpha=1)
plt.scatter(np.where(sell_signals_pred_adjusted)[0], short_term_ma_pred[sell_signals_pred_adjusted], marker='v', color='r', label='Sell Signal on Prediction', alpha=1)

plt.title('AI Predicted Stock Price with Buy and Sell Signals')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

capital = 1000
holdings = 0
buy_price = 0

for i in range(len(buy_signals_pred_adjusted)):
    predicted_price = inverse_scaled_predictions[i, 3]

    if buy_signals_pred_adjusted[i] and capital > 0:
        # Buy
        holdings = capital / predicted_price
        buy_price = predicted_price
        capital = 0  # Invest all capital
        print(f"Buying at {predicted_price} on day {i}, holdings: {holdings}")

    elif sell_signals_pred_adjusted[i] and holdings > 0:
        # Sell
        capital = holdings * predicted_price
        holdings = 0  # Sell all holdings
        print(f"Selling at {predicted_price} on day {i}, capital: {capital}")

# Final capital calculation
if holdings > 0:
    capital = holdings * inverse_scaled_predictions[-1, 3]  # Final day's price

print(f"Final capital: {capital}")

