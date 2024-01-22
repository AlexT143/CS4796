
import tensorflow as tf
import numpy as np
import joblib
import yfinance as yf


from BaseTransformerModules.transformer import Transformer
from BaseTransformerModules.attention import MultiHeadAttention

custom_objects = {
    'Transformer': Transformer,
    'MultiHeadAttention': MultiHeadAttention
    # Include other custom classes if there are any
}

transformer = tf.keras.models.load_model("models/baseTSF", custom_objects=custom_objects)

# Load the MinMaxScaler
scaler = joblib.load("models/baseTSF/scalar.save")

# Fetch new data for prediction
new_data = yf.download("BAC", start="2021-01-01")
print(new_data.tail())
new_features = new_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
scaled_new_data = scaler.transform(new_features)

# Assuming seq_length and other relevant variables are set as in the training script
recent_data = scaled_new_data[-16:]
predicted_price, _ = transformer(recent_data, np.zeros_like(recent_data), False, None, None, None)

print(predicted_price)

# predicted_price_transformed = scaler.inverse_transform(inverse_transform_input)[0][3]

# print(f"Predicted Close Price: {predicted_price_transformed}")
