import yfinance as yf

#Boeing stock
ticker = "BA"
data = yf.download(ticker)

from sklearn.preprocessing import MinMaxScaler
import numpy as np

import tensorflow as tf

# Assume 'data' is your DataFrame from yfinance
scaler = MinMaxScaler()

print(data[:1])
scaled_data = scaler.fit_transform(data)

# Printing the first few rows of the scaled data (as a NumPy array)
print(scaled_data[:1])  # This prints the first 5 rows

actual = scaler.inverse_transform(scaled_data)
print(actual[:1])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 16  # Length of the sequence
X, y = create_sequences(scaled_data, seq_length)



# Splitting the data
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]



def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# Multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)

        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2
 # Modify the model building function to include a decoder
def build_transformer(input_shape, d_model, num_heads, dff, num_decoder_layers, rate=0.1):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inputs)

    pos_encoding = positional_encoding(input_shape[0], d_model)
    x += pos_encoding[:, :input_shape[0], :]

    encoder_output = TransformerEncoderLayer(d_model, num_heads, dff, rate)(x, training=False, mask=None)

    decoder_output = encoder_output
    for _ in range(num_decoder_layers):
        decoder_output, _, _ = TransformerDecoderLayer(d_model, num_heads, dff, rate)(
            decoder_output, encoder_output, training=False, look_ahead_mask=None, padding_mask=None)

    x = tf.keras.layers.Flatten()(encoder_output)
    x = tf.keras.layers.Dense(6, activation='linear')(x)  # Output layer for all 6 features

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# Model parameters
d_model = 12 # The dimensionality of the output space of the Dense layers/embeddings
num_heads = 4  # Number of attention heads
dff = 48  # Dimensionality of the inner layers of the feedforward networks
dropout_rate = 0.1  # Dropout rate
num_decoder_layers = 2  # Number of decoder layers

transformer = build_transformer(
    input_shape=(seq_length, X_train.shape[-1]), 
    d_model=d_model, 
    num_heads=num_heads, 
    dff=dff, 
    num_decoder_layers=num_decoder_layers, 
    rate=dropout_rate)

# Compile the model
transformer.compile(optimizer=tf.keras.optimizers.Adam(), 
                                       loss='mean_squared_error')

# Summary of the model
transformer.summary()

history = transformer.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test))

test_loss = transformer.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# After training, make predictions
predictions = transformer.predict(X_test)

# Directly apply inverse scaling
inverse_scaled_predictions = scaler.inverse_transform(predictions)

print(inverse_scaled_predictions[:5])

# Plot the predictions for closing vs the actual values for closing
import matplotlib.pyplot as plt

plt.plot(inverse_scaled_predictions[:, 3], label='Predicted')
plt.plot(actual[train_size + seq_length:, 3], label='Actual')
plt.legend()
plt.show()

# Prepare the most recent sequence
latest_sequence = scaled_data[-seq_length:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)  # Reshape for the model

# Make the prediction
next_day_prediction = transformer.predict(latest_sequence)

# Inverse scale the prediction
next_day_prediction = scaler.inverse_transform(next_day_prediction)

# Extract the closing price
predicted_closing_price = next_day_prediction[0, 3]
print(f"Predicted Closing Price for Tomorrow: {predicted_closing_price}")


def moving_average(data, window_size):
    """ Returns the moving average of the given data. """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

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
plt.plot(actual[train_size + seq_length:, 3], label='Actual Closing Price', alpha=0.5)
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
