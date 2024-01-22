# import yfinance as yf
# from transformer import Transformer
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# import tensorflow as tf

# stock_data = yf.download('MSFT', end='2017-01-01')

# features = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(features)
# #print("Data scaled successfully.")

# def create_sequences(data, seq_length):
#     xs, ys = [], []
#     for i in range(len(data) - seq_length):
#         x = data[i:(i + seq_length)]  # Sequence of features
#         y = data[i + seq_length, 3]  # Index 3 for 'Close' price
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

# seq_length = 16
# X, y = create_sequences(scaled_data, seq_length)

# # Now, X is your training data and y is your training labels
# train_data = X
# train_labels = y

# sequence_length = 16  # Length of input sequences
# d_model = 6  # Embedding dimension
# d_ff = 24  # Feed forward dimension
# n = 6  # Number of layers
# h = 6  # Number of attention heads
# rate = 0.1  # Dropout rate
# d_k = 64  # Key dimension
# d_v = 64  # Value dimension

# # Assuming you have defined the Transformer class as discussed
# transformer = Transformer(sequence_length=sequence_length, d_model=d_model, d_ff=d_ff, n=n, rate=rate, h=h, d_k=d_k, d_v=d_v)

# # Compile the model
# transformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                     loss='mean_squared_error')



# # Training hyperparameters
# batch_size = 32
# epochs = 10

# # Train the model
# transformer.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)

# h = 8  # Number of self-attention heads
# d_k = 64  # Dimensionality of the linearly projected queries and keys
# d_v = 64  # Dimensionality of the linearly projected values
# d_ff = 2048  # Dimensionality of the inner fully connected layer
# d_model = 512  # Dimensionality of the model sub-layers' outputs
# n = 6  # Number of layers in the encoder stack

# batch_size = 64  # Batch size from the training process
# dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

from encoder import EncoderLayer

encoder = EncoderLayer(16, 8, 64, 64, 512, 2048, 0.1)
encoder.build_graph().summary()


from decoder import DecoderLayer

decoder = DecoderLayer(16, 8, 64, 64, 512, 2048, 0.1)
decoder.build_graph().summary()
