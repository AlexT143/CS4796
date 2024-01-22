import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2020-12-31")

# Select relevant features
features = data[["Open", "High", "Low", "Close", "Volume"]].values

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)


# Create sequences for training
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequence = data[i : i + sequence_length]
        label = data[i + sequence_length, 3]  # using 'Close' price as label
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)


sequence_length = 21  # using 21 days of stock data
sequences, labels = create_sequences(scaled_data, sequence_length)

# Split into training and testing data
train_size = int(len(sequences) * 0.8)
X_train, X_test = sequences[:train_size], sequences[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]


def pos_enc_matrix(L, d, n=10000):
    """Create positional encoding matrix

    Args:
        L: Input dimension (length)
        d: Output dimension (depth), even only
        n: Constant for the sinusoidal functions

    Returns:
        numpy matrix of floats of dimension L-by-d. At element (k,2i) the value
        is sin(k/n^(2i/d)) while at element (k,2i+1) the value is cos(k/n^(2i/d))
    """
    assert d % 2 == 0, "Output dimension needs to be an even integer"
    d2 = d // 2
    P = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1)  # L-column vector
    i = np.arange(d2).reshape(1, -1)  # d-row vector
    denom = np.power(n, -i / d2)  # n**(-2*i/d)
    args = k * denom  # (L,d) matrix
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P


# Plot the positional encoding matrix
pos_matrix = pos_enc_matrix(L=2048, d=512)
assert pos_matrix.shape == (2048, 512)
plt.pcolormesh(pos_matrix, cmap="RdBu")
plt.xlabel("Depth")
plt.ylabel("Position")
plt.colorbar()
plt.show()

assert pos_matrix.shape == (2048, 512)
# Plot the positional encoding matrix, alternative way
plt.pcolormesh(np.hstack([pos_matrix[:, ::2], pos_matrix[:, 1::2]]), cmap='RdBu')
plt.xlabel('Depth')
plt.ylabel('Position')
plt.colorbar()
plt.show()

import numpy as np
import tensorflow as tf


def pos_enc_matrix(L, d, n=10000):
    """Create positional encoding matrix

    Args:
        L: Input dimension (length)
        d: Output dimension (depth), even only
        n: Constant for the sinusoidal functions

    Returns:
        numpy matrix of floats of dimension L-by-d. At element (k,2i) the value
        is sin(k/n^(2i/d)) while at element (k,2i+1) the value is cos(k/n^(2i/d))
    """
    assert d % 2 == 0, "Output dimension needs to be an even integer"
    d2 = d // 2
    P = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1)  # L-column vector
    i = np.arange(d2).reshape(1, -1)  # d-row vector
    denom = np.power(n, -i / d2)  # n**(-2*i/d)
    args = k * denom  # (L,d) matrix
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P


class PositionalEmbedding(tf.keras.layers.Layer):
    """Positional embedding layer. Assume tokenized input, transform into
    embedding and returns positional-encoded output."""

    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        """
        Args:
            sequence_length: Input sequence length
            vocab_size: Input vocab size, for setting up embedding matrix
            embed_dim: Embedding vector size, for setting up embedding matrix
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim  # d_model in paper
        matrix = pos_enc_matrix(sequence_length, embed_dim)
        self.position_embeddings = tf.constant(matrix, dtype="float32")

    def call(self, inputs):
        return inputs + self.position_embeddings[:inputs.shape[1], :]

    # this layer is using an Embedding layer, which can take a mask
    # see https://www.tensorflow.org/guide/keras/masking_and_padding#passing_mask_tensors_directly_to_layers
    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args, **kwargs)

    def get_config(self):
        # to make save and load a model using custom layer possible
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "embed_dim": self.embed_dim,
            }
        )
        return config


seq_length = 20

def self_attention(input_shape, prefix="att", mask=False, **kwargs):
    """Self-attention layers at transformer encoder and decoder. Assumes its
    input is the output from positional encoding layer.

    Args:
        prefix (str): The prefix added to the layer names
        masked (bool): whether to use causal mask. Should be False on encoder and
                       True on decoder. When True, a mask will be applied such that
                       each location only has access to the locations before it.
    """
    # create layers
    inputs = tf.keras.layers.Input(
        shape=input_shape, dtype="float32", name=f"{prefix}_in1"
    )
    attention = tf.keras.layers.MultiHeadAttention(name=f"{prefix}_attn1", **kwargs)
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm1")
    add = tf.keras.layers.Add(name=f"{prefix}_add1")
    # functional API to connect input to output
    attout = attention(query=inputs, value=inputs, key=inputs, use_causal_mask=mask)
    outputs = norm(add([inputs, attout]))
    # create model and return
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}_att")
    return model


seq_length = 20
key_dim = 6
num_heads = 8

model = self_attention(
    input_shape=(seq_length, key_dim), num_heads=num_heads, key_dim=key_dim
)
tf.keras.utils.plot_model(
    model,
    "self-attention.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="BT",
    show_layer_activations=True,
)

import tensorflow as tf


# def cross_attention(input_shape, context_shape, prefix="att", **kwargs):
#     """Cross-attention layers at transformer decoder. Assumes its
#     input is the output from positional encoding layer at decoder
#     and context is the final output from encoder.

#     Args:
#         prefix (str): The prefix added to the layer names
#     """
    
#     # Ensure input_shape and context_shape are tuples of integers
#     assert all(isinstance(dim, int) for dim in input_shape), "input_shape must contain integers"
#     assert all(isinstance(dim, int) for dim in context_shape), "context_shape must contain integers"
    
#     # create layers
#     context = tf.keras.layers.Input(
#         shape=context_shape, dtype="float32", name=f"{prefix}_ctx2"
#     )
#     inputs = tf.keras.layers.Input(
#         shape=input_shape, dtype="float32", name=f"{prefix}_in2"
#     )
#     attention = tf.keras.layers.MultiHeadAttention(name=f"{prefix}_attn2", **kwargs)
#     norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm2")
#     add = tf.keras.layers.Add(name=f"{prefix}_add2")
#     # functional API to connect input to output
#     attout = attention(query=inputs, value=context, key=context)
#     outputs = norm(add([attout, inputs]))
#     print(attout)
#     print("outputs", outputs)
#     # create model and return
#     model = tf.keras.Model(
#         inputs=[context, inputs], outputs=outputs, name=f"{prefix}_cross"
#     )
    
#     return model


# seq_length = 20
# key_dim = 128
# num_heads = 8

# model = cross_attention(
#     input_shape=(seq_length, key_dim),
#     context_shape=(seq_length, key_dim),
#     num_heads=num_heads,
#     key_dim=key_dim,
# )
# tf.keras.utils.plot_model(
#     model,
#     "cross-attention.png",
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir="BT",
#     show_layer_activations=True,
# )

import tensorflow as tf


def feed_forward(input_shape, model_dim, ff_dim, dropout=0.1, prefix="ff"):
    """Feed-forward layers at transformer encoder and decoder. Assumes its
    input is the output from an attention layer with add & norm, the output
    is the output of one encoder or decoder block

    Args:
        model_dim (int): Output dimension of the feed-forward layer, which
                         is also the output dimension of the encoder/decoder
                         block
        ff_dim (int): Internal dimension of the feed-forward layer
        dropout (float): Dropout rate
        prefix (str): The prefix added to the layer names
    """
    # create layers
    inputs = tf.keras.layers.Input(
        shape=input_shape, dtype="float32", name=f"{prefix}_in3"
    )
    dense1 = tf.keras.layers.Dense(ff_dim, name=f"{prefix}_ff1", activation="relu")
    
    dense2 = tf.keras.layers.Dense(model_dim, name=f"{prefix}_ff2")
    drop = tf.keras.layers.Dropout(dropout, name=f"{prefix}_drop")
    add = tf.keras.layers.Add(name=f"{prefix}_add3")
    # functional API to connect input to output
    ffout = drop(dense2(dense1(inputs)))
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm3")
    print("ffout", ffout)
    print("inputs", inputs)
    outputs = norm(add([inputs, ffout]))
    # create model and return
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}_ff")
    return model


# seq_length = 20
# key_dim = 128
# ff_dim = 512

# model = feed_forward(
#     input_shape=(seq_length, key_dim), model_dim=key_dim, ff_dim=ff_dim
# )
# tf.keras.utils.plot_model(
#     model,
#     "feedforward.png",
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir="BT",
#     show_layer_activations=True,
# )


def encoder(input_shape, key_dim, ff_dim, dropout=0.1, prefix="enc", **kwargs):
    """One encoder unit. The input and output are in the same shape so we can
    daisy chain multiple encoder units into one larger encoder"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype='float32', name=f"{prefix}_in0"),
        self_attention(input_shape, prefix=prefix, key_dim=key_dim, mask=False, **kwargs),
        
        print("input_shape", input_shape),
        print("key_dim", key_dim),
        print("ff_dim", ff_dim),
        print("dropout", dropout),
        
        feed_forward(input_shape, key_dim, ff_dim, dropout, prefix),
    ], name=prefix)
    return model

# seq_length = 20
# key_dim = 128
# ff_dim = 512
# num_heads = 8

# # Ensure that num_heads is passed as a keyword argument to the self_attention layer
# model = encoder(input_shape=(seq_length, key_dim), key_dim=key_dim, ff_dim=ff_dim, num_heads=num_heads)
# tf.keras.utils.plot_model(model, "encoder.png", show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='BT', show_layer_activations=True)


def decoder(input_shape, key_dim, ff_dim, dropout=0.1, prefix="dec", **kwargs):
    """One decoder unit. The input and output are in the same shape so we can
    daisy chain multiple decoder units into one larger decoder. The context
    vector is also assumed to be the same shape for convenience"""
    inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32', name=f"{prefix}_in0")
    context = tf.keras.layers.Input(shape=input_shape, dtype='float32', name=f"{prefix}_ctx0")

    # Self-attention layer
    attmodel = self_attention(input_shape, key_dim=key_dim, prefix=prefix, **kwargs)

    # # Cross-attention layer
    # crossmodel = cross_attention(input_shape, input_shape, key_dim=key_dim, prefix=prefix, **kwargs)

    # Feed-forward network
    ffmodel = feed_forward(input_shape, key_dim, ff_dim, dropout, prefix)

    x = attmodel(inputs)
    # x = crossmodel([(context, x)])
    output = ffmodel(x)

    model = tf.keras.Model(inputs=[inputs, context], outputs=output, name=prefix)
    return model

# # Example usage
# seq_length = 20
# key_dim = 128
# ff_dim = 512
# num_heads = 8

# model = decoder(input_shape=(seq_length, key_dim), key_dim=key_dim, ff_dim=ff_dim, num_heads=num_heads)
# tf.keras.utils.plot_model(model, "decoder.png", show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='BT')



def transformer(
    num_layers,
    num_heads,
    seq_len,
    key_dim,
    ff_dim,
    num_features,
    dropout=0.1,
    name="stock_transformer",
):
    print(num_features)
    # Note: num_features corresponds to the number of features in your input data
    input_shape = (seq_len, num_features)  # Adjust input shape

    print(input_shape)
    
    input_layer = tf.keras.layers.Input(
        shape=input_shape, dtype="float32", name="stock_input"
    )

    # You can retain the encoder structure but adapt it for your input data
    x = input_layer
    for i in range(num_layers):
        x = encoder(
            input_shape=input_shape,
            key_dim=key_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            prefix=f"enc{i}",
            num_heads=num_heads,
        )(x)

    # Output layer for regression (predicting stock price)
    output = tf.keras.layers.Dense(1, activation="linear", name="output")(
        x[:, -1, :]
    )  # We only take the last time step's output for prediction

    model = tf.keras.Model(inputs=input_layer, outputs=output, name=name)
    return model


seq_len = 20
num_layers = 4
num_heads = 8
key_dim = 6
ff_dim = 512
dropout = 0.1
model = transformer(
    num_layers,
    num_heads,
    seq_len,
    key_dim,
    ff_dim,
    6,
    dropout,
)
tf.keras.utils.plot_model(
    model,
    "transformer.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="BT",
    show_layer_activations=True,
)

import matplotlib.pyplot as plt
import tensorflow as tf


# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     "Custom learning rate for Adam optimizer"

#     def __init__(self, key_dim, warmup_steps=4000):
#         super().__init__()
#         self.key_dim = key_dim
#         self.warmup_steps = warmup_steps
#         self.d = tf.cast(self.key_dim, tf.float32)

#     def __call__(self, step):
#         step = tf.cast(step, dtype=tf.float32)
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps**-1.5)
#         return tf.math.rsqrt(self.d) * tf.math.minimum(arg1, arg2)

#     def get_config(self):
#         # to make save and load a model using custom layer possible0
#         config = {
#             "key_dim": self.key_dim,
#             "warmup_steps": self.warmup_steps,
#         }
#         return config


key_dim = 6
# lr = CustomSchedule(key_dim)
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

plt.plot(lr(tf.range(50000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()


def masked_loss(label, pred):
    mask = label != 0

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


seq_len = 20
num_layers = 4
num_heads = 8
key_dim = 6
ff_dim = 512
dropout = 0.1
model = transformer(
    num_layers,
    num_heads,
    seq_len,
    key_dim,
    ff_dim,
    6,
    dropout,
)
# lr = CustomSchedule(key_dim)
optimizer = tf.keras.optimizers.legacy.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

import matplotlib.pyplot as plt
import tensorflow as tf


# Create and train the model
seq_len = 20
num_layers = 4
num_heads = 8
key_dim = 6
ff_dim = 512
dropout = 0.1

print("Creating transformer model...")
model = transformer(
    num_layers,
    num_heads,
    seq_len,
    key_dim,
    ff_dim,
    6,
    dropout,
)
print("Transformer model created.")

# Compile the model
print("Compiling the model...")
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
print("Model compilation complete.")

# Train the model
epochs = 20
print("Starting training...")
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
print("Training complete.")

# Plot training history
print("Plotting training history...")
# ... [Your plotting code]
print("Plot complete.")

# Save the model
print("Saving the model...")
model.save("newTransformer.h5")
print("Model saved.")

# Plot the loss and accuracy history
fig, axs = plt.subplots(2, figsize=(6, 8), sharex=True)
fig.suptitle("Training history")
x = list(range(1, epochs + 1))
axs[0].plot(x, history.history["loss"], alpha=0.5, label="loss")
axs[0].plot(x, history.history["val_loss"], alpha=0.5, label="val_loss")
axs[0].set_ylabel("Loss")
axs[0].legend(loc="upper right")
axs[1].plot(x, history.history["masked_accuracy"], alpha=0.5, label="acc")
axs[1].plot(x, history.history["val_masked_accuracy"], alpha=0.5, label="val_acc")
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("epoch")
axs[1].legend(loc="lower right")
plt.show()
