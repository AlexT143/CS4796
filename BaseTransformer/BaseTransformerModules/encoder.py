import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.attention import MultiHeadAttention
from BaseTransformerModules.utils import point_wise_feed_forward_network
from BaseTransformerModules.embedding import get_positional_encoding

import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        # # Add a dense layer for feature expansion
        # self.dense = tf.keras.layers.Dense(d_model)
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Print input shape
        #print("EncoderLayer - Input shape:", tf.shape(x))

        # Apply dense layer for feature expansion
        # x = self.dense(x)
        # Print shape after dense layer
        #print("EncoderLayer - After dense shape:", tf.shape(x))

        attn_output, _ = self.mha(x, x, x, mask)  # Self attention
        #print("EncoderLayer - After MHA shape:", tf.shape(attn_output))
        attn_output = self.dropout1(attn_output, training=training)
        #print("EncoderLayer - After dropout shape:", tf.shape(attn_output))
        out1 = self.layernorm1(x + attn_output)  # Add & Norm

        ffn_output = self.ffn(out1)
        #print("EncoderLayer - After FFN shape:", tf.shape(ffn_output))
        ffn_output = self.dropout2(ffn_output, training=training)
        #print("EncoderLayer - After dropout shape:", tf.shape(ffn_output))
        out2 = self.layernorm2(out1 + ffn_output)  # Add & Norm

        # Print shape after processing
        #print("EncoderLayer - Output shape:", tf.shape(out2))

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        rate=0.1,
        use_positional_encoding=True,
        maximum_position_encoding=1000,  # Adjust as needed
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = get_positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        #print("Encoder - Input shape:", tf.shape(x))

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            # After all layers, x should have a shape of [batch_size, seq_len, d_model (128)]

        # Now add positional encoding
        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding[:tf.shape(x)[1], :]
            pos_encoding = tf.expand_dims(pos_encoding, 0)
            x += pos_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            #print(f"Encoder - After layer {i} shape:", tf.shape(x))

        #print("Encoder - Final output shape:", tf.shape(x))
        return x  # (batch_size, input_seq_len, d_model)
