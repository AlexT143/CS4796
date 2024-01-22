import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.encoder import Encoder
from BaseTransformerModules.decoder import Decoder

import tensorflow as tf

from tensorflow.keras.layers import TimeDistributed


class Transformer(tf.keras.Model):
    def __init__(
        self, num_layers, d_model, num_heads, dff, pe_input, pe_target, rate=0.1
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate, pe_input)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate, pe_target)

        # Replace the final dense layer with one suitable for regression or classification
        self.final_layer = TimeDistributed(tf.keras.layers.Dense(1))

    def call(
        self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask
    ):
        # print("Transformer - Input shape:", tf.shape(inp))  # Input shape

        enc_output = self.encoder(inp, training, enc_padding_mask)
        # print("Transformer - Encoder output shape:", tf.shape(enc_output))  # Encoder output shape

        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # Apply final Dense layer to each time step of each sequence
        final_output = self.final_layer(dec_output)

        # print("Transformer - Final output shape:", tf.shape(final_output))  # Final output shape

        return final_output, attention_weights
