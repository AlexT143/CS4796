import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.utils import positional_encoding
from BaseTransformerModules.encoder import TransformerEncoderLayer
from BaseTransformerModules.decoder import TransformerDecoderLayer

import tensorflow as tf

def build_transformer(input_shape, d_model, num_heads, dff, num_encoder_layers=1, num_decoder_layers=0, rate=0.1, attention_ar_order=1, attention_ma_order=1, attention_mechanism='base'):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inputs)

    pos_encoding = positional_encoding(input_shape[0], d_model)
    x += pos_encoding[:, :input_shape[0], :]

    encoder_output = x
    for _ in range(num_encoder_layers):
        encoder_output = TransformerEncoderLayer(d_model, num_heads, dff, rate)(encoder_output, training=False, mask=None)

    decoder_output = encoder_output
    for _ in range(num_decoder_layers):
        decoder_output, _, _ = TransformerDecoderLayer(d_model, num_heads, dff, rate, attention_ar_order, attention_ma_order, attention_mechanism)(
            decoder_output, encoder_output, training=False, look_ahead_mask=None, padding_mask=None)

    x = tf.keras.layers.Flatten()(encoder_output)
    x = tf.keras.layers.Dense(6, activation='linear')(x)  # Output layer for all 6 features

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model