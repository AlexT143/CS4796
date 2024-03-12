import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.utils import (
    positional_encoding,
    exponential_smoothing_positional_encoding,
)
from VanillaTransformerComponents.encoder import TransformerEncoderLayer
from VanillaTransformerComponents.decoder import TransformerDecoderLayer

import tensorflow as tf


def build_transformer(
    input_shape,
    d_model,
    num_heads,
    dff,
    num_encoder_layers=1,
    num_decoder_layers=0,
    rate=0.1,
    attention_mechanism="vanilla",
    attention_ar_order=1,
    attention_ma_order=1,
    garch_order=(1, 1),
    positional_encoding_type="vanilla",
    arima_forecast=None,
    es_forecast=None,
    residual_connection_type="vanilla",
    alpha=0.1,
):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inputs)

    if positional_encoding_type == "vanilla":
        pos_encoding = positional_encoding(input_shape[0], d_model)
    elif positional_encoding_type == "exponential_smoothing":
        pos_encoding = exponential_smoothing_positional_encoding(
            input_shape[0], d_model, alpha
        )
    else:
        raise ValueError(
            f"Invalid positional_encoding_type: {positional_encoding_type}"
        )

    x += pos_encoding[:, : input_shape[0], :]

    for _ in range(num_encoder_layers):
        x = TransformerEncoderLayer(d_model, num_heads, dff, rate)(
            x, training=False, mask=None
        )

    if residual_connection_type == "time_series":
        if arima_forecast is not None:
            alpha = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            x = x + alpha * arima_forecast
        if es_forecast is not None:
            beta = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            x = x + beta * es_forecast

    encoder_output = x

    decoder_output = encoder_output

    for _ in range(num_decoder_layers):
        decoder_output, _, _ = TransformerDecoderLayer(
            d_model,
            num_heads,
            dff,
            rate,
            attention_ar_order,
            attention_ma_order,
            garch_order,
            attention_mechanism,
        )(
            decoder_output,
            encoder_output,
            training=False,
            look_ahead_mask=None,
            padding_mask=None,
        )

    x = tf.keras.layers.Flatten()(encoder_output)
    x = tf.keras.layers.Dense(6, activation="linear")(x)  # Output layer for all 6 features

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model
