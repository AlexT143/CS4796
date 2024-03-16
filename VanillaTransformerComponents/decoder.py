import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from VanillaTransformerComponents.attention_mechanism import BaseMultiHeadAttention
from CustomTransformerComponents.arma_multi_head_attention import ARMAMultiHeadAttention
from CustomTransformerComponents.VolatilityAwareMultiHeadAttention import VolatilityAwareMultiHeadAttention

import tensorflow as tf

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, ar_order=1, ma_order=1, garch_order=(1, 1), attention_mechanism="vanilla"):
        super(TransformerDecoderLayer, self).__init__()

        if attention_mechanism == "vanilla":
            print('Using base attention mechanism in decoder')
            self.mha1 = BaseMultiHeadAttention(d_model, num_heads)
            self.mha2 = BaseMultiHeadAttention(d_model, num_heads)
        elif attention_mechanism == 'arma':
            print('Using ARIMA-inspired attention mechanism in decoder with AR order of', ar_order, 'and MA order of', ma_order)
            self.mha1 = ARMAMultiHeadAttention(d_model, num_heads, ar_order, ma_order)
            self.mha2 = ARMAMultiHeadAttention(d_model, num_heads, ar_order, ma_order)
        elif attention_mechanism == 'volatility':
            print('Using volatility-aware attention mechanism in decoder')
            self.mha1 = VolatilityAwareMultiHeadAttention(d_model, num_heads, garch_order)
            self.mha2 = VolatilityAwareMultiHeadAttention(d_model, num_heads, garch_order)
        

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