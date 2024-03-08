import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.attention_mechanism import BaseMultiHeadAttention
# from CustomTransformerModules.ArimaInspiredMultiHeadAttention import ArimaInspiredMultiHeadAttention

import tensorflow as tf

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, attention_ar_order=1, attention_ma_order=1, attention_mechanism='base'):
        super(TransformerEncoderLayer, self).__init__()

        if attention_mechanism == 'base':
            print('Using base attention mechanism in encoder')
            self.mha = BaseMultiHeadAttention(d_model, num_heads)
        elif attention_mechanism == 'arima':
            print('Using ARIMA-inspired attention mechanism in encoder with AR order of', attention_ar_order, 'and MA order of', attention_ma_order)
            
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