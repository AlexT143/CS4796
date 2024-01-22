import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.attention import MultiHeadAttention
from BaseTransformerModules.utils import point_wise_feed_forward_network
from BaseTransformerModules.embedding import get_positional_encoding

import tensorflow as tf

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        #print("DecoderLayer - Input x shape:", tf.shape(x))
        #print("DecoderLayer - Input enc_output shape:", tf.shape(enc_output))

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # Self-attention
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        #print("DecoderLayer - After MHA1 out1 shape:", tf.shape(out1))

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # Encoder-decoder attention
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # Add & Norm
        #print("DecoderLayer - After MHA2 out2 shape:", tf.shape(out2))

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # Add & Norm
        #print("DecoderLayer - After FFN out3 shape:", tf.shape(out3))

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, maximum_position_encoding=1000):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = get_positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        #print("Decoder - Input x shape:", tf.shape(x))
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        #print("Decoder - Input x shape:", tf.shape(x))
        #print("Decoder - Input enc_output shape:", tf.shape(enc_output))

        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding[:seq_len, :]
            pos_encoding = tf.expand_dims(pos_encoding, 0)  # Add batch dimension
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += pos_encoding  # Apply positional encoding
            #print("Decoder - After pos_encoding x shape:", tf.shape(x))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            #print(f"Decoder - After layer {i} x shape:", tf.shape(x))

            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        return x, attention_weights
