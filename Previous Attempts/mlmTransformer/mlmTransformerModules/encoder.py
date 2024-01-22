
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout, Input
from tensorflow.keras import Model
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from mlmTransformerModules.utils import AddNormalization, FeedForward, PositionalEncoding, MultiHeadAttention

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.build(input_shape=[None, sequence_length, d_model])
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
 
    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))
 
    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)
 
        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)
 
        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)
 
# Implementing the Encoder
class Encoder(Layer):
    def __init__(self, sequence_length, d_model, d_ff, n, rate, h, d_k, d_v, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
 
    def call(self, input_data, training):
        # Apply positional encoding
        pos_encoding_output = self.pos_encoding(input_data)
        
        # Apply dropout
        x = self.dropout(pos_encoding_output, training=training)
        
        # Pass the data through each encoder layer
        for layer in self.encoder_layer:
            x = layer(x, training)
        
        return x
