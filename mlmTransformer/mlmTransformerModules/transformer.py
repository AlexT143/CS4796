import tensorflow as tf
from tensorflow.keras.layers import Dense
from encoder import Encoder
from decoder import Decoder

class TransformerModel(tf.keras.Model):
    def __init__(self, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Set up the decoder
        self.decoder = Decoder(dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Define the final dense layer
        # Adjust the output dimension according to your task's requirements
        self.model_last_layer = Dense(1)
    
    def call(self, encoder_input, decoder_input, training):
        # Generate padding masks for encoder and decoder
        enc_padding_mask = self.create_padding_mask(encoder_input)
        dec_padding_mask = self.create_padding_mask(encoder_input)
        
        # Look-ahead mask for the decoder
        dec_lookahead_mask = self.create_lookahead_mask(tf.shape(decoder_input)[1])
        
        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)
        
        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_lookahead_mask, dec_padding_mask, training)
        
        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)
        
        return model_output

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # add extra dimensions to add the padding to the attention logits.

    def create_lookahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
