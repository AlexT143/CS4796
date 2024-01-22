import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.decoder import (
    Decoder,
    DecoderLayer,
)

import tensorflow as tf
import unittest

class TestDecoderLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 256
        self.num_heads = 8
        self.dff = 512
        self.rate = 0.1
        self.decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate)

    def test_output_shape(self):
        batch_size = 1
        seq_length = 16  # Example sequence length
        sample_input = tf.random.uniform((batch_size, seq_length, self.d_model))
        sample_output, _, _ = self.decoder_layer(sample_input, sample_input, False, None, None)
        # Check if the output shape is as expected
        self.assertEqual(sample_output.shape, (batch_size, seq_length, self.d_model))

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.num_layers = 2
        self.d_model = 6
        self.num_heads = 2
        self.dff = 512
        self.rate = 0.1
        self.maximum_position_encoding = 1000
        self.decoder = Decoder(
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.dff,
            rate=self.rate,
            maximum_position_encoding=self.maximum_position_encoding
        )

    def test_output_shape(self):
        batch_size = 1
        seq_length = 16
        sample_input = tf.random.uniform((batch_size, seq_length, self.d_model))
        enc_output = tf.random.uniform((batch_size, seq_length, self.d_model))
        sample_output, _ = self.decoder(sample_input, enc_output, False, None, None)
        self.assertEqual(sample_output.shape, (batch_size, seq_length, self.d_model))

if __name__ == "__main__":
    unittest.main()
