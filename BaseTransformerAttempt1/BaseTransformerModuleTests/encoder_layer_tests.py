import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.encoder import (
    Encoder,
    EncoderLayer,
)

import tensorflow as tf
import unittest


class TestEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 256
        self.num_heads = 8
        self.dff = 512
        self.rate = 0.1
        self.encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)

    def test_output_shape(self):
        batch_size = 1
        seq_length = 16  # Example sequence length
        sample_input = tf.random.uniform((batch_size, seq_length, self.d_model))
        sample_output = self.encoder_layer(sample_input, False, None)
        # Check if the output shape is as expected
        self.assertEqual(sample_output.shape, (batch_size, seq_length, self.d_model))

    # Additional tests for behavior under different conditions, training modes, and masks

class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.num_layers = 2
        self.d_model = 6  # Even with 5 features, adding a dense layer then converts it to 6 for the heads: open, high, low, close, volume
        self.num_heads = 2  # Adjust as appropriate, ensuring d_model is divisible by num_heads
        self.dff = 512
        self.rate = 0.1

        # You might need to adjust the Encoder class if it's designed for NLP tasks
        self.encoder = Encoder(
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.dff,
            rate=self.rate
        )

    def test_output_shape(self):
        batch_size = 1
        seq_length = 16  # Example: 16 days of trading data
        sample_input = tf.random.uniform((batch_size, seq_length, self.d_model))
        sample_output = self.encoder(sample_input, False, None)
        self.assertEqual(sample_output.shape, (batch_size, seq_length, self.d_model))

if __name__ == "__main__":
    unittest.main()

