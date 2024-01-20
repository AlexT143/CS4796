import os
import sys
import unittest
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.transformer import Transformer  # Adjust the import path accordingly

class TestFinancialTransformer(unittest.TestCase):
    def setUp(self):
        # Initialize the FinancialTransformer with example parameters
        self.num_layers = 2
        self.d_model = 128
        self.num_heads = 8
        self.dff = 512
        self.pe_input = 1000  # Adjust as needed
        self.pe_target = 600  # Adjust as needed
        self.rate = 0.1

        self.transformer = Transformer(
            self.num_layers, 
            self.d_model, 
            self.num_heads, 
            self.dff, 
            self.pe_input, 
            self.pe_target, 
            self.rate
        )

    def test_output_shape(self):
        batch_size = 1
        input_seq_length = 16  # Example input sequence length
        target_seq_length = 16  # Example target sequence length
        sample_input = tf.random.uniform((batch_size, input_seq_length, self.d_model))
        sample_target = tf.random.uniform((batch_size, target_seq_length, self.d_model))

        result, _ = self.transformer(sample_input, sample_target, False, None, None, None)

        # Test if the output shape is as expected
        self.assertEqual(result.shape, (batch_size, target_seq_length, 1))  # Assuming output size of 1

    # Additional tests can be added here

if __name__ == "__main__":
    unittest.main()
