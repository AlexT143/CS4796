import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)

import tensorflow as tf
import numpy as np

import unittest

class TestScaledDotProductAttention(tf.test.TestCase):
    def test_output_shape(self):
        seq_len_q = 60
        seq_len_k = 75
        depth = 20

        q = tf.random.uniform((1, seq_len_q, depth))
        k = tf.random.uniform((1, seq_len_k, depth))
        v = tf.random.uniform((1, seq_len_k, depth))

        output, _ = scaled_dot_product_attention(q, k, v, None)
        self.assertAllEqual(output.shape, (1, seq_len_q, depth))

    def test_mask(self):
        q = tf.random.uniform((1, 3, 5))
        k = tf.random.uniform((1, 5, 5))
        v = tf.random.uniform((1, 5, 5))
        mask = tf.constant([[[0., -1e9, -1e9, -1e9, -1e9]]])

        _, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # The sum of attention weights over the keys should be close to 1 for each query position
        expected_sum = tf.ones_like(tf.reduce_sum(attention_weights, axis=-1))
        self.assertAllClose(tf.reduce_sum(attention_weights, axis=-1), expected_sum)



class TestMultiHeadAttention(tf.test.TestCase):
    def test_output_shape(self):
        d_model = 128
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        batch_size = 1
        seq_len = 50
        depth = d_model // num_heads

        q = tf.random.uniform((batch_size, seq_len, d_model))
        k = tf.random.uniform((batch_size, seq_len, d_model))
        v = tf.random.uniform((batch_size, seq_len, d_model))

        output, _ = mha(v, k, q, None)
        self.assertAllEqual(output.shape, (batch_size, seq_len, d_model))


if __name__ == "__main__":
    unittest.main()