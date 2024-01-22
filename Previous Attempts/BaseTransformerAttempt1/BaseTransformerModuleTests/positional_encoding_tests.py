import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BaseTransformerModules.embedding import get_positional_encoding

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# This test is designed to ensure that the output of the get_positional_encoding function is correct in terms of its shape and type.
def test_shape_and_type():
    seq_length = 16
    d_model = 256
    positional_encoding = get_positional_encoding(seq_length, d_model)
    assert positional_encoding.shape == (seq_length, d_model), "Shape mismatch"
    assert isinstance(
        positional_encoding, tf.Tensor
    ), "Output is not a TensorFlow tensor"
    #print("Test for correct shape and type passed.")


test_shape_and_type()


def plot_positional_encoding(positional_encoding, seq_length, d_model):
    plt.figure(figsize=(10, 10))
    plt.imshow(positional_encoding.numpy())
    plt.colorbar()
    plt.title("Positional Encoding")
    plt.xlabel("Depth")
    plt.ylabel("Position")
    plt.show()


plot_positional_encoding(get_positional_encoding(16, 256), 16, 256)


def test_stability_with_sequence_lengths():
    for seq_length in [10, 20, 30]:
        positional_encoding = get_positional_encoding(seq_length, 256)
        assert positional_encoding.shape == (
            seq_length,
            256,
        ), f"Shape mismatch for seq_length {seq_length}"
    #print("Test for stability with different sequence lengths passed.")


test_stability_with_sequence_lengths()
