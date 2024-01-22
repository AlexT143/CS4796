import tensorflow as tf
import numpy as np

def get_positional_encoding(seq_length, d_model):
    pos_enc = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

    pos_enc_tensor = tf.constant(pos_enc, dtype=tf.float32)

    # Print the shape of the positional encoding
    #print("Positional Encoding Shape:", pos_enc_tensor.shape)

    # Optionally, print some sample values from the positional encoding matrix
    #print("Sample values from Positional Encoding:\n", pos_enc_tensor[:5, :5])  # Prints first 5 positions and first 5 dimensions

    return pos_enc_tensor
