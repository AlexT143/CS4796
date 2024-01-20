import tensorflow as tf
import numpy as np

def create_embedding_layer(num_tokens, d_model, input_sequence_length):
    # num_tokens: Number of different tokens/numbers in the numerical sequence
    # d_model: The dimension of the embedding vectors
    # input_sequence_length: Length of input sequences
    return tf.keras.layers.Embedding(input_dim=num_tokens, 
                                     output_dim=d_model, 
                                     input_length=input_sequence_length)

def get_positional_encoding(d_model, seq_length):
    pos_enc = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    return tf.constant(pos_enc, dtype=tf.float32)

# Example usage
seq_length = 50 # Example sequence length
d_model = 512 # Example dimension of embedding vector
positional_encoding = get_positional_encoding(d_model, seq_length)