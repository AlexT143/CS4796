import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask):
    print("scaled_dot_product_attention - q shape:", tf.shape(q))
    print("scaled_dot_product_attention - k shape:", tf.shape(k))
    print("scaled_dot_product_attention - v shape:", tf.shape(v))

    matmul_qk = tf.matmul(q, k, transpose_b=True)
    print("scaled_dot_product_attention - matmul_qk shape:", tf.shape(matmul_qk))

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    print("scaled_dot_product_attention - attention_weights shape:", tf.shape(attention_weights))

    output = tf.matmul(attention_weights, v)
    print("scaled_dot_product_attention - output shape:", tf.shape(output))

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        print("MultiHeadAttention - d_model:", d_model)
        print("MultiHeadAttention - num_heads:", num_heads)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        print("MultiHeadAttention - q shape after wq:", tf.shape(q))
        print("MultiHeadAttention - k shape after wk:", tf.shape(k))
        print("MultiHeadAttention - v shape after wv:", tf.shape(v))

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        print("MultiHeadAttention - Split q shape:", tf.shape(q))
        print("MultiHeadAttention - Split k shape:", tf.shape(k))
        print("MultiHeadAttention - Split v shape:", tf.shape(v))

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        print("MultiHeadAttention - scaled_attention shape:", tf.shape(scaled_attention))

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        print("MultiHeadAttention - output shape:", tf.shape(output))

        return output, attention_weights