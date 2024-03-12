import tensorflow as tf

class ArimaInspiredMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ar_order=1, ma_order=1, use_ar=True, use_ma=True):
        super(ArimaInspiredMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.use_ar = use_ar
        self.use_ma = use_ma
        
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

        if self.use_ar:
            self.ar_weights = self.add_weight(shape=(self.ar_order,), initializer='random_normal', trainable=True)
            self.alpha = self.add_weight(shape=(), initializer='zeros', trainable=True)

        if self.use_ma:
            self.ma_weights = self.add_weight(shape=(self.ma_order,), initializer='random_normal', trainable=True)
            self.beta = self.add_weight(shape=(), initializer='zeros', trainable=True)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def compute_ar_term(self, x):
        ar_terms = []
        for i in range(1, self.ar_order + 1):
            ar_term = tf.reduce_sum(x[:, :, :-i, :] * self.ar_weights[i-1], axis=-1, keepdims=True)
            ar_terms.append(ar_term)
        ar_terms = tf.concat(ar_terms, axis=-1)
        return ar_terms

    def compute_ma_term(self, x):
        ma_terms = []
        for i in range(1, self.ma_order + 1):
            ma_term = tf.reduce_sum(x[:, :, i:, :] * self.ma_weights[i-1], axis=-1, keepdims=True)
            ma_terms.append(ma_term)
        ma_terms = tf.concat(ma_terms, axis=-1)
        return ma_terms

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        modified_attention_logits = scaled_attention_logits

        if self.use_ar:
            ar_term = self.compute_ar_term(v)  # (batch_size, num_heads, seq_len_v, ar_order)
            modified_attention_logits += self.alpha * ar_term

        if self.use_ma:
            ma_term = self.compute_ma_term(v)  # (batch_size, num_heads, seq_len_v, ma_order)
            modified_attention_logits += self.beta * ma_term

        attention_weights = tf.nn.softmax(modified_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights