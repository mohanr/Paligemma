import tensorflow as tf

class GemmaRotaryEmbedding(tf.keras.Model):
    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        half_dim = dim // 2
        freq_seq = tf.range(half_dim, dtype=tf.float32)

        inv_freq = 1.0 / (base ** (freq_seq / half_dim))

        self.inv_freq = tf.reshape(inv_freq, (1, 1, half_dim))

    def call(self, x, position_ids, seq_len=None, trainable=False):

        b = tf.shape(x)[0]
        l = tf.shape(x)[1]

        if len(position_ids.shape) == 1:
            position_ids = tf.expand_dims(position_ids, 0)

        position_ids = tf.cast(position_ids, tf.float32)

        inv = tf.squeeze(self.inv_freq, axis=0)

        freqs = tf.einsum("bl,ld->bld", position_ids, inv)
        freqs = tf.concat([freqs, freqs], axis=-1)
        cos = tf.cos(freqs)[:, None, :, :]   # (b, 1, l, half_dim)
        sin = tf.sin(freqs)[:, None, :, :]   # (b, 1, l, half_dim)

        return cos, sin
