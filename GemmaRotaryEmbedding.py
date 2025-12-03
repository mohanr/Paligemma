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

        self.inv_freq = tf.reshape(inv_freq, (1, half_dim))

    def call(self, x, position_ids, seq_len=None, trainable=False):
        if len(position_ids.shape) == 1:
            position_ids = tf.expand_dims(position_ids, 0)

        position_ids = tf.cast(position_ids, tf.float32)  # (B, L)

        inv = self.inv_freq

        freqs = tf.einsum("bl,hd->bld", position_ids, inv)

        freqs = tf.concat([freqs, freqs], axis=-1)

        cos = tf.expand_dims(tf.cos(freqs), axis=1)
        sin = tf.expand_dims(tf.sin(freqs), axis=1)


        return cos, sin