import tensorflow as tf


class KVCache:
    def __init__(self, max_seq_len=4096):
        self.key_cache = {}
        self.value_cache = {}
        self.sequence_len = 0
        self.max_seq_len = max_seq_len
        print(f"KVCache initialized with max_seq_len={max_seq_len}")

    def reset(self):
        self.sequence_len = 0
        self.key_cache = {}
        self.value_cache = {}

    def update(self, key_states, value_states, layer_idx):
        seq_len = tf.shape(key_states)[2]


        # Initialize cache if needed
        if layer_idx not in self.key_cache:
            batch_size = tf.shape(key_states)[0]
            num_heads = tf.shape(key_states)[1]
            head_dim = tf.shape(key_states)[3]

            cache_shape = (batch_size, num_heads, self.max_seq_len, head_dim)
            self.key_cache[layer_idx] = tf.Variable(
                tf.zeros(cache_shape, dtype=key_states.dtype),
                trainable=False
            )
            self.value_cache[layer_idx] = tf.Variable(
                tf.zeros(cache_shape, dtype=value_states.dtype),
                trainable=False
            )
        if seq_len == 1:
            pos = self.sequence_len
            if pos < self.max_seq_len:
                self.key_cache[layer_idx][:, :, pos:pos + 1, :].assign(key_states)
                self.value_cache[layer_idx][:, :, pos:pos + 1, :].assign(value_states)

                if layer_idx == 0:
                    self.sequence_len += 1
            else:
                print(f" Cache full at position {pos}")

        else:
            # Prefill path: each layer must write its own full prefix cache.
            if seq_len <= self.max_seq_len:
                self.key_cache[layer_idx][:, :, :seq_len, :].assign(key_states)
                self.value_cache[layer_idx][:, :, :seq_len, :].assign(value_states)
                # Track logical cache length once (layer 0), shared across layers.
                if layer_idx == 0:
                    self.sequence_len = seq_len
            else:
                print(f" Context sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
    def get_cache(self, layer_idx):
        if layer_idx in self.key_cache and self.sequence_len > 0:
            return (
                self.key_cache[layer_idx][:, :, :self.sequence_len, :],
                self.value_cache[layer_idx][:, :, :self.sequence_len, :]
            )
        return None, None

    def num_items(self):
        return self.sequence_len
