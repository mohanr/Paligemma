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

                # --- CRITICAL FIX: Only increment the counter on the first layer ---
                if layer_idx == 0:
                    self.sequence_len += 1
                    # --- END CRITICAL FIX ---
            else:
                print(f" Cache full at position {pos}")
        # if seq_len == 1:
        #     pos = self.sequence_len
        #     if pos < self.max_seq_len:
        #         self.key_cache[layer_idx][:, :, pos:pos + 1, :].assign(key_states)
        #         self.value_cache[layer_idx][:, :, pos:pos + 1, :].assign(value_states)
        #         self.sequence_len += 1
        #     else:
        #         print(f" Cache full at position {pos}")
        # else:
        #
        #     for i in range(seq_len):
        #         pos = self.sequence_len + i
        #         if pos < self.max_seq_len:
        #             self.key_cache[layer_idx][:, :, pos:pos + 1, :].assign(
        #                 key_states[:, :, i:i + 1, :]
        #             )
        #             self.value_cache[layer_idx][:, :, pos:pos + 1, :].assign(
        #                 value_states[:, :, i:i + 1, :]
        #             )
        #     self.sequence_len += seq_len

        else:

            # --- CRITICAL FIX FOR CONTEXT CACHING ---

            # Only perform the initial cache write if the cache is completely empty (sequence_len is 0)

            if self.sequence_len == 0:

                # Vectorized update to write the 1030 tokens (faster and less prone to looping errors)

                if seq_len <= self.max_seq_len:

                    self.key_cache[layer_idx][:, :, :seq_len, :].assign(key_states)

                    self.value_cache[layer_idx][:, :, :seq_len, :].assign(value_states)

                    # IMPORTANT: REMOVE THE FAULTY 'num_layers' CHECK!

                    # If this block executes, it is the first time. We set the length

                    # immediately to prevent other layers from entering this block.

                    # --- FIX 1: SET LENGTH IMMEDIATELY ---

                    self.sequence_len = seq_len

                    # --- END FIX 1 ---


                else:

                    print(f" Context sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

            # The 'else' block for subsequent layers remains empty, which is correct (they only read the cache)
    def get_cache(self, layer_idx):
        if layer_idx in self.key_cache and self.sequence_len > 0:
            return (
                self.key_cache[layer_idx][:, :, :self.sequence_len, :],
                self.value_cache[layer_idx][:, :, :self.sequence_len, :]
            )
        return None, None

    def num_items(self):
        return self.sequence_len