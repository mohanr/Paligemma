import tensorflow as tf

class KVCache:
    def __init__(self):
        self.key_cache = {}
        self.value_cache = {}
        self.sequence_len = 0   # global

    def update(self, key_states, value_states, layer_idx):
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = tf.Variable(
                tf.zeros_like(key_states), trainable=False)
            self.value_cache[layer_idx] = tf.Variable(
                tf.zeros_like(value_states), trainable=False)

        start = self.sequence_len
        end   = start + key_states.shape[2]

        self.key_cache[layer_idx][:, :, start:end, :].assign(key_states)
        self.value_cache[layer_idx][:, :, start:end, :].assign(value_states)

        self.sequence_len = end

    def num_items(self):
        return self.sequence_len
