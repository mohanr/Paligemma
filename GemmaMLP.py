import tensorflow as tf

class GemmaMLP(tf.keras.Model):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size // 2
        self.hidden_size = config.hidden_size
        self.gate_proj = tf.keras.layers.Dense(self.intermediate_size, use_bias=False)
        self.up_proj = tf.keras.layers.Dense(self.intermediate_size, use_bias=False)
        self.down_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=False)

    def call(self,x):
        g = tf.nn.silu(self.gate_proj(x))  # correct activation
        u = self.up_proj(x)
        return self.down_proj(g * u)
        # return self.down_proj(tf.nn.gelu(self.gate_proj(x),approximate=True) * self.up_proj(x))