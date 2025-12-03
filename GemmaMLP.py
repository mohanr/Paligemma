import tensorflow as tf

class GemmaMLP(tf.keras.Model):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.gate_proj = tf.keras.layers.Dense(self.intermediate_size, use_bias=False)
        self.up_proj = tf.keras.layers.Dense(self.intermediate_size, use_bias=False)
        self.down_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=False)
        self.intermediate_size = config.intermediate_size


    def call(self,x):
        g = tf.nn.silu(self.gate_proj(x))  # correct activation
        u = self.up_proj(x)
        intermediate_product = g * u

        CLIPPING_VALUE = 65504.0  # Maximum finite value for float16
        intermediate_product = tf.clip_by_value(
            intermediate_product,
            clip_value_min=-CLIPPING_VALUE,
            clip_value_max=CLIPPING_VALUE
        )

        return self.down_proj(intermediate_product)

