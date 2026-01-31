import tensorflow as tf

class SiglipMLP(tf.keras.Model):
    def __init__(self, config):
        super(SiglipMLP,self).__init__()
        self.config = config
        self.fc1 = tf.keras.layers.Dense(config.intermediate_size,
                                         input_shape=(config.hidden_size,),
                                         activation=None, use_bias=True)

        self.fc2 = tf.keras.layers.Dense(config.hidden_size,
                                         input_shape=(config.intermediate_size,),
                                         activation=None, use_bias=True)

    def call(self, hidden_states):
        hidden_states = self.fc1( hidden_states )
        # Use gelu_pytorch_tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        hidden_states = 0.5 * hidden_states * (1.0 + tf.tanh(
            tf.sqrt(2.0 / 3.141592653589793) * (hidden_states + 0.044715 * tf.pow(hidden_states, 3))
        ))
        hidden_states = self.fc2( hidden_states )
        return hidden_states