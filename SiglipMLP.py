import tensorflow as tf

class SiglipMLP(tf.keras.Model):
    def __init__(self, config):
        super(SiglipMLP,self).__init__()
        self.config = config
        self.fc1 = tf.keras.layers.Dense(config.intermediate_size,
                                         input_shape=(config.hidden_size,),
                                         activation=None, use_bias=False)

        self.fc2 = tf.keras.layers.Dense(config.hidden_size,
                                         input_shape=(config.intermediate_size,),
                                         activation=None, use_bias=False)

    def call(self, hidden_states):
        hidden_states = self.fc1( hidden_states )
        hidden_states = tf.nn.gelu(hidden_states, approximate=True)
        hidden_states = self.fc2( hidden_states )
        return hidden_states
