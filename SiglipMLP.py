import tensorflow as tf

class SiglipMLP(tf.keras.layers.Layer):
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
        # Build layers if needed
        if not self.fc1.built:
            self.fc1.build(hidden_states.shape)
        if not self.fc2.built:
            self.fc2.build((None, self.config.intermediate_size))


        tf.print("MLP called - input[0,0,:3]:", hidden_states[0, 0, :3])
        tf.print("MLP fc1.kernel[0,:3]:", self.fc1.kernel[0, :3])
        tf.print("MLP fc1.bias[:3]:", self.fc1.bias[:3])

        x = tf.matmul(hidden_states, self.fc1.kernel) + self.fc1.bias
        tf.print("After fc1[0,0,:3]:", x[0, 0, :3])

        x = 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3))))
        tf.print("After GELU[0,0,:3]:", x[0, 0, :3])

        x = tf.matmul(x, self.fc2.kernel) + self.fc2.bias
        tf.print("After fc2[0,0,:3]:", x[0, 0, :3])

        return x