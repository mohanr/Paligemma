import tensorflow as tf

class PaliGemmaMultiModalProjector(tf.keras.layers.Layer):
    def __init__(self,config):
        self.config = config
        super(PaliGemmaMultiModalProjector,self).__init__(
        )

        self.linear = tf.keras.layers.Dense(config.vision_config.hidden_size,
                                            input_shape=config.vision_config.projection_dim,
                                            activation=None, use_bias=True)


    def call(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states