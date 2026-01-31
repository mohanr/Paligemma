import tensorflow as tf

from SiglipEncoder import SiglipEncoder
from SiglipEncoderLayer import SiglipEncoderLayer
from SiglipVisionEmbeddings import SiglipVisionEmbeddings

class SiglipVisionTransformer(tf.keras.Model):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=config.layer_norm_eps,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones'
        )
    def call(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        tf.print("After embeddings std:", tf.math.reduce_std(hidden_states))
        tf.print("SiglipVisionTransformer layer_norm1 epsilon:", self.config.layer_norm_eps)

        last_hidden_state = self.encoder(hidden_states)
        tf.print("After encoder (before post_norm) std:", tf.math.reduce_std(last_hidden_state))
        tf.print("post_layernorm epsilon:", self.post_layernorm.epsilon)
        tf.print("post_layernorm.trainable:", self.post_layernorm.trainable)
        if len(self.post_layernorm.weights) > 0:
            tf.print("post_layernorm weight 0 (gamma)[:5]:", self.post_layernorm.weights[0][:5])
        if len(self.post_layernorm.weights) > 1:
            tf.print("post_layernorm weight 1 (beta)[:5]:", self.post_layernorm.weights[1][:5])
        last_hidden_state = self.post_layernorm(last_hidden_state)
        tf.print("After post_norm std:", tf.math.reduce_std(last_hidden_state))

        return last_hidden_state