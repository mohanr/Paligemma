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
        self.post_layernorm = tf.keras.layers.LayerNormalization(axis=-1,epsilon=config.layer_norm_eps)

    def call(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state