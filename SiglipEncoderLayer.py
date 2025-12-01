import tensorflow as tf
from SiglipMLP import  SiglipMLP
from SiglipAttention import  SiglipAttention

class SiglipEncoderLayer(tf.keras.Model):

    def __init__(self, config):
        super(SiglipEncoderLayer,self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(axis=-1,epsilon=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=config.layer_norm_eps)

    def call(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states