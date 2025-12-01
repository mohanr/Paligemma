import tensorflow as tf
from SiglipEncoderLayer import SiglipEncoderLayer

class SiglipEncoder(tf.keras.Model):

    def __init__(self, config):
        super(SiglipEncoder,self).__init__()
        self.config = config

        self.layers = [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]


    def call(self, inputs_embeds):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers :
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

