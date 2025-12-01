import tensorflow as tf
from keras.layers import Embedding
from GemmaRMSNorm import  GemmaRMSNorm
from GemmaDecoderLayer import  GemmaDecoderLayer

class GemmaModel(tf.keras.Model):
    def __init__(self,
                 config):
        super().__init__(
            )
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Embedding(config.vocab_size,config.hidden_size,self.padding_idx)

        self.net = tf.keras.Sequential(
            layers=[
                GemmaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def call(self,
             attention_mask,
             position_ids,
             inputs_embeds,
             kv_cache):
        hidden_states = inputs_embeds
        normalizer = tf.constant( self.config.hidden_size ** 0.5)
        hidden_states = hidden_states * normalizer
        for decoder_layer in self.net.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states