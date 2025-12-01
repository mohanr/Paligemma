import tensorflow as tf

from GemmaAttention import GemmaAttention
from GemmaMLP import GemmaMLP
from GemmaRMSNorm import GemmaRMSNorm

class GemmaDecoderLayer(tf.keras.Model):
    def __init__(self,
                 config,
                 layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config,layer_idx)
        self.mlp = GemmaMLP(config)

        self.input_layernorm = GemmaRMSNorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size,config.rms_norm_eps)

    def call(self,
             hidden_states,
             attention_mask,
             position_ids,
             kv_cache):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states,_ = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states