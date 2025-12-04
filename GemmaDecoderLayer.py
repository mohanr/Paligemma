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
        self.layer_idx=layer_idx

        self.input_layernorm = GemmaRMSNorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size,config.rms_norm_eps)

    def call(self,
             hidden_states,
             attention_mask,
             position_ids,
             kv_cache):
        residual_attn = hidden_states

        normed_hidden_states = self.input_layernorm(hidden_states)

        attn_output, _ = self.self_attn(
                                   normed_hidden_states,
                                    attention_mask,
                                    position_ids,
                                    kv_cache=kv_cache
                                )
        print(f"L{self.layer_idx} Norm StdDev: {tf.math.reduce_std(normed_hidden_states):.6f}")
        hidden_states = residual_attn + attn_output * (1.0 / tf.math.sqrt(tf.cast(18, tf.float32)))
        print(f"L{self.layer_idx} Attn+Res StdDev: {tf.math.reduce_std(hidden_states):.6f}")


        residual_mlp = hidden_states

        normed_hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = residual_mlp + mlp_output * (1.0 / tf.math.sqrt(tf.cast(18, tf.float32)))

        return hidden_states