import tensorflow as tf

from GemmaModel import GemmaModel


class GemmaForCausalLM(tf.keras.Model):
    def __init__(self,
                 config):
        super().__init__(
        )
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = tf.keras.layers.Dense(
                                            units=config.vocab_size,
                                            activation=None, use_bias=False)
    def get_input_embeddings(self):
        return self.model.embed_tokens


    def tie_weights(self):
        lm_head_kernel = self.lm_head.kernel
        embedding_tensor = self.model.embed_tokens.embeddings
        lm_head_kernel.assign(tf.transpose( embedding_tensor))

    def call(self,
             attention_mask,
             position_ids,
             inputs_embeds,
             kv_cache):
        outputs = self.model(attention_mask,
                             position_ids,
                             inputs_embeds,
                             kv_cache=kv_cache)
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = tf.cast(logits,tf.float32)
        return_data = { "logits" : logits}
        if kv_cache is not None :
            return_data["kv_cache"] = kv_cache
        return return_data