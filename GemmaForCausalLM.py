import tensorflow as tf
import numpy as np

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
        # Manual logit computation for debugging
        last_hidden = hidden_states[:, -1, :]  # [1, 2048]
        if not self.lm_head.built:
            self.lm_head.build((None, self.config.hidden_size))
        last_hidden = hidden_states[:, -1, :].numpy()
        np.save('/Users/anu/PycharmProjects/Siglip/tf_last_hidden.npy', last_hidden[0])

        tf.print("Hidden states shape:", tf.shape(hidden_states))
        tf.print("Hidden states mean:", tf.reduce_mean(hidden_states))
        tf.print("Hidden states std:", tf.math.reduce_std(hidden_states))
        tf.print("Hidden states[-1, -1, :5]:", hidden_states[-1, -1, :5])
        tf.print("LM head kernel shape:", tf.shape(self.lm_head.kernel))
        tf.print("LM head kernel mean:", tf.reduce_mean(self.lm_head.kernel))
        logits = self.lm_head(hidden_states)
        logits = tf.cast(logits,tf.float32)
        return_data = { "logits" : logits}
        if kv_cache is not None :
            return_data["kv_cache"] = kv_cache
        return return_data