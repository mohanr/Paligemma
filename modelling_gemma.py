from SiglipVisionConfig import SiglipVisionConfig
from SiglipVisionModel import SiglipVisionModel
from PaliGemmaMultiModalProjector import PaliGemmaMultiModalProjector
from GemmaForCausalLM import  GemmaForCausalLM

import tensorflow as tf

class PaliGemmaForConditionalGeneration(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.image_token_index = config.image_token_index
        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
            self,
            image_features,
            input_embeds,
            input_ids,
            attention_mask,
            kv_cache
        ):
        batch_size = tf.shape(input_ids)[0]
        sequence_length = tf.shape(input_ids)[1]
        embed_dim = tf.shape(image_features)[-1]

        print(f' config.hidden_size {self.config.hidden_size}')
        print(f' image_features {image_features}')

        scaled_image_features = tf.divide( image_features, tf.sqrt(tf.cast(self.config.hidden_size,tf.float32)))
        #Combine all the image tokens, text tokens and mask all the padding tokens
        final_embedding = tf.zeros((batch_size,sequence_length,embed_dim),dtype=input_embeds.dtype)
        text_mask = tf.not_equal(input_ids, self.config.image_token_index) & tf.not_equal(input_ids, self.pad_token_id)
        image_mask = tf.equal(input_ids, self.image_token_index)
        pad_mask = tf.equal(input_ids, self.pad_token_id)

        text_mask_expanded = tf.expand_dims(text_mask, axis=-1)
        image_mask_expanded = tf.expand_dims(image_mask, axis=-1)
        pad_mask_expanded = tf.expand_dims(pad_mask, axis=-1)

        final_embedding = tf.where(text_mask_expanded, input_embeds, final_embedding)

        indices = tf.where(image_mask)
        if (tf.size(indices) > 0):
            updates = tf.reshape(scaled_image_features, (-1,embed_dim))
            image_scatter = tf.scatter_nd( indices, updates,(batch_size,sequence_length,embed_dim))
            final_embedding = tf.where(image_mask_expanded, image_scatter,final_embedding)
        final_embedding = tf.where(pad_mask_expanded, tf.zeros_like(final_embedding),final_embedding)
        q_len = tf.shape(input_embeds)[1]
        cache_len = kv_cache.num_items()
        print("size:", cache_len)
        causal_mask = tf.cond(cache_len == 0,
            lambda : tf.fill( (batch_size, q_len, q_len), 0),
            lambda : tf.fill( (batch_size, q_len, tf.add(cache_len, q_len)), 0)
        )
        causal_mask = tf.expand_dims(causal_mask, axis=1)
        print("attention_mask:", attention_mask)
        position_ids = tf.math.cumsum(attention_mask, axis=-1)[:, -1]
        print("position_ids:", tf.shape(position_ids))
        if (kv_cache.num_items() > 1):
            shape_list = tf.shape(position_ids)
            if len(shape_list) == 1:
                position_ids = tf.expand_dims(position_ids, axis=0)
            else:
                cumsum = tf.math.cumsum(attention_mask, axis=-1)
                mask = tf.equal(attention_mask , 0)
                position_ids = tf.where(mask, tf.ones_like(cumsum), cumsum)
        return final_embedding, causal_mask, position_ids,kv_cache

    def call(self,
             input_ids,
             attention_mask,
             pixel_values,
             kv_cache):
        # tf.print("input_ids.shape:", tf.shape(input_ids))
        # tf.print("attention_mask.shape:", tf.shape(attention_mask))
        # tf.print("pixel_values.shape :", tf.shape(pixel_values))
        input_embeddding_layer = self.language_model.get_input_embeddings()
        input_embeds = input_embeddding_layer(input_ids)
        if pixel_values is not None:
            selected_image_features = self.vision_tower(tf.cast(pixel_values,input_embeds.dtype))
            image_features = self.multi_modal_projector(selected_image_features)

            image_features = tf.reduce_mean(image_features, axis=1, keepdims=True)

            embed_dim = tf.shape(image_features)[-1]
            input_dim = tf.shape(input_embeds)[-1]
            # print("hidden_size:", self.config.hidden_size, type(self.config.hidden_size))
            # print("vision_dim:", self.config.vision_config.hidden_size, type(self.config.vision_config.hidden_size))
            # print("text_dim:", self.config.text_config.hidden_size, type(self.config.text_config.hidden_size))

            if embed_dim != input_dim:
                projector = tf.keras.layers.Dense(
                    self.config.hidden_size,
                    use_bias=False,
                    name="image_feature_projection"
                )
                image_features = projector(image_features)
            input_embeds,attention_mask,position_ids,kv_cache=self._merge_input_ids_with_image_features(
                image_features,
                input_embeds,
                input_ids,
                attention_mask,
                kv_cache
            )
        else:
            q_len = tf.shape(input_embeds)[1]
            cache_len = kv_cache.num_items()
            causal_mask = tf.cond(
                cache_len == 0,
                lambda: tf.fill((1, q_len, q_len), 0),
                lambda: tf.fill((1, q_len, tf.add(cache_len, q_len)), 0)
            )
            causal_mask = tf.expand_dims(causal_mask, axis=1)
            position_ids = tf.math.cumsum(attention_mask, axis=-1)[:, -1]

        outputs=self.language_model(
            attention_mask,
            position_ids,
            input_embeds,
            kv_cache=kv_cache
        )
        return outputs
