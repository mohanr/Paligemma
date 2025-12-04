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
        tf.print("Shape of initial input_embeds:", tf.shape(input_embeds))

        # print(f' config.hidden_size {self.config.hidden_size}')
        # print(f' image_features {image_features}')

        final_embedding = tf.zeros((batch_size,sequence_length,embed_dim),dtype=input_embeds.dtype)
        text_mask = tf.not_equal(input_ids, self.config.image_token_index) & tf.not_equal(input_ids, self.pad_token_id)
        image_mask = tf.equal(input_ids, self.image_token_index)
        tf.print("Image Mask (first 10 tokens):", image_mask[0, 0:10])
        image_mask = tf.equal(input_ids, self.image_token_index)
        tf.print("Image Mask DType:", image_mask.dtype)
        tf.print("Image Mask Shape:", tf.shape(image_mask))
        indices = tf.where(image_mask)
        pad_mask = tf.equal(input_ids, self.pad_token_id)

        text_mask_expanded = tf.expand_dims(text_mask, axis=-1)
        image_mask_expanded = tf.expand_dims(image_mask, axis=-1)
        pad_mask_expanded = tf.expand_dims(pad_mask, axis=-1)

        final_embedding = tf.where(text_mask_expanded, input_embeds, final_embedding)
        tf.print("Input IDs (snippet):", input_ids[0, 0:10], summarize=10)  # See what tokens are being fed
        tf.print("Image Token Index:", self.image_token_index)
        tf.print("Image Mask Sum (V-tokens found):", tf.reduce_sum(tf.cast(image_mask, tf.int32)))

        indices = tf.where(image_mask)
        print("Update indices ", tf.size(indices))
        if (tf.size(indices) > 0):
            updates = tf.reshape(image_features, (-1,embed_dim))
            updates = tf.reshape(image_features, (-1, embed_dim))

            tf.print("--- Image Feature Statistics ---")
            tf.print("Mean:", tf.reduce_mean(image_features))
            tf.print("Min:", tf.reduce_min(image_features))
            tf.print("Max:", tf.reduce_max(image_features))

            print("Indices", updates)
            # ... rest of the code ...
            print("Indices",updates)
            image_scatter = tf.scatter_nd( indices, updates,(batch_size,sequence_length,embed_dim))
            final_embedding = tf.where(image_mask_expanded, image_scatter,final_embedding)
        final_embedding = tf.where(pad_mask_expanded, tf.zeros_like(final_embedding),final_embedding)
        q_len = tf.shape(input_embeds)[1]
        cache_len = kv_cache.num_items()

        neg_inf = tf.constant(-1e9, dtype=tf.float32)

        if cache_len == 0:
            mask_shape = (q_len, q_len)

            causal_mask_bool = tf.linalg.band_part(tf.ones(mask_shape, dtype=tf.bool), -1, 0)

            causal_mask_bool = tf.logical_not(causal_mask_bool)

            additive_mask = tf.where(causal_mask_bool, neg_inf, tf.zeros(mask_shape, dtype=tf.float32))

            causal_mask = tf.expand_dims(additive_mask, axis=0)  # [1, q_len, q_len]
            causal_mask = tf.expand_dims(causal_mask, axis=1)  # [1, 1, q_len, q_len]

        else:
            kv_len = tf.add(cache_len, q_len)

            causal_mask = tf.fill((batch_size, 1, q_len, kv_len), 0.0)
            causal_mask = tf.cast(causal_mask, tf.float32)

        causal_mask = tf.expand_dims(causal_mask, axis=1)
        sequence_length = tf.shape(final_embedding)[1]

        position_ids = tf.range(start=0, limit=sequence_length, dtype=tf.int32)

        position_ids = tf.expand_dims(position_ids, axis=0)
        if kv_cache.num_items() > 1 and tf.shape(input_ids)[1] == 1:
            position_offset = kv_cache.num_items()

            position_ids = tf.range(start=position_offset,
                                    limit=position_offset + 1,
                                    dtype=tf.int32)

            position_ids = tf.expand_dims(position_ids, axis=0)

        tf.print("Shape of attention_mask BEFORE return:", tf.shape(attention_mask))
        tf.print("Shape of input_embeds BEFORE return:", tf.shape(input_embeds))
        tf.print("Shape of position_ids BEFORE return:", tf.shape(position_ids))

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
        tf.print("Position IDs (snippet):", position_ids[0:10], summarize=10)
        tf.print("Position IDs (around Image Tokens):", position_ids[1020:1040], summarize=20)
        tf.print("Total Sequence Length:",tf.shape(position_ids)[0])
        if tf.rank(attention_mask) == 2:
            attention_mask = tf.expand_dims(attention_mask, axis=1)  # [1, 1, 1030]
            attention_mask = tf.expand_dims(attention_mask, axis=1)  # [1, 1, 1, 1030]
        tf.print("Input Embeds Mean:", tf.reduce_mean(input_embeds))
        tf.print("Input Embeds Std:", tf.math.reduce_std(input_embeds))
        outputs=self.language_model(
            attention_mask,
            position_ids,
            input_embeds,
            kv_cache=kv_cache
        )
        return outputs
