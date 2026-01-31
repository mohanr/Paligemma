from SiglipVisionConfig import SiglipVisionConfig
from SiglipVisionModel import SiglipVisionModel
from PaliGemmaMultiModalProjector import PaliGemmaMultiModalProjector
from GemmaForCausalLM import  GemmaForCausalLM
import numpy as np
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

        # Create masks
        text_mask = tf.not_equal(input_ids, self.config.image_token_index) & tf.not_equal(input_ids, self.pad_token_id)
        image_mask = tf.equal(input_ids, self.config.image_token_index)
        pad_mask = tf.equal(input_ids, self.pad_token_id)

        tf.print("=== MERGE MASKS DEBUG ===")
        tf.print("input_ids shape:", tf.shape(input_ids))
        tf.print("input_ids first 10:", input_ids[0, :10])
        tf.print("image_token_index:", self.config.image_token_index)
        tf.print("Number of image tokens in input_ids:", tf.reduce_sum(tf.cast(image_mask, tf.int32)))
        tf.print("Number of text tokens:", tf.reduce_sum(tf.cast(text_mask, tf.int32)))
        tf.print("Number of image features:", tf.shape(image_features)[1])


        # Expand for broadcasting
        text_mask_expanded = tf.expand_dims(text_mask, axis=-1)
        image_mask_expanded = tf.expand_dims(image_mask, axis=-1)
        pad_mask_expanded = tf.expand_dims(pad_mask, axis=-1)

        # Start with text embeddings
        final_embedding = input_embeds

        indices = tf.where(image_mask)
        if tf.size(indices) > 0:
            # ADD THESE DEBUG LINES:
            tf.print("=== MERGE DIAGNOSTIC ===")
            tf.print("Number of image token positions:", tf.shape(indices)[0])
            tf.print("Number of image features:", tf.shape(image_features)[0] * tf.shape(image_features)[1])
            tf.print("First image feature being inserted:", image_features[0, 0, :3])
            tf.print("Last image feature being inserted:", image_features[0, -1, :3])
            tf.print("First index where image will be placed:", indices[0])
            tf.print("Last index where image will be placed:", indices[-1])

        if tf.size(indices) > 0:
            updates = tf.reshape(image_features, (-1, embed_dim))
            image_scatter = tf.scatter_nd(indices, updates, (batch_size, sequence_length, embed_dim))
            final_embedding = tf.where(image_mask_expanded, image_scatter, final_embedding)

        # Zero out padding
        final_embedding = tf.where(pad_mask_expanded, tf.zeros_like(final_embedding), final_embedding)

        # Create attention mask
        q_len = tf.shape(final_embedding)[1]
        cache_len = kv_cache.num_items()

        if cache_len == 0:
            causal_mask = tf.zeros((1, 1, q_len, q_len), dtype=tf.float32)
            # Prefill: standard causal mask
            # causal_mask = tf.linalg.band_part(tf.ones((q_len, q_len), dtype=tf.float32), -1, 0)
            # Make it additive: 0 for allowed, -inf for masked
            # causal_mask = (1.0 - causal_mask) * -1e9
            # causal_mask = tf.expand_dims(causal_mask, axis=0)
            # causal_mask = tf.expand_dims(causal_mask, axis=0)
        else:
            # Generation: new token attends to all previous
            kv_len = cache_len + q_len
            causal_mask = tf.zeros((batch_size, 1, q_len, kv_len), dtype=tf.float32)

        if cache_len == 0:
            # Prefill phase - PaliGemma uses 1-indexed positions!
            position_ids = tf.range(start=1, limit=q_len + 1, dtype=tf.int32)  # ← Start from 1!
            position_ids = tf.expand_dims(position_ids, axis=0)
        else:
            # Generation phase - position is cache length + 1
            position_ids = tf.fill((batch_size, q_len), cache_len + 1)  # ← Add +1!
            position_ids = tf.cast(position_ids, tf.int32)
        tf.print(
            "FIRST 10 POS IDS:", position_ids[0, :10]
        )
        tf.print(
            "LAST IMAGE POS IDS:", position_ids[0, 1014:1024]
        )
        tf.print(
            "FIRST TEXT POS IDS:", position_ids[0, 1024:1034]
        )

        tf.print("=== MERGE DEBUG ===")
        tf.print("Text embedding[0,0,:3]:", input_embeds[0, 0, :3])
        tf.print("Image feature[0,0,:3]:", image_features[0, 0, :3])
        tf.print("Final embedding[0,0,:3]:", final_embedding[0, 0, :3])
        tf.print("Are image features present?",
                 not tf.reduce_all(tf.equal(final_embedding[0, 0, :], input_embeds[0, 0, :])))
        return final_embedding, causal_mask, position_ids, kv_cache

    def call(self,
             input_ids,
             attention_mask,
             pixel_values,
             kv_cache):
        input_embeddding_layer = self.language_model.get_input_embeddings()
        input_embeds = input_embeddding_layer(input_ids)
        if pixel_values is not None:
            selected_image_features = self.vision_tower(tf.cast(pixel_values, input_embeds.dtype))
            tf.print(
                "VISION TOKENS:", tf.shape(selected_image_features)[1]
            )

            tf.print(
                "IMAGE TOKENS IN INPUT_IDS:",
                tf.reduce_sum(
                    tf.cast(input_ids == self.config.image_token_index, tf.int32)
                )
            )

            import numpy as np
            np.save(
                "tf_vision_output.npy",
                selected_image_features.numpy()
            )

            tf.print("=== VISION TOWER OUTPUT ===")
            tf.print("Vision features shape:", tf.shape(selected_image_features))
            tf.print("Vision features mean:", tf.reduce_mean(selected_image_features))
            tf.print("Vision features std:", tf.math.reduce_std(selected_image_features))
            tf.print("Vision features[0,0,:5]:", selected_image_features[0, 0, :5])
            print(f"!!! DEBUG BEFORE PROJECTOR: vision features std = {tf.math.reduce_std(selected_image_features)}")
            image_features = self.multi_modal_projector(selected_image_features)
            import numpy as np
            np.save('/Users/anu/tf_projector_output.npy', image_features.numpy())
            print(f"!!! DEBUG AFTER PROJECTOR: projected features std = {tf.math.reduce_std(image_features)}")
            tf.print("=== PROJECTOR OUTPUT ===")
            tf.print("Projected features shape:", tf.shape(image_features))
            tf.print("Projected features mean:", tf.reduce_mean(image_features))
            tf.print("Projected features std:", tf.math.reduce_std(image_features))
            tf.print("Projected features[0,0,:5]:", image_features[0, 0, :5])
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

            # Create proper 4D causal mask
            if cache_len == 0:
                # Prefill: [batch=1, heads=1, q_len, q_len]
                attention_mask = tf.fill((1, 1, q_len, q_len), 0.0)
            else:
                # Generation: [batch=1, heads=1, q_len=1, kv_len]
                kv_len = cache_len + q_len
                attention_mask = tf.fill((1, 1, q_len, kv_len), 0.0)

            # Position IDs
            if cache_len > 0:
                position_ids = tf.reshape(cache_len, (1, 1))
                position_ids = tf.cast(position_ids, tf.int32)
            else:
                position_ids = tf.range(q_len, dtype=tf.int32)[None, :]
        tf.print("Position IDs (first 10 image tokens):", position_ids[0, :10])
        tf.print("Position IDs (last image tokens + first text):", position_ids[0, 1020:1030])
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
        print(f"!!! DEBUG POST-CONTEXT: KVCache sequence_len in model instance: {kv_cache.sequence_len}")
        # <<<<<<<<<<<<<<<< INSERT HERE >>>>>>>>>>>>>>>>>
        return outputs