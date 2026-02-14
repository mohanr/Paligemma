"""PaliGemma TF model aligned to Hugging Face inference behavior."""

from SiglipVisionConfig import SiglipVisionConfig
from SiglipVisionModel import SiglipVisionModel
from PaliGemmaMultiModalProjector import PaliGemmaMultiModalProjector
from GemmaForCausalLM import GemmaForCausalLM
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

    def _build_position_ids(self, batch_size, q_len, cache_len):
        # HF Paligemma uses 1-indexed position ids.
        start = tf.cast(cache_len + 1, tf.int32)
        stop = tf.cast(cache_len + q_len + 1, tf.int32)
        base = tf.range(start=start, limit=stop, dtype=tf.int32)
        return tf.tile(base[None, :], [batch_size, 1])

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

        # Expand for broadcasting
        text_mask_expanded = tf.expand_dims(text_mask, axis=-1)
        image_mask_expanded = tf.expand_dims(image_mask, axis=-1)
        pad_mask_expanded = tf.expand_dims(pad_mask, axis=-1)

        # Start with text embeddings
        final_embedding = input_embeds

        # Insert image features where image tokens are
        indices = tf.where(image_mask)
        if tf.size(indices) > 0:
            updates = tf.reshape(image_features, (-1, embed_dim))
            image_scatter = tf.scatter_nd(indices, updates, (batch_size, sequence_length, embed_dim))
            final_embedding = tf.where(image_mask_expanded, image_scatter, final_embedding)

        # Zero out padding
        final_embedding = tf.where(pad_mask_expanded, tf.zeros_like(final_embedding), final_embedding)
        q_len = tf.shape(final_embedding)[1]
        cache_len = kv_cache.num_items()
        kv_len = cache_len + q_len

        # HF inference prefill is prefix-bidirectional; use all-zeros additive mask.
        causal_mask = tf.zeros((batch_size, 1, q_len, kv_len), dtype=tf.float32)
        position_ids = self._build_position_ids(batch_size, q_len, cache_len)

        return final_embedding, causal_mask, position_ids, kv_cache

    def call(self,
             input_ids,
             attention_mask,
             pixel_values,
             kv_cache):
        input_embeddding_layer = self.language_model.get_input_embeddings()
        input_embeds = input_embeddding_layer(input_ids)

        if pixel_values is not None:
            # Process image
            selected_image_features = self.vision_tower(tf.cast(pixel_values, input_embeds.dtype))
            image_features = self.multi_modal_projector(selected_image_features)
            image_features = image_features / tf.math.sqrt(
                tf.cast(self.config.text_config.hidden_size, image_features.dtype)
            )

            # Merge with text embeddings and create proper attention mask
            input_embeds, attention_mask, position_ids, kv_cache = self._merge_input_ids_with_image_features(
                image_features,
                input_embeds,
                input_ids,
                attention_mask,
                kv_cache
            )
        else:
            # Text-only path
            q_len = tf.shape(input_embeds)[1]
            cache_len = kv_cache.num_items()
            batch_size = tf.shape(input_embeds)[0]

            kv_len = cache_len + q_len
            attention_mask = tf.zeros((batch_size, 1, q_len, kv_len), dtype=tf.float32)
            position_ids = self._build_position_ids(batch_size, q_len, cache_len)

        # Run through language model
        outputs = self.language_model(
            attention_mask,
            position_ids,
            input_embeds,
            kv_cache=kv_cache
        )

        return outputs
