import tensorflow as tf
from GemmaRotaryEmbedding import GemmaRotaryEmbedding


class GemmaAttention(tf.keras.Model):
    def __init__(self,
                 config,
                 layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = tf.math.floordiv(self.num_heads, self.num_key_value_heads)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.is_causal = True

        self.q_proj = tf.keras.layers.Dense(self.hidden_size, activation=None, use_bias=config.attention_bias)
        self.k_proj = tf.keras.layers.Dense(self.num_key_value_heads * self.head_dim, activation=None,
                                            use_bias=config.attention_bias)
        self.v_proj = tf.keras.layers.Dense(self.num_key_value_heads * self.head_dim, activation=None,
                                            use_bias=config.attention_bias)
        self.o_proj = tf.keras.layers.Dense(self.hidden_size, activation=None, use_bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def rotate(self, x):
        # last dimension
        last_dim_size = tf.shape(x)[-1]

        # midpoint
        midpoint = last_dim_size // 2
        x1 = x[..., :midpoint]
        x2 = x[..., midpoint:]
        return tf.concat([-x2, x1], axis=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):

        if q is not None:
            q_embed = tf.add(tf.multiply(q, cos), tf.multiply(self.rotate(q), sin))
        else:
            q_embed = None

        if k is not None:
            k_embed = tf.add(tf.multiply(k, cos), tf.multiply(self.rotate(k), sin))
        else:
            k_embed = None

        return q_embed, k_embed

    def repeat_kv(self, x):
        # x shape: [batch, num_kv_heads, seq_len, head_dim]
        # Need to repeat each KV head num_key_value_groups times
        # PyTorch uses repeat_interleave, TF needs manual implementation

        batch, num_kv_heads, slen, head_dim = tf.unstack(tf.shape(x))

        if self.num_key_value_groups == 1:
            return x

        # Expand and reshape to interleave repetitions
        # [batch, num_kv_heads, 1, seq_len, head_dim]
        x = tf.expand_dims(x, axis=2)

        # [batch, num_kv_heads, num_key_value_groups, seq_len, head_dim]
        x = tf.tile(x, [1, 1, self.num_key_value_groups, 1, 1])

        # [batch, num_kv_heads * num_key_value_groups, seq_len, head_dim]
        x = tf.reshape(x, [batch, num_kv_heads * self.num_key_value_groups, slen, head_dim])

        return x

    def call(self, hidden_states, attention_mask, position_ids, kv_cache):
        self._cache_updated = False

        # if kv_cache:
        #     print(f"Cache has {kv_cache.num_items()} items")

        if tf.rank(attention_mask) == 5:
            attention_mask = tf.squeeze(attention_mask, axis=2)

        shape_list = tf.shape(hidden_states)
        bsz = shape_list[0]
        q_len = shape_list[1]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention heads
        query_states = tf.reshape(
            query_states, (bsz, q_len, self.num_heads, self.head_dim)
        )
        query_states = tf.transpose(query_states, (0, 2, 1, 3))  # [B, num_heads, T, head_dim]

        key_states = tf.reshape(
            key_states, (bsz, q_len, self.num_key_value_heads, self.head_dim)
        )
        key_states = tf.transpose(key_states, (0, 2, 1, 3))  # [B, num_kv_heads, T, head_dim]

        value_states = tf.reshape(
            value_states, (bsz, q_len, self.num_key_value_heads, self.head_dim)
        )
        value_states = tf.transpose(value_states, (0, 2, 1, 3))

        if tf.rank(position_ids) == 1:
            position_ids = tf.expand_dims(position_ids, axis=0)

        if tf.rank(position_ids) == 2 and tf.shape(position_ids)[0] != bsz:
            position_ids = tf.tile(position_ids, [bsz, 1])

        q_position_ids = position_ids[:, -q_len:]

        q_cos, q_sin = self.rotary_emb(hidden_states, q_position_ids, seq_len=q_len)

        query_states, _ = self.apply_rotary_pos_emb(query_states, None, q_cos, q_sin)

        if kv_cache is not None:
            # Apply rotary to NEW keys BEFORE caching
            k_position_ids = position_ids[:, -q_len:]
            k_cos, k_sin = self.rotary_emb(hidden_states, k_position_ids, seq_len=q_len)
            _, key_states = self.apply_rotary_pos_emb(None, key_states, k_cos, k_sin)

            # Cache the already-rotated keys
            kv_cache.update(key_states, value_states, self.layer_idx)

            # Retrieve full cache (already rotated!)
            key_states, value_states = kv_cache.get_cache(self.layer_idx)
        else:
            # No cache: apply rotary to current keys
            k_position_ids = position_ids[:, -q_len:]
            k_cos, k_sin = self.rotary_emb(hidden_states, k_position_ids, seq_len=q_len)
            _, key_states = self.apply_rotary_pos_emb(None, key_states, k_cos, k_sin)

        # --- GQA EXPANSION ---
        # Repeat key/value heads for GQA
        if self.num_key_value_groups > 1:
            key_states = self.repeat_kv(key_states)
            value_states = self.repeat_kv(value_states)

        query_states = query_states / tf.math.sqrt(tf.cast(self.head_dim, query_states.dtype))

        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        kv_len = tf.shape(key_states)[-2]

        mask_shape = tf.shape(attention_mask)
        expected_shape = (bsz, 1, q_len, kv_len)

        shapes_equal = tf.reduce_all(tf.equal(mask_shape, expected_shape))

        if not shapes_equal:

            # Get current mask dimensions
            mask_bsz = mask_shape[0]
            mask_heads = mask_shape[1]
            mask_q_len = mask_shape[2]
            mask_kv_len = mask_shape[3]

            # Reshape or pad the mask
            if mask_kv_len < kv_len:
                # Pad mask on the kv dimension
                pad_len = kv_len - mask_kv_len
                padding = tf.zeros([mask_bsz, mask_heads, mask_q_len, pad_len],
                                   dtype=attention_mask.dtype)
                attention_mask = tf.concat([attention_mask, padding], axis=-1)
            elif mask_kv_len > kv_len:
                # Truncate mask on the kv dimension
                attention_mask = attention_mask[..., :kv_len]

            # Also check q_len dimension
            if mask_q_len < q_len:
                # Pad mask on the q dimension
                pad_len = q_len - mask_q_len
                padding = tf.zeros([mask_bsz, mask_heads, pad_len, kv_len],
                                   dtype=attention_mask.dtype)
                attention_mask = tf.concat([attention_mask, padding], axis=-2)
            elif mask_q_len > q_len:
                # Truncate mask on the q dimension
                attention_mask = attention_mask[..., -q_len:, :]

        # `attention_mask` is expected to already be additive:
        # 0.0 for allowed positions and a large negative value for masked ones.
        attn_weights = attn_weights + tf.cast(attention_mask, attn_weights.dtype)

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, dtype=query_states.dtype)

        if self.attention_dropout > 0:
            attn_weights = tf.nn.dropout(attn_weights, rate=self.attention_dropout)

        attn_output = tf.matmul(attn_weights, value_states)

        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.num_heads * self.head_dim))
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
