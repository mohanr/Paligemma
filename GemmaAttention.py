import tensorflow as tf
from GemmaRotaryEmbedding import  GemmaRotaryEmbedding

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
        self.num_key_value_groups = tf.math.floordiv(self.num_heads , self.num_key_value_heads )
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.is_causal = True

        # assert tf.math.floormod(self.hidden_size, self.num_heads) == 0
        self.q_proj = tf.keras.layers.Dense(self.hidden_size,activation=None, use_bias=config.attention_bias)
        self.k_proj=tf.keras.layers.Dense(256,activation=None, use_bias=config.attention_bias)
        self.v_proj=tf.keras.layers.Dense(256,activation=None, use_bias=config.attention_bias)
        self.o_proj= tf.keras.layers.Dense( self.hidden_size,activation=None, use_bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base = self.rope_theta
        )

    def rotate(self,x):
            # last dimension
            last_dim_size = tf.shape(x)[-1]

            # midpoint
            midpoint = last_dim_size // 2
            x1 = x[..., :midpoint]
            x2 = x[..., midpoint:]
            return tf.concat([-x2,x1], axis=-1)

    def apply_rotary_pos_emb(self,q, k, cos, sin, unsqueeze_dim=1):
            q_embed = tf.add(tf.multiply(q,cos), tf.multiply(self.rotate(q) , sin ))
            k_embed =tf.add(tf.multiply(k,cos), tf.multiply(self.rotate(k) , sin ))
            return q_embed,k_embed

    def repeat_kv(self, x):
        return tf.repeat(x, repeats=self.num_key_value_groups, axis=1)

    def call(self,
                 hidden_states,
                 attention_mask,
                 position_ids,
                 kv_cache):
            shape_list = tf.shape(hidden_states)
            bsz = shape_list[0]
            q_len = shape_list[1]
            query_states = self.q_proj(hidden_states)
            key_states =self.k_proj(hidden_states)
            value_states =self.v_proj(hidden_states)
            print(self.q_proj.kernel.shape)

            query_states = tf.reshape(
                query_states, (bsz, q_len, self.num_heads, self.head_dim)
            )
            query_states = tf.transpose(query_states, (0, 2, 1, 3))  # [B, 8, T, 256]
            key_states = tf.reshape(
                key_states, (bsz, q_len, self.num_key_value_heads, self.head_dim)
            )
            key_states = tf.transpose(key_states, (0, 2, 1, 3))  # [B, 1, T, 256]
            value_states = tf.reshape(
                value_states, (bsz, q_len, self.num_key_value_heads, self.head_dim)
            )
            value_states = tf.transpose(value_states, (0, 2, 1, 3))
            cos, sin = self.rotary_emb(hidden_states, position_ids, seq_len=q_len)
            query_states,key_states = self.apply_rotary_pos_emb(query_states,key_states,cos,sin)
            kv_cache.update(key_states,value_states,self.layer_idx)

            full_key_states = kv_cache.key_cache[self.layer_idx][:, :, :kv_cache.sequence_len, :]
            full_value_states = kv_cache.value_cache[self.layer_idx][:, :, :kv_cache.sequence_len, :]
            key_states = self.repeat_kv(full_key_states)
            value_states = self.repeat_kv( full_value_states)

            attn_weights = tf.matmul(query_states, key_states, transpose_b=True)
            attn_weights = attn_weights / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

            # print("attn_weights:", attn_weights.shape)
            # print("attention_mask:", attention_mask.shape)
            # print("kv_len:", seq_len)
            kv_len = tf.shape(kv_cache.key_cache[self.layer_idx])[2]

            attention_mask = attention_mask[..., :kv_len]

            attn_weights = tf.add(attn_weights, tf.cast(attention_mask,tf.float32))

            attn_weights = tf.nn.softmax(attn_weights, axis=-1)
            attn_weights = tf.cast(attn_weights, dtype=query_states.dtype)

            attn_weights = tf.nn.dropout( attn_weights, rate=self.attention_dropout)
            attn_output = tf.matmul(attn_weights,value_states)

            shape_list = tf.shape( attn_output )
            assert shape_list[0] == bsz
            assert shape_list[1] == self.num_heads
            assert shape_list[2] == q_len
            assert shape_list[3] == self.head_dim
            attn_output = tf.transpose(attn_output, perm=[0,2,1,3])
            attn_output = tf.reshape(attn_output, (bsz, q_len, self.num_heads * self.head_dim))
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights