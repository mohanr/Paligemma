import tensorflow as tf

class SiglipAttention(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = tf.math.floordiv(self.embed_dim,self.num_heads)
        self.scale = tf.math.pow(tf.cast(self.head_dim,tf.float32),-0.5)
        self.dropout = config.attention_dropout
        self.k_proj = tf.keras.layers.Dense(self.embed_dim,
                                         input_shape=(self.embed_dim,),
                                         activation=None, use_bias=False)
        self.q_proj = tf.keras.layers.Dense(self.embed_dim,
                                         input_shape=(self.embed_dim,),
                                         activation=None, use_bias=False)
        self.v_proj = tf.keras.layers.Dense(self.embed_dim,
                                         input_shape=(self.embed_dim,),
                                         activation=None, use_bias=False)
        self.o_proj = tf.keras.layers.Dense(self.embed_dim,
                                         input_shape=(self.embed_dim,),
                                         activation=None, use_bias=False)

    def call(self, hidden_states):
        shape_list = tf.shape(hidden_states).numpy().tolist()
        batch_size = shape_list[0]
        seq_len = shape_list[1]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape the tensor
        query_states = tf.reshape(query_states, [batch_size, seq_len, self.num_heads, self.head_dim])

        # Transpose the tensor (swap dimensions 1 and 2)
        query_states = tf.transpose(query_states, perm=[0, 2, 1, 3])
        # Reshape the tensor
        key_states = tf.reshape(key_states, [batch_size, seq_len, self.num_heads, self.head_dim])

        # Transpose the tensor (swap dimensions 1 and 2)
        key_states = tf.transpose(key_states, perm=[0, 2, 1, 3])
        # Reshape the tensor
        value_states = tf.reshape(value_states, [batch_size, seq_len, self.num_heads, self.head_dim])

        # Transpose the tensor (swap dimensions 1 and 2)
        value_states = tf.transpose(value_states, perm=[0, 2, 1, 3])


        output = tf.matmul(query_states, tf.transpose(key_states, perm=[0, 1, 3, 2]))
        attn_weights = output * self.scale
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_output = tf.matmul(attn_weights,value_states)
        attn_output = tf.transpose(attn_output, perm=[0, 1, 2, 3])
        attn_output = tf.reshape( attn_output,(batch_size, seq_len, self.embed_dim))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

