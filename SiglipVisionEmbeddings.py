import tensorflow as tf

class SiglipVisionEmbeddings(tf.keras.Model):

    def __init__(self, config):
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        super(SiglipVisionEmbeddings, self).__init__()
        self.patch_embedding = \
            tf.keras.layers.Conv2D(
                filters=self.embed_dim,  # Corresponds to out_channels in PyTorch
                kernel_size=config.patch_size,  # Same as kernel_size in PyTorch
                strides=config.patch_size,  # Same as stride in PyTorch
                padding='valid',  # Same as padding in PyTorch
                data_format="channels_last"
            )

        num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = num_patches_per_side * num_patches_per_side
        self.num_positions = self.num_patches
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_positions,  # Number of positions
            output_dim=self.embed_dim  # Embedding dimension
        )

        # self.position_ids = tf.expand_dims(tf.range(self.num_positions),axis=0)

    def call(self, pixel_values):
        # print("pixel_values.shape:", pixel_values.shape)
        patch_embeds = self.patch_embedding(pixel_values)
        B, H_patch, W_patch, C_patch = tf.shape(patch_embeds)[0], tf.shape(patch_embeds)[1], tf.shape(patch_embeds)[2], \
        tf.shape(patch_embeds)[3]
        position_ids = tf.range(H_patch * W_patch)[tf.newaxis, :]
        embeddings = tf.reshape(patch_embeds, (B, H_patch * W_patch, C_patch))
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings