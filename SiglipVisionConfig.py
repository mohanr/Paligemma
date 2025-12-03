import tensorflow as tf

class SiglipVisionConfig():
    def __init__(self,
                 hidden_size = 1152,
                 intermediate_size = 4304,
                 num_hidden_layers = 24,
                 num_attention_heads = 16,
                 num_channels = 3,
                 image_size = 448,
                 patch_size = 14,
                 layer_norm_eps = 1e-6,
                 attention_dropout = 0.0,
                 projection_dim=2304,
                 num_image_tokens : int = 1024,
                 max_position_embeddings=1024
                 ):
        super().__init__()
        self.projection_dim=projection_dim
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_channels=num_channels
        self.image_size=image_size
        self.patch_size=patch_size
        self.layer_norm_eps=layer_norm_eps
        self.attention_dropout=attention_dropout
        self.num_image_tokens=num_image_tokens
        self.max_position_embeddings=max_position_embeddings
         
