from SiglipVisionConfig import SiglipVisionConfig
from GemmaConfig import GemmaConfig

class PaliGemmaConfig:
    def __init__(self,
                 vision_config=None,
                 text_config=None,
                 ignore_index=-100,
                 image_token_index=256000,
                 # vocab_size=257152,
                 vocab_size=257216,
                 projection_dim=2304,
                 # hidden_size=2048,
                 hidden_size=2048,
                 pad_token_id=None,
                 **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        self.vocab_size=vocab_size
        # self.num_image_tokens= (self.image_size // self.patch_size) ** 2
        self.num_image_tokens= (224 // 16) ** 2
        self.vision_config = SiglipVisionConfig(
            hidden_size=1152,
            projection_dim=self.projection_dim
        )

        # Pass the correct VLM dimensions to GemmaConfig (the language model's dimensions are already correct from GemmaConfig.py defaults):
        self.text_config = GemmaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size
        )