import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
import requests
from PIL import Image
# https://colab.research.google.com/drive/1NYjR3tjOiDJ2v8nv3mhrph-_IM4p9goS?usp=sharing#scrollTo=1c800352
hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-448",
    torch_dtype=torch.float32,
    device_map="cpu"
)
state_dict = hf_model.state_dict()
print("\n=== PATCH EMBEDDING DEBUG ===")
print("PyTorch patch_embedding keys:")
for key in state_dict.keys():
    if "patch_embedding" in key:
        print(f"  {key}: {state_dict[key].shape}")
hf_processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-448")

# 2. Test with a simple image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("/Users/anu/PyCharmProjects/SigLip/image_1.png") #(requests.get(url, stream=True).raw)
prompt = "What is this?"
# image = Image.open("P.jpeg")  # YOUR image
inputs = hf_processor(text="What is this?", images=image, return_tensors="pt")
# 3. Get HF model output
inputs = hf_processor(text=prompt, images=image, return_tensors="pt")

with torch.no_grad():
    hf_output = hf_model(**inputs)
hf_logits = hf_output.logits[0, -1, :]
hf_top_token = torch.argmax(hf_logits).item()
hf_top_text = hf_processor.tokenizer.decode([hf_top_token])
print(f"PyTorch logit range: {hf_logits.min():.2f} to {hf_logits.max():.2f}")
print(f"PyTorch top token: {hf_logits.argmax()} with logit {hf_logits.max():.2f}")
print(f"PyTorch token 229711 logit: {hf_logits[229711]:.2f}")
print(f"PyTorch token 34371 logit: {hf_logits[34371]:.2f}")
print(f"Hugging Face model predicts: {hf_top_token} ('{hf_top_text}')")
hf_inputs = hf_processor(text=prompt, images=image, return_tensors="pt")
print(f"HF input_ids shape: {hf_inputs['input_ids'].shape}")
print(f"HF first 20 tokens: {hf_inputs['input_ids'][0, :20].tolist()}")

# Decode to see format
decoded = hf_processor.tokenizer.decode(hf_inputs['input_ids'][0])
print(f"HF decoded input: {repr(decoded[:200])}")

# Count image tokens
image_token_id = hf_processor.tokenizer.convert_tokens_to_ids("<image>")
num_image_tokens = (hf_inputs['input_ids'][0] == image_token_id).sum().item()
print(f"HF number of <image> tokens: {num_image_tokens}")
print(f"HF <image> token ID: {image_token_id}")

# Forward pass
with torch.no_grad():
    hf_outputs = hf_model(**hf_inputs)
    pt_embeds = hf_model.language_model.embed_tokens(torch.tensor([[34371]]))
    print(f"PyTorch embedding for 'cats'[:5]: {pt_embeds[0,0,:5]}")
    hf_logits = hf_outputs.logits[0, -1, :]
    hidden_states = hf_model.model.vision_tower.vision_model.embeddings(inputs['pixel_values'])
    encoder_output = hf_model.model.vision_tower.vision_model.encoder(inputs_embeds=hidden_states).last_hidden_state
    post_norm_output = hf_model.model.vision_tower.vision_model.post_layernorm(encoder_output)

    print(f"Post-norm output std: {torch.std(post_norm_output).item():.4f}")

    # Apply projector
    projected = hf_model.model.multi_modal_projector(post_norm_output)

    print(f"After projector (before scaling) std: {torch.std(projected).item():.4f}")

    # Apply scaling
    scaled = projected / (hf_model.config.text_config.hidden_size ** 0.5)

print(f"After scaling std: {torch.std(scaled).item():.4f}")
hf_top_token = torch.argmax(hf_logits).item()
hf_top_text = hf_processor.tokenizer.decode([hf_top_token])

print(f"\nHF model prediction: {hf_top_token} ('{hf_top_text}')")

# Check logits for token 229711
if hf_logits.shape[0] > 229711:
    hf_increa_logit = hf_logits[229711].item()
    print(f"HF token 229711 ('increa') logit: {hf_increa_logit:.2f}")

from transformers import PaliGemmaForConditionalGeneration
import torch

# Load HF model
hf_model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")

print("="*80)
print("HUGGINGFACE MODEL ARCHITECTURE")
print("="*80)

# Print main model structure
print(f"Model class: {hf_model.__class__.__name__}")
print(f"Model dtype: {hf_model.dtype}")
print(f"Device: {hf_model.device}")

# Print config
print("\nCONFIG:")
config = hf_model.config
for key in ['vocab_size', 'hidden_size', 'intermediate_size',
            'num_attention_heads', 'num_hidden_layers',
            'num_image_tokens', 'image_token_index', 'pad_token_id']:
    if hasattr(config, key):
        print(f"  {key}: {getattr(config, key)}")

# Print vision tower
print("\nVISION TOWER:")
vision_tower = hf_model.vision_tower
print(f"  Type: {vision_tower.__class__.__name__}")
print(f"  Config hidden_size: {vision_tower.config.hidden_size}")
print(f"  Config num_image_tokens: {vision_tower.config.num_image_tokens}")
print(f"  Config image_size: {vision_tower.config.image_size}")

# Print projector
print("\nMULTI-MODAL PROJECTOR:")
projector = hf_model.multi_modal_projector
print(f"  Type: {projector.__class__.__name__}")
print(f"  Linear layer: {projector.linear.__class__.__name__}")

# Print language model
print("\nLANGUAGE MODEL:")
language_model = hf_model.language_model
print(f"  Type: {language_model.__class__.__name__}")

# Print language model config
lm_config = language_model.config
print(f"  Config vocab_size: {lm_config.vocab_size}")
print(f"  Config hidden_size: {lm_config.hidden_size}")
print(f"  Config num_hidden_layers: {lm_config.num_hidden_layers}")
print(f"  Config num_attention_heads: {lm_config.num_attention_heads}")
print(f"  Config max_position_embeddings: {lm_config.max_position_embeddings}")

# Print embedding layer
print("\nEMBEDDINGS:")
embed_tokens = language_model.embed_tokens
print(f"  Type: {embed_tokens.__class__.__name__}")
print(f"  Weight shape: {embed_tokens.weight.shape}")

# Print layers
print("\nLAYERS:")
print(f"  Number of layers: {len(language_model.layers)}")
if len(language_model.layers) > 0:
    first_layer = language_model.layers[0]
    print(f"  First layer type: {first_layer.__class__.__name__}")
    # Print layer components
    print(f"    self_attn: {first_layer.self_attn.__class__.__name__}")
    print(f"    mlp: {first_layer.mlp.__class__.__name__}")
    print(f"    input_layernorm: {first_layer.input_layernorm.__class__.__name__}")
    print(f"    post_attention_layernorm: {first_layer.post_attention_layernorm.__class__.__name__}")

# Print output layer
print("\nOUTPUT LAYER:")
if hasattr(language_model, 'lm_head'):
    lm_head = language_model.lm_head
    print(f"  Type: {lm_head.__class__.__name__}")
    if hasattr(lm_head, 'weight'):
        print(f"  Weight shape: {lm_head.weight.shape}")
    if hasattr(lm_head, 'bias'):
        print(f"  Bias shape: {lm_head.bias.shape if lm_head.bias is not None else 'None'}")
else:
    print("  No lm_head (tied embeddings)")
    print(f"  Tie word embeddings: {language_model.config.tie_word_embeddings}")
