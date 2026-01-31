"""
Compare PyTorch and TensorFlow weights for each SigLIP layer
"""
import torch
import tensorflow as tf
import numpy as np
from transformers import PaliGemmaForConditionalGeneration
from modelling_gemma import PaliGemmaForConditionalGeneration as TFPaliGemma
from Utils import load_gemma_tf_model
from PaliGemmaConfig import PaliGemmaConfig
import KVCache

# Load PyTorch model
print("Loading PyTorch model...")
pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-448",
    torch_dtype=torch.float32,
    device_map="cpu"
)
pt_state = pt_model.state_dict()

# Create and load TensorFlow model
print("Loading TensorFlow model...")
config = PaliGemmaConfig()
tf_model = TFPaliGemma(config=config)

# Build
dummy_img = tf.zeros((1, 448, 448, 3), dtype=tf.float32)
dummy_ids = tf.zeros((1, 10), dtype=tf.int32)
dummy_mask = tf.ones((1, 10), dtype=tf.int32)
_ = tf_model(
    input_ids=dummy_ids,
    pixel_values=dummy_img,
    attention_mask=dummy_mask,
    kv_cache=KVCache.KVCache(),
    training=False
)

# Load weights
_, tf_model = load_gemma_tf_model(tf_model)

print("\n" + "="*80)
print("CHECKING VISION ENCODER WEIGHTS")
print("="*80)

# Check patch embedding
pt_patch = pt_state["model.vision_tower.vision_model.embeddings.patch_embedding.weight"].cpu().numpy()
tf_patch = tf_model.vision_tower.vision_model.embeddings.patch_embedding.kernel.numpy()

print("\n1. PATCH EMBEDDING:")
print(f"   PyTorch shape: {pt_patch.shape}")  # [out, in, h, w]
print(f"   TensorFlow shape: {tf_patch.shape}")  # [h, w, in, out]
# Need to transpose: [out, in, h, w] -> [h, w, in, out]
pt_patch_transposed = pt_patch.transpose(2, 3, 1, 0)
print(f"   Match: {np.allclose(pt_patch_transposed, tf_patch, atol=1e-5)}")
if not np.allclose(pt_patch_transposed, tf_patch, atol=1e-5):
    print(f"   ❌ MISMATCH! Max diff: {np.abs(pt_patch_transposed - tf_patch).max()}")

# Check position embeddings
pt_pos = pt_state["model.vision_tower.vision_model.embeddings.position_embedding.weight"].cpu().numpy()
tf_pos = tf_model.vision_tower.vision_model.embeddings.position_embedding.embeddings.numpy()

print("\n2. POSITION EMBEDDINGS:")
print(f"   PyTorch shape: {pt_pos.shape}")
print(f"   TensorFlow shape: {tf_pos.shape}")
print(f"   Match: {np.allclose(pt_pos, tf_pos, atol=1e-5)}")
if not np.allclose(pt_pos, tf_pos, atol=1e-5):
    print(f"   ❌ MISMATCH! Max diff: {np.abs(pt_pos - tf_pos).max()}")

# Check each encoder layer
num_layers = 24
for layer_idx in range(num_layers):
    print(f"\n3.{layer_idx} ENCODER LAYER {layer_idx}:")

    prefix_pt = f"model.vision_tower.vision_model.encoder.layers.{layer_idx}"

    # Layer norm 1
    pt_ln1_weight = pt_state[f"{prefix_pt}.layer_norm1.weight"].cpu().numpy()
    pt_ln1_bias = pt_state[f"{prefix_pt}.layer_norm1.bias"].cpu().numpy()
    tf_ln1_weight = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].layer_norm1.gamma.numpy()
    tf_ln1_bias = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].layer_norm1.beta.numpy()

    ln1_match = np.allclose(pt_ln1_weight, tf_ln1_weight, atol=1e-5) and np.allclose(pt_ln1_bias, tf_ln1_bias, atol=1e-5)
    print(f"   LayerNorm1: {'✓' if ln1_match else '❌'}")
    if not ln1_match:
        print(f"      Weight max diff: {np.abs(pt_ln1_weight - tf_ln1_weight).max()}")
        print(f"      Bias max diff: {np.abs(pt_ln1_bias - tf_ln1_bias).max()}")

    # Attention weights
    pt_q = pt_state[f"{prefix_pt}.self_attn.q_proj.weight"].cpu().numpy()
    pt_k = pt_state[f"{prefix_pt}.self_attn.k_proj.weight"].cpu().numpy()
    pt_v = pt_state[f"{prefix_pt}.self_attn.v_proj.weight"].cpu().numpy()
    pt_o = pt_state[f"{prefix_pt}.self_attn.out_proj.weight"].cpu().numpy()

    tf_q = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].self_attn.q_proj.kernel.numpy()
    tf_k = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].self_attn.k_proj.kernel.numpy()
    tf_v = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].self_attn.v_proj.kernel.numpy()
    tf_o = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].self_attn.o_proj.kernel.numpy()

    attn_match = (np.allclose(pt_q.T, tf_q, atol=1e-5) and
                  np.allclose(pt_k.T, tf_k, atol=1e-5) and
                  np.allclose(pt_v.T, tf_v, atol=1e-5) and
                  np.allclose(pt_o.T, tf_o, atol=1e-5))
    print(f"   Attention: {'✓' if attn_match else '❌'}")
    if not attn_match:
        print(f"      Q max diff: {np.abs(pt_q.T - tf_q).max()}")
        print(f"      K max diff: {np.abs(pt_k.T - tf_k).max()}")
        print(f"      V max diff: {np.abs(pt_v.T - tf_v).max()}")
        print(f"      O max diff: {np.abs(pt_o.T - tf_o).max()}")

    # Layer norm 2
    pt_ln2_weight = pt_state[f"{prefix_pt}.layer_norm2.weight"].cpu().numpy()
    pt_ln2_bias = pt_state[f"{prefix_pt}.layer_norm2.bias"].cpu().numpy()
    tf_ln2_weight = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].layer_norm2.gamma.numpy()
    tf_ln2_bias = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].layer_norm2.beta.numpy()

    ln2_match = np.allclose(pt_ln2_weight, tf_ln2_weight, atol=1e-5) and np.allclose(pt_ln2_bias, tf_ln2_bias, atol=1e-5)
    print(f"   LayerNorm2: {'✓' if ln2_match else ''}")
    if not ln2_match:
        print(f"      Weight max diff: {np.abs(pt_ln2_weight - tf_ln2_weight).max()}")
        print(f"      Bias max diff: {np.abs(pt_ln2_bias - tf_ln2_bias).max()}")

    # MLP weights
    pt_fc1 = pt_state[f"{prefix_pt}.mlp.fc1.weight"].cpu().numpy()
    pt_fc2 = pt_state[f"{prefix_pt}.mlp.fc2.weight"].cpu().numpy()

    tf_fc1 = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].mlp.fc1.kernel.numpy()
    tf_fc2 = tf_model.vision_tower.vision_model.encoder.layers[layer_idx].mlp.fc2.kernel.numpy()

    mlp_match = np.allclose(pt_fc1.T, tf_fc1, atol=1e-5) and np.allclose(pt_fc2.T, tf_fc2, atol=1e-5)
    print(f"   MLP: {'✓' if mlp_match else '❌'}")
    if not mlp_match:
        print(f"      FC1 max diff: {np.abs(pt_fc1.T - tf_fc1).max()}")
        print(f"      FC2 max diff: {np.abs(pt_fc2.T - tf_fc2).max()}")

    # Stop at first layer with issues for debugging
    if not (ln1_match and attn_match and ln2_match and mlp_match):
        print(f"\n    Found mismatches in layer {layer_idx}!")
        break

# Check post layernorm
pt_post = pt_state["model.vision_tower.vision_model.post_layernorm.weight"].cpu().numpy()
pt_post_bias = pt_state["model.vision_tower.vision_model.post_layernorm.bias"].cpu().numpy()
tf_post = tf_model.vision_tower.vision_model.post_layernorm.gamma.numpy()
tf_post_bias = tf_model.vision_tower.vision_model.post_layernorm.beta.numpy()

print("\n4. POST LAYERNORM:")
post_match = np.allclose(pt_post, tf_post, atol=1e-5) and np.allclose(pt_post_bias, tf_post_bias, atol=1e-5)
print(f"   Match: {'✓' if post_match else '❌'}")
if not post_match:
    print(f"   Weight max diff: {np.abs(pt_post - tf_post).max()}")
    print(f"   Bias max diff: {np.abs(pt_post_bias - tf_post_bias).max()}")

print("\n" + "="*80)
print("WEIGHT COMPARISON COMPLETE")
print("="*80)