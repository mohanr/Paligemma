import torch
from transformers import PaliGemmaForConditionalGeneration, AutoTokenizer
import numpy as np
import h5py  # Ensure this import is at the top of Utils.py

import tensorflow as tf


def load_vision_layer(tf_layer, state_dict, i):
    # PyTorch layer prefix for ViT
    prefix = f"model.vision_tower.vision_model.encoder.layers.{i}."

    def load_and_assign_vision(pyt_key, tf_var, transpose=False):  # Default to NO transpose
        try:
            tensor = state_dict[prefix + pyt_key].cpu().numpy()
            tf_tensor = tf.convert_to_tensor(tensor)

            if transpose:
                tf_tensor = tf.transpose(tf_tensor)

            tf_var.assign(tf_tensor)

        except KeyError as e:
            print(f"KeyError: Failed to find ViT key for layer {i}: {prefix}{pyt_key}")
            raise e

    # Use set_weights for LayerNorm (gamma + beta together)
    ln1_gamma = state_dict[prefix + "layer_norm1.weight"].cpu().numpy()
    ln1_beta = state_dict[prefix + "layer_norm1.bias"].cpu().numpy()
    tf_layer.layer_norm1.set_weights([ln1_gamma, ln1_beta])

    ln2_gamma = state_dict[prefix + "layer_norm2.weight"].cpu().numpy()
    ln2_beta = state_dict[prefix + "layer_norm2.bias"].cpu().numpy()
    tf_layer.layer_norm2.set_weights([ln2_gamma, ln2_beta])

    print(f"Layer {i} norm1 has gamma? {hasattr(tf_layer.layer_norm1, 'gamma')}")
    print(
        f"Layer {i} norm1 gamma shape: {tf_layer.layer_norm1.gamma.shape if hasattr(tf_layer.layer_norm1, 'gamma') else 'NOT BUILT'}")

    if i == 0:  # Only print for first layer
        tf_gamma = tf_layer.layer_norm1.gamma.numpy()
        tf_beta = tf_layer.layer_norm1.beta.numpy()

        print(f"\n=== LAYER {i} NORM1 VERIFICATION ===")
        print(f"PyTorch gamma shape: {ln1_gamma.shape}")
        print(f"TF gamma shape: {tf_gamma.shape}")
        print(f"PyTorch gamma[:5]: {ln1_gamma[:5]}")
        print(f"TF gamma[:5]: {tf_gamma[:5]}")
        print(f"Gamma match: {np.allclose(ln1_gamma, tf_gamma)}")
        print(f"PyTorch beta[:5]: {ln1_beta[:5]}")
        print(f"TF beta[:5]: {tf_beta[:5]}")
        print(f"Beta match: {np.allclose(ln1_beta, tf_beta)}")

    # Load attention and MLP weights (these are fine with assign since they're individual tensors)
    load_and_assign_vision("self_attn.q_proj.weight", tf_layer.self_attn.q_proj.kernel, transpose=True)
    load_and_assign_vision("self_attn.k_proj.weight", tf_layer.self_attn.k_proj.kernel, transpose=True)
    load_and_assign_vision("self_attn.v_proj.weight", tf_layer.self_attn.v_proj.kernel, transpose=True)
    load_and_assign_vision("self_attn.out_proj.weight", tf_layer.self_attn.o_proj.kernel, transpose=True)
    load_and_assign_vision("mlp.fc1.weight", tf_layer.mlp.fc1.kernel, transpose=True)
    load_and_assign_vision("mlp.fc1.bias", tf_layer.mlp.fc1.bias, transpose=False)
    load_and_assign_vision("mlp.fc2.weight", tf_layer.mlp.fc2.kernel, transpose=True)
    load_and_assign_vision("mlp.fc2.bias", tf_layer.mlp.fc2.bias, transpose=False)


def load_layer(tf_layer, state_dict, i):
    prefix = f"model.language_model.layers.{i}."

    def load_and_assign(pyt_key, tf_var, transpose=True):

        try:
            tensor = state_dict[prefix + pyt_key].cpu().numpy()
            # print(f"DEBUG: Layer {i} Key: {pyt_key}, Pytorch Shape: {tensor.shape}")
            tf_tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)

            # if pyt_key == "mlp.down_proj.weight":
            #     print(f"DEBUG: Pytorch Shape for {pyt_key}: {tensor.shape}")

            if transpose:
                tf_tensor = tf.transpose(tf_tensor)

                # DEBUG: Check the TensorFlow shape just before the assignment (where it fails)
            # if pyt_key == "mlp.down_proj.weight":
            # print(f"DEBUG: TensorFlow Shape (Post-Transpose) for {pyt_key}: {tf_tensor.shape}")
            # Print the expected Keras shape for comparison
            # print(f"DEBUG: Keras Kernel Target Shape: {tf_var.shape}")

            tf_var.assign(tf_tensor)

        except KeyError as e:
            print(f"KeyError: Failed to find PyTorch key for layer {i}: {prefix}{pyt_key}")
            raise e

    load_and_assign("self_attn.q_proj.weight", tf_layer.self_attn.q_proj.kernel, transpose=True)
    load_and_assign("self_attn.k_proj.weight", tf_layer.self_attn.k_proj.kernel, transpose=True)
    load_and_assign("self_attn.v_proj.weight", tf_layer.self_attn.v_proj.kernel, transpose=True)
    load_and_assign("self_attn.o_proj.weight", tf_layer.self_attn.o_proj.kernel, transpose=True)

    load_and_assign("input_layernorm.weight", tf_layer.input_layernorm.weight, transpose=False)
    load_and_assign("post_attention_layernorm.weight", tf_layer.post_attention_layernorm.weight, transpose=False)

    load_and_assign("mlp.gate_proj.weight", tf_layer.mlp.gate_proj.kernel, transpose=True)
    load_and_assign("mlp.up_proj.weight", tf_layer.mlp.up_proj.kernel, transpose=True)
    load_and_assign("mlp.down_proj.weight", tf_layer.mlp.down_proj.kernel, transpose=True)


def print_language_model_details(lm):
    """Print detailed language model structure."""
    print("\nLANGUAGE MODEL DETAILS:")

    # Embeddings
    if hasattr(lm, 'embed_tokens'):
        embed = lm.embed_tokens
        print(f"  embed_tokens: {embed.__class__.__name__}")
        print(f"    Weight shape: {embed.weights[0].shape}")

    # Layers
    if hasattr(lm, 'layers'):
        print(f"  Number of layers: {len(lm.layers)}")
        if len(lm.layers) > 0:
            layer = lm.layers[0]
            print(f"  First layer structure:")

            # Check for attention
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                print(f"    self_attn: {attn.__class__.__name__}")
                # Print attention weights
                for w in attn.weights:
                    print(f"      {w.name}: {w.shape}")

            # Check for MLP
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                print(f"    mlp: {mlp.__class__.__name__}")
                for w in mlp.weights:
                    print(f"      {w.name}: {w.shape}")

            # Check for normalization
            norm_layers = ['input_layernorm', 'post_attention_layernorm', 'norm']
            for norm_name in norm_layers:
                if hasattr(layer, norm_name):
                    norm = getattr(layer, norm_name)
                    print(f"    {norm_name}: {norm.__class__.__name__}")

    # Final norm
    if hasattr(lm, 'norm'):
        print(f"  Final norm: {lm.norm.__class__.__name__}")

    # Output layer
    if hasattr(lm, 'lm_head'):
        lm_head = lm.lm_head
        print(f"  lm_head: {lm_head.__class__.__name__}")
        print(f"    Weight shape: {lm_head.weights[0].shape}")
        if len(lm_head.weights) > 1:
            print(f"    Bias shape: {lm_head.weights[1].shape}")


def load_gemma_tf_model(tf_model):
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-mix-448")
    hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-mix-448",
        torch_dtype=torch.float32,  # Ensure full precision
        device_map="cpu"  # Load onto CPU to prevent memory issues
    )
    state_dict = hf_model.state_dict()
    embed_tensor = state_dict["model.language_model.embed_tokens.weight"].cpu().numpy()

    tf_model.language_model.model.embed_tokens.embeddings.assign(
        tf.convert_to_tensor(embed_tensor)
    )
    num_layers = len(tf_model.language_model.model.net.layers)
    for i in range(num_layers):
        # print(f"Loading layer {i}…")
        load_layer(tf_model.language_model.model.net.layers[i], state_dict, i)
    norm_weight = state_dict["model.language_model.norm.weight"].cpu().numpy()
    tf_model.language_model.model.norm.weight.assign(
        tf.convert_to_tensor(norm_weight)
    )
    # Load lm_head weights (in PyTorch these are tied to embeddings, same tensor)
    lm_head_weight = state_dict["lm_head.weight"].cpu().numpy()
    tf_model.language_model.lm_head.kernel.assign(
        tf.transpose(tf.convert_to_tensor(lm_head_weight))
    )

    patch_tensor = state_dict["model.vision_tower.vision_model.embeddings.patch_embedding.weight"].cpu().numpy()

    # Transpose PyTorch (Out, In, H, W) to Keras (H, W, In, Out)
    tf_tensor = tf.transpose(tf.convert_to_tensor(patch_tensor), perm=[2, 3, 1, 0])

    patch_bias = state_dict["model.vision_tower.vision_model.embeddings.patch_embedding.bias"].cpu().numpy()

    # Use set_weights() for Conv2D layer (kernel + bias)
    tf_model.vision_tower.vision_model.embeddings.patch_embedding.set_weights([
        tf_tensor,
        patch_bias
    ])

    print("\n=== BIAS VERIFICATION ===")
    loaded = tf_model.vision_tower.vision_model.embeddings.patch_embedding.bias.numpy()
    print(f"Expected: {patch_bias[:5]}")
    print(f"Loaded: {loaded[:5]}")
    print(f"Match: {np.allclose(patch_bias, loaded)}")
    tf_model.vision_tower.vision_model.embeddings.position_embedding.embeddings.assign(
        tf.convert_to_tensor(
            state_dict["model.vision_tower.vision_model.embeddings.position_embedding.weight"].cpu().numpy())
    )
    print("\n=== POSITION EMBEDDING VERIFICATION ===")
    pt_pos = state_dict["model.vision_tower.vision_model.embeddings.position_embedding.weight"].cpu().numpy()
    tf_pos = tf_model.vision_tower.vision_model.embeddings.position_embedding.embeddings.numpy()
    print(f"PyTorch shape: {pt_pos.shape}")
    print(f"TF shape: {tf_pos.shape}")
    print(f"First 5 values match: {np.allclose(pt_pos[0, :5], tf_pos[0, :5])}")
    print(f"All values match: {np.allclose(pt_pos, tf_pos)}")
    num_vision_layers = 24
    for i in range(num_vision_layers):
        load_vision_layer(tf_model.vision_tower.vision_model.encoder._layers[i], state_dict, i)

    # Build post_layernorm with a dummy forward pass
    dummy = tf.zeros((1, 1024, 1152))
    _ = tf_model.vision_tower.vision_model.post_layernorm(dummy)

    # Use set_weights() for LayerNorm (gamma + beta)
    post_norm_gamma = state_dict["model.vision_tower.vision_model.post_layernorm.weight"].cpu().numpy()
    post_norm_beta = state_dict["model.vision_tower.vision_model.post_layernorm.bias"].cpu().numpy()

    tf_model.vision_tower.vision_model.post_layernorm.set_weights([
        post_norm_gamma,
        post_norm_beta
    ])

    print("\n=== CHECKING WHAT WE JUST ASSIGNED ===")
    print(f"weights[0] name: {tf_model.vision_tower.vision_model.post_layernorm.weights[0].name}")
    print(f"weights[1] name: {tf_model.vision_tower.vision_model.post_layernorm.weights[1].name}")
    print(f"Has gamma attr? {hasattr(tf_model.vision_tower.vision_model.post_layernorm, 'gamma')}")
    if hasattr(tf_model.vision_tower.vision_model.post_layernorm, 'gamma'):
        print(
            f"gamma is weights[0]? {tf_model.vision_tower.vision_model.post_layernorm.gamma is tf_model.vision_tower.vision_model.post_layernorm.weights[0]}")
        print(f"gamma values[:5]: {tf_model.vision_tower.vision_model.post_layernorm.gamma.numpy()[:5]}")
    pt_gamma = state_dict["model.vision_tower.vision_model.post_layernorm.weight"].cpu().numpy()
    pt_beta = state_dict["model.vision_tower.vision_model.post_layernorm.bias"].cpu().numpy()
    tf_gamma = tf_model.vision_tower.vision_model.post_layernorm.gamma.numpy()
    tf_beta = tf_model.vision_tower.vision_model.post_layernorm.beta.numpy()

    print(f"PyTorch gamma[:5]: {pt_gamma[:5]}")
    print(f"TF gamma[:5]: {tf_gamma[:5]}")
    print(f"Gamma match: {np.allclose(pt_gamma, tf_gamma)}")
    print(f"PyTorch beta[:5]: {pt_beta[:5]}")
    print(f"TF beta[:5]: {tf_beta[:5]}")
    print(f"Beta match: {np.allclose(pt_beta, tf_beta)}")
    print("\n=== BUILDING PROJECTOR ===")
    dummy_input = tf.zeros((1, 1024, 1152), dtype=tf.float32)
    _ = tf_model.multi_modal_projector(dummy_input)
    print(f"Projector built with kernel shape: {tf_model.multi_modal_projector.linear.kernel.shape}")

    projection_key_weight = "model.multi_modal_projector.linear.weight"
    projection_tensor = state_dict[projection_key_weight].cpu().numpy()

    projection_key_bias = "model.multi_modal_projector.linear.bias"
    bias_tensor = state_dict[projection_key_bias].cpu().numpy()

    # Use set_weights() like KerasHub does, NOT assign()
    tf_model.multi_modal_projector.linear.set_weights([
        projection_tensor.T,  # kernel (transposed)
        bias_tensor  # bias
    ])
    print("\n=== PROJECTOR WEIGHT VERIFICATION ===")
    loaded_weight = tf_model.multi_modal_projector.linear.kernel.numpy()
    print(f"PyTorch projector shape: {projection_tensor.shape}")
    print(f"TF projector shape (after transpose): {loaded_weight.shape}")
    print(f"PyTorch first 5 values: {projection_tensor.flatten()[:5]}")
    print(f"TF first 5 values: {loaded_weight.flatten()[:5]}")
    print(f"Weights match (transposed)? {np.allclose(projection_tensor.T, loaded_weight, atol=1e-5)}")

    # Build lm_head BEFORE tie_weights so it doesn't reinitialize later
    if not tf_model.language_model.lm_head.built:
        tf_model.language_model.lm_head.build((None, tf_model.config.text_config.hidden_size))

    tf_model.tie_weights()
    tf_kernel = tf_model.multi_modal_projector.linear.kernel.numpy()
    tf_bias = tf_model.multi_modal_projector.linear.bias.numpy()

    pt_kernel = np.load('pt_projector_kernel.npy')
    pt_bias = np.load('pt_projector_bias.npy')

    print(f"TF kernel shape: {tf_kernel.shape}")  # Should be (1152, 2048)
    print(f"TF kernel std: {tf_kernel.std():.6f}")
    print(f"PT kernel std: {pt_kernel.std():.6f}")
    print(f"Kernels match (transposed)? {np.allclose(pt_kernel.T, tf_kernel, atol=1e-5)}")

    print(f"TF bias std: {tf_bias.std():.6f}")
    print(f"PT bias std: {pt_bias.std():.6f}")
    print(f"Bias match? {np.allclose(pt_bias, tf_bias, atol=1e-5)}")

    if not np.allclose(pt_kernel.T, tf_kernel, atol=1e-5):
        print("WEIGHTS DO NOT MATCH!")
        print(f"PT kernel [0,:5]: {pt_kernel[0,:5]}")
        print(f"TF kernel [:5,0]: {tf_kernel[:5,0]}")
    return tokenizer, tf_model