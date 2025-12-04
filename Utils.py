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

    load_and_assign_vision("layer_norm1.weight", tf_layer.layer_norm1.gamma, transpose=False)
    load_and_assign_vision("layer_norm2.weight", tf_layer.layer_norm2.gamma, transpose=False)
    # ViT norms often have a bias as well
    load_and_assign_vision("layer_norm1.bias", tf_layer.layer_norm1.beta, transpose=False)
    load_and_assign_vision("layer_norm2.bias", tf_layer.layer_norm2.beta, transpose=False)

    load_and_assign_vision("self_attn.q_proj.weight", tf_layer.self_attn.q_proj.kernel)
    load_and_assign_vision("self_attn.k_proj.weight", tf_layer.self_attn.k_proj.kernel)
    load_and_assign_vision("self_attn.v_proj.weight", tf_layer.self_attn.v_proj.kernel)
    load_and_assign_vision("self_attn.out_proj.weight", tf_layer.self_attn.o_proj.kernel)

    load_and_assign_vision(
        "mlp.fc1.weight",
        tf_layer.mlp.fc1.kernel,
        transpose=True
    )
    load_and_assign_vision(
        "mlp.fc1.bias",
        tf_layer.mlp.fc1.bias,
        transpose=False
    )

    load_and_assign_vision(
        "mlp.fc2.weight",
        tf_layer.mlp.fc2.kernel,
        transpose=True
    )
    load_and_assign_vision(
        "mlp.fc2.bias",
        tf_layer.mlp.fc2.bias,
        transpose=False
    )
def load_layer(tf_layer, state_dict, i):

    prefix = f"model.language_model.layers.{i}."

    def load_and_assign(pyt_key, tf_var, transpose=True):

        try:
            tensor = state_dict[prefix + pyt_key].cpu().numpy()
            # print(f"DEBUG: Layer {i} Key: {pyt_key}, Pytorch Shape: {tensor.shape}")
            tf_tensor = tf.convert_to_tensor(tensor,dtype=tf.float32)

            # if pyt_key == "mlp.down_proj.weight":
            #     print(f"DEBUG: Pytorch Shape for {pyt_key}: {tensor.shape}")

            if transpose:
                tf_tensor = tf.transpose(tf_tensor)

                # DEBUG: Check the TensorFlow shape just before the assignment (where it fails)
            if pyt_key == "mlp.down_proj.weight":
                print(f"DEBUG: TensorFlow Shape (Post-Transpose) for {pyt_key}: {tf_tensor.shape}")
                # Print the expected Keras shape for comparison
                print(f"DEBUG: Keras Kernel Target Shape: {tf_var.shape}")

            tf_var.assign(tf_tensor)

        except KeyError as e:
            print(f"KeyError: Failed to find PyTorch key for layer {i}: {prefix}{pyt_key}")
            raise e

    load_and_assign("self_attn.q_proj.weight", tf_layer.self_attn.q_proj.kernel,transpose=True)
    load_and_assign("self_attn.k_proj.weight", tf_layer.self_attn.k_proj.kernel,transpose=True)
    load_and_assign("self_attn.v_proj.weight", tf_layer.self_attn.v_proj.kernel,transpose=True)
    load_and_assign("self_attn.o_proj.weight", tf_layer.self_attn.o_proj.kernel,transpose=True)

    load_and_assign("input_layernorm.weight", tf_layer.input_layernorm.weight, transpose=True)
    load_and_assign("post_attention_layernorm.weight", tf_layer.post_attention_layernorm.weight, transpose=True)

    load_and_assign("mlp.gate_proj.weight", tf_layer.mlp.gate_proj.kernel,transpose=True)
    load_and_assign("mlp.up_proj.weight", tf_layer.mlp.up_proj.kernel,transpose=True)
    load_and_assign("mlp.down_proj.weight", tf_layer.mlp.down_proj.kernel,transpose=True)


def load_gemma_tf_model(tf_model):
    tokenizer  = AutoTokenizer.from_pretrained("google/paligemma-3b-mix-448")
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
            print(f"Loading layer {i}…")
            load_layer(tf_model.language_model.model.net.layers[i], state_dict, i)
    norm_weight = state_dict["model.language_model.norm.weight"].cpu().numpy()
    tf_model.language_model.model.norm.weight.assign(
        tf.convert_to_tensor(norm_weight)
    )
    lm_head_weight = state_dict["lm_head.weight"].cpu().numpy()
    tf_model.language_model.lm_head.kernel.assign(
        tf.transpose(tf.convert_to_tensor(lm_head_weight))
    )

    patch_tensor = state_dict["model.vision_tower.vision_model.embeddings.patch_embedding.weight"].cpu().numpy()

    # Transpose PyTorch (Out, In, H, W) to Keras (H, W, In, Out)
    tf_tensor = tf.transpose(tf.convert_to_tensor(patch_tensor), perm=[2, 3, 1, 0])

    tf_model.vision_tower.vision_model.embeddings.patch_embedding.kernel.assign(tf_tensor)
    tf_model.vision_tower.vision_model.embeddings.patch_embedding.kernel.assign(
        tf_tensor
    )

    tf_model.vision_tower.vision_model.embeddings.position_embedding.embeddings.assign(
        tf.convert_to_tensor(
            state_dict["model.vision_tower.vision_model.embeddings.position_embedding.weight"].cpu().numpy())
    )

    num_vision_layers = 24
    for i in range(num_vision_layers):
        load_vision_layer(tf_model.vision_tower.vision_model.encoder._layers[i], state_dict, i)

    tf_model.vision_tower.vision_model.post_layernorm.gamma.assign(
        tf.convert_to_tensor(state_dict["model.vision_tower.vision_model.post_layernorm.weight"].cpu().numpy())
    )
    tf_model.vision_tower.vision_model.post_layernorm.beta.assign(
        tf.convert_to_tensor(state_dict["model.vision_tower.vision_model.post_layernorm.bias"].cpu().numpy())
    )
    target_dtype = tf_model.multi_modal_projector.linear.kernel.dtype

    projection_key_weight = "model.multi_modal_projector.linear.weight"
    projection_tensor = state_dict[projection_key_weight].cpu().numpy()
    tf_model.multi_modal_projector.linear.kernel.assign(
        tf.cast(tf.transpose(tf.convert_to_tensor(projection_tensor)), target_dtype)
    )

    projection_key_bias = "model.multi_modal_projector.linear.bias"
    bias_tensor = state_dict[projection_key_bias].cpu().numpy()
    tf_model.multi_modal_projector.linear.bias.assign(
        tf.cast(tf.convert_to_tensor(bias_tensor), target_dtype)
    )
    tf_model.tie_weights()
    # projection_key_weight = "model.multi_modal_projector.linear.weight"
    # projection_tensor = state_dict[projection_key_weight].cpu().numpy()
    #
    # tf_model.multi_modal_projector.linear.kernel.assign(
    #     tf.transpose(tf.convert_to_tensor(projection_tensor))
    # )
    #
    # projection_key_bias = "model.multi_modal_projector.linear.bias"
    # bias_tensor = state_dict[projection_key_bias].cpu().numpy()
    #
    # tf_model.multi_modal_projector.linear.bias.assign(
    #     tf.convert_to_tensor(bias_tensor)
    # )
    print("Gemma TF model fully loaded.")
    return tokenizer, tf_model


