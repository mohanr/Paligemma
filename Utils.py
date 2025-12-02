import torch
from transformers import PaliGemmaForConditionalGeneration


def get_tokenizer():
    # tokenizer = AutoTokenizer.from_pretrained("/Users/anu/PycharmProjects/Siglip/gemma-keras-gemma_1.1_instruct_2b_en-v3")
    global tokenizer
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-mix-448")
    except Exception as e:
        print(f"Error loading Paligemma tokenizer: {e}")
    return tokenizer


import h5py  # Ensure this import is at the top of Utils.py


def load_tf_model():
    # --- 1. Tokenizer Load ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-mix-448")

    # --- 2. Load Weights from HDF5 (.h5) File ---
    keras_h5_path = "/Users/anu/PycharmProjects/paligemma-3b-mix-448-keras_weights/model.weights.h5"

    weights_dict = {}

    try:
        with h5py.File(keras_h5_path, 'r') as f:

            def load_h5_group(group, prefix=''):
                for name, item in group.items():
                    key = prefix + name
                    if isinstance(item, h5py.Dataset):
                        weights_dict[key] = item[()]
                    elif isinstance(item, h5py.Group):
                        load_h5_group(item, key + '/')

            load_h5_group(f)

    except Exception as e:
        print(f"Error loading HDF5 file with h5py: {e}")
        raise

    print(f"Total weights loaded from .h5: {len(weights_dict)}")
    print("Example keys:", list(weights_dict.keys()))  # Check the format of the loaded keys

    return tokenizer, weights_dict  # Return the weights dictionary


import tensorflow as tf


# --- New function in Utils.py ---

def load_vision_layer(tf_layer, state_dict, i):
    # PyTorch layer prefix for ViT
    prefix = f"model.vision_tower.vision_model.encoder.layers.{i}."
    print("\n--- DEBUG: VISION MLP KEYS (LAYER 0) ---")
    # Inside load_gemma_tf_model, near the loading line:
    print("\n--- DEBUG: PROJECTOR KEYS ---")
    for key in state_dict.keys():
        if "multi_modal_projector" in key:
            print(f"Key: {key}")
    print("------------------------------\n")
    print(f"DEBUG: Loading layer {i}. Object type: {type(tf_layer)}")
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
            tf_tensor = tf.convert_to_tensor(tensor)

            if transpose:
                tf_tensor = tf.transpose(tf_tensor)

            tf_var.assign(tf_tensor)

        except KeyError as e:
            print(f"KeyError: Failed to find PyTorch key for layer {i}: {prefix}{pyt_key}")
            raise e

    load_and_assign("self_attn.q_proj.weight", tf_layer.self_attn.q_proj.kernel)
    load_and_assign("self_attn.k_proj.weight", tf_layer.self_attn.k_proj.kernel)
    load_and_assign("self_attn.v_proj.weight", tf_layer.self_attn.v_proj.kernel)
    load_and_assign("self_attn.o_proj.weight", tf_layer.self_attn.o_proj.kernel)

    load_and_assign("input_layernorm.weight", tf_layer.input_layernorm.weight, transpose=False)
    load_and_assign("post_attention_layernorm.weight", tf_layer.post_attention_layernorm.weight, transpose=False)

    load_and_assign("mlp.gate_proj.weight", tf_layer.mlp.gate_proj.kernel)
    load_and_assign("mlp.up_proj.weight", tf_layer.mlp.up_proj.kernel)
    load_and_assign("mlp.down_proj.weight", tf_layer.mlp.down_proj.kernel)


def load_gemma_tf_model(tf_model):
    tokenizer ,  weights = load_tf_model()
    hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-mix-224",
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
        expected_key = f"language_model.model.layers.{i}.self_attn.q_proj.weight"

        if expected_key in weights:
            print(f"Loading layer {i}…")
            load_layer(tf_model.language_model.model.net.layers[i], state_dict, i)
        else:
            print(f"Stopping load: Weight for layer {i} not found. Assuming checkpoint has {i} layers.")
            break
    norm_weight = state_dict["model.language_model.norm.weight"].cpu().numpy()
    tf_model.language_model.model.norm.weight.assign(
        tf.convert_to_tensor(norm_weight)
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
    projection_key_weight = "model.multi_modal_projector.linear.weight"
    projection_tensor = state_dict[projection_key_weight].cpu().numpy()

    tf_model.multi_modal_projector.linear.kernel.assign(
        tf.transpose(tf.convert_to_tensor(projection_tensor))
    )

    projection_key_bias = "model.multi_modal_projector.linear.bias"
    bias_tensor = state_dict[projection_key_bias].cpu().numpy()

    tf_model.multi_modal_projector.linear.bias.assign(
        tf.convert_to_tensor(bias_tensor)
    )
    print("Gemma TF model fully loaded.")
    return tokenizer, tf_model


