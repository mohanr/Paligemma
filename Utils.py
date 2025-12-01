import keras_nlp
import tensorflow as tf
import json
from safetensors import safe_open
import os
import torch
import sentencepiece as spm


def load_gemma_decoder_layer(tf_layer, pt_weights, idx):
    prefix = f"model.layers.{idx}."

    # ATTENTION
    q = pt_weights[prefix + "self_attn.q_proj.weight"]
    k = pt_weights[prefix + "self_attn.k_proj.weight"]
    v = pt_weights[prefix + "self_attn.v_proj.weight"]
    o = pt_weights[prefix + "self_attn.o_proj.weight"]

    tf_layer.self_attn.q_proj.kernel.assign(tf.convert_to_tensor(q).T)
    tf_layer.self_attn.k_proj.kernel.assign(tf.convert_to_tensor(k).T)
    tf_layer.self_attn.v_proj.kernel.assign(tf.convert_to_tensor(v).T)
    tf_layer.self_attn.o_proj.kernel.assign(tf.convert_to_tensor(o).T)

    # RMS Norms
    tf_layer.input_layernorm.weight.assign(tf.convert_to_tensor(
        pt_weights[prefix + "input_layernorm.weight"]
    ))

    tf_layer.post_attention_layernorm.weight.assign(tf.convert_to_tensor(
        pt_weights[prefix + "post_attention_layernorm.weight"]
    ))

    # MLP
    tf_layer.mlp.gate_proj.kernel.assign(
        tf.convert_to_tensor(pt_weights[prefix + "mlp.gate_proj.weight"]).T
    )
    tf_layer.mlp.up_proj.kernel.assign(
        tf.convert_to_tensor(pt_weights[prefix + "mlp.up_proj.weight"]).T
    )
    tf_layer.mlp.down_proj.kernel.assign(
        tf.convert_to_tensor(pt_weights[prefix + "mlp.down_proj.weight"]).T
    )

def load_tf_model():
    # Load the tokenizer
    # vocab_file = "/Users/anu/PycharmProjects/assets/tokenizer/vocabulary.spm"
    # import sentencepiece as spm
    # sp = spm.SentencePieceProcessor()
    # sp.load(vocab_file)
    #
    # class TokenizerWrapper:
    #     def __init__(self, sp):
    #         self.sp = sp
    #         self.eos_token_id = sp.eos_id()  # usually Gemma uses 0 or 1
    #         self.pad_token_id = sp.pad_id()  # if needed
    #
    #     def encode(self, text, out_type=int):
    #         return [self.sp.encode(t, out_type=out_type) if isinstance(t, str) else t for t in
    #                 ([text] if isinstance(text, str) else text)]
    #
    #     def decode(self, token_ids, skip_special_tokens=True):
    #         # Flatten in case of list-of-lists
    #         if isinstance(token_ids[0], list):
    #             token_ids = [t for sub in token_ids for t in sub]
    #         return self.sp.decode(token_ids)
    #
    # tokenizer = TokenizerWrapper(sp)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
    print("tokenizer.vocab_size:", tokenizer.vocab_size)
    print("example token -> id:", tokenizer.encode("this building is", add_special_tokens=False))

    # tokenizer = AutoTokenizer.from_pretrained("/Users/anu/PycharmProjects/Siglip/gemma-keras-gemma_1.1_instruct_2b_en-v3")
    all_tensors = {}
    with open(os.path.join("/Users/anu/PycharmProjects", "model.safetensors.index.json"), "r") as f:
        index = json.load(f)
    # weight_map = index["weight_map"]

    with safe_open("/Users/anu/PycharmProjects/model-00001-of-00002.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)
    with safe_open("/Users/anu/PycharmProjects/model-00002-of-00002.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    print("Total tensors loaded:", len(all_tensors))
    print("keys:", list(all_tensors.keys()))
    for k in all_tensors.keys():
        if "mlp" in k:
            print("mlp:", k)

    return tokenizer, all_tensors

def load_layer(tf_layer, weights, idx):
        prefix = f"model.layers.{idx}."

    # --- ATTENTION ---
        tf_layer.self_attn.q_proj.kernel.assign(
            tf.transpose(tf.convert_to_tensor(weights[prefix + "self_attn.q_proj.weight"]
                                 .to(torch.float32)
                                 .cpu()
                                 .numpy()
            ))
        )
        tf_layer.self_attn.k_proj.kernel.assign(
            tf.transpose(tf.convert_to_tensor(weights[prefix + "self_attn.k_proj.weight"]
                                .to(torch.float32)
                                .cpu()
                                .numpy()
            ))
        )
        tf_layer.self_attn.v_proj.kernel.assign(
            tf.transpose(tf.convert_to_tensor(weights[prefix + "self_attn.v_proj.weight"]
                                .to(torch.float32)
                                .cpu()
                                .numpy()
            ))
        )
        tf_layer.self_attn.o_proj.kernel.assign(
            tf.transpose(tf.convert_to_tensor(weights[prefix + "self_attn.o_proj.weight"]
                                .to(torch.float32)
                                .cpu()
                                .numpy()
        ))
        )

        # --- RMS NORMS ---
        tf_layer.input_layernorm.weight.assign(
            tf.convert_to_tensor(weights[prefix + "input_layernorm.weight"]
                                .to(torch.float32)
                                .cpu()
                                .numpy()
        )
        )
        tf_layer.post_attention_layernorm.weight.assign(
            tf.convert_to_tensor(weights[prefix + "post_attention_layernorm.weight"]
                                .to(torch.float32)
                                .cpu()
                                .numpy()
        )
        )

        # --- MLP ---
        tf_layer.mlp.gate_proj.kernel.assign(
            tf.transpose(tf.convert_to_tensor(weights[prefix + "mlp.gate_proj.weight"]
                                .to(torch.float32)
                                .cpu()
                                .numpy()
            ))
        )
        tf_layer.mlp.up_proj.kernel.assign(
            tf.transpose(tf.convert_to_tensor(weights[prefix + "mlp.up_proj.weight"]
                                .to(torch.float32)
                                .cpu()
                                .numpy()
        ))
        )
        tf_layer.mlp.down_proj.kernel.assign(
            tf.transpose(tf.convert_to_tensor(weights[prefix + "mlp.down_proj.weight"]
                                .to(torch.float32)
                                .cpu()
                                .numpy()
        ))
        )


def load_gemma_tf_model(tf_model):
    tokenizer , weights = load_tf_model()
    # --- Token embeddings ---
    tf_model.language_model.model.embed_tokens.embeddings.assign(
        tf.convert_to_tensor(weights["model.embed_tokens.weight"]
                             .to(torch.float32)
                             .cpu()
                             .numpy())
    )

    # --- Decoder Layers ---
    num_layers = len(tf_model.language_model.model.net.layers)
    for i in range(num_layers):
        print(f"Loading layer {i}…")
        load_layer(tf_model.language_model.model.net.layers[i], weights, i)

    # --- Final RMSNorm ---
    tf_model.language_model.model.norm.weight.assign(
        tf.convert_to_tensor(weights["model.norm.weight"]
        .to(torch.float32)
        .cpu()
        .numpy())
    )

    print("Gemma TF model fully loaded.")
    return tokenizer,tf_model

