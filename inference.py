from PIL import Image
import tensorflow as tf
import tensorflow_probability as tfp
import fire
from processing_paligemma import PaligemmaProcessor
from KVCache import KVCache
from Utils import load_gemma_tf_model
import KVCache
from modelling_gemma import PaliGemmaForConditionalGeneration
from PaliGemmaConfig import PaliGemmaConfig
import numpy as np


def get_model_inputs(processor,
                     prompt,
                     image_file_path):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    print("\n=== IMAGE PREPROCESSING DEBUG ===")
    print(f"Pixel values shape: {model_inputs['pixel_values'].shape}")
    print(f"Pixel values mean: {tf.reduce_mean(model_inputs['pixel_values'])}")
    print(f"Pixel values std: {tf.math.reduce_std(model_inputs['pixel_values'])}")
    print(
        f"Pixel values min/max: {tf.reduce_min(model_inputs['pixel_values'])}/{tf.reduce_max(model_inputs['pixel_values'])}")

    return model_inputs


def sample_top_p(logits, temperature=1.0, top_p=0.99):
    # >>>>>>>>>>>>> CRITICAL LOGIT STABILIZATION FIX <<<<<<<<<<<<<<<<
    # Shift logits to prevent overflow in exp and improve stability.
    # Subtracting the max logit makes the largest logit 0.0.
    max_logit = tf.reduce_max(logits, axis=-1, keepdims=True)
    logits = logits - max_logit
    # >>>>>>>>>>>>> END CRITICAL FIX <<<<<<<<<<<<<<<<

    # Apply temperature
    logits = logits / temperature
    probs = tf.nn.softmax(logits, axis=-1)

    # Sort probs descending
    sorted_probs, sorted_indices = tf.math.top_k(probs, k=tf.shape(probs)[-1], sorted=True)
    cumsum_probs = tf.cumsum(sorted_probs, axis=-1, exclusive=False)

    # Mask tokens outside top-p
    mask = cumsum_probs > top_p
    mask = tf.concat([tf.zeros_like(mask[:, :1], dtype=tf.bool), mask[:, 1:]], axis=-1)
    sorted_probs = tf.where(mask, tf.zeros_like(sorted_probs), sorted_probs)

    # Renormalize remaining probabilities
    sum_probs = tf.reduce_sum(sorted_probs, axis=-1, keepdims=True)
    # Avoid division by zero in case all probs were masked (rare, but possible)
    safe_sum_probs = tf.where(tf.equal(sum_probs, 0.0), tf.ones_like(sum_probs), sum_probs)
    sorted_probs = sorted_probs / safe_sum_probs

    # Sample one token per batch
    next_token_idx = tfp.distributions.Categorical(probs=sorted_probs).sample()
    next_token = tf.gather(sorted_indices, next_token_idx, batch_dims=1)
    return next_token


def test_inference(model,
                   processor,
                   device,
                   prompt,
                   image_file_path,
                   max_tokens_to_generate,
                   temperature,
                   top_p,
                   do_sample):
    model_inputs = get_model_inputs(processor, prompt, image_file_path)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    # DEBUG: Print input_ids to compare with PyTorch
    print("\n=== INPUT IDS DEBUG ===")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs (first 30): {input_ids.numpy()[0, :30]}")
    print(f"Input IDs (last 10): {input_ids.numpy()[0, -10:]}")
    print(
        f"Number of token 257152 (image token): {tf.reduce_sum(tf.cast(tf.equal(input_ids, 257152), tf.int32)).numpy()}")
    print(f"Unique token IDs: {tf.unique(tf.reshape(input_ids, [-1]))[0][:20].numpy()}")

    kv_cache = KVCache.KVCache()
    kv_cache.reset()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache
    )

    kv_cache = outputs["kv_cache"]
    # Update kv_cache sequence length after the pre-fill step
    kv_cache.sequence_len = tf.shape(model_inputs["input_ids"])[1]
    next_token_logits = outputs["logits"][:, -1, :]

    # <<<<<<<<<<<<<<<< LOGIT DIAGNOSTIC (Still showing RAW unstable logits) >>>>>>>>>>>>>>>>>
    logits_tensor = next_token_logits[0]
    logits_min = tf.reduce_min(logits_tensor)
    logits_max = tf.reduce_max(logits_tensor)
    tf.print(f"STEP 1 LOGIT DIAGNOSTIC (Raw Logits):")
    tf.print(f"  Min Logit: {logits_min:.4f}, Max Logit: {logits_max:.4f}")
    top_logits, top_indices = tf.math.top_k(logits_tensor, k=5)
    tf.print(f"  Top 5 Raw Logits: {top_logits.numpy()}")
    # <<<<<<<<<<<<<<<< END DIAGNOSTIC >>>>>>>>>>>>>>>>>

    # --- STEP 2: Generate the First Token (Forces STABILIZATION via sample_top_p) ---
    if do_sample:
        # Calls sample_top_p which contains the stabilization fix
        first_token = sample_top_p(next_token_logits, temperature=temperature, top_p=top_p)
    else:
        # If no sampling, manually apply logit stabilization for argmax
        max_logit = tf.reduce_max(next_token_logits, axis=-1, keepdims=True)
        stable_logits = next_token_logits - max_logit
        first_token = tf.math.argmax(stable_logits, axis=-1)

    # Prepare and store the first token
    next_token = tf.reshape(first_token, (1, 1))
    print(f"First token ID: {next_token.numpy().item()}")
    print(f"First token decoded: '{processor.tokenizer.decode([next_token.numpy().item()])}'")
    print(f"Token 34371 decoded (PyTorch prediction): '{processor.tokenizer.decode([34371])}'")
    print(f"Token 34371 logit in TF: {next_token_logits[0, 34371].numpy():.4f}")
    print(
        f"Token {next_token.numpy().item()} logit in TF: {next_token_logits[0, next_token.numpy().item()].numpy():.4f}")

    generated_tokens.append(next_token)
    print(f"First token ID: {next_token.numpy().item()}")
    print(f"Stop token ID: {stop_token}")
    print(f"Is stop token? {next_token.numpy().item() == stop_token}")
    if next_token.numpy().item() == stop_token:
        print("STOPPING: First token is EOS!")
        pass
    else:
        for i in range(max_tokens_to_generate - 1):
            print(f"\n=== Generation step {i + 1} ===")

            input_ids = next_token
            attention_mask = tf.concat([attention_mask, tf.ones((1, 1), dtype=tf.int32)], axis=-1)

            outputs = model(
                input_ids=input_ids,
                pixel_values=None,
                attention_mask=attention_mask,
                kv_cache=kv_cache
            )

            kv_cache = outputs["kv_cache"]
            print(f"  KV cache items AFTER call: {kv_cache.num_items()}")

            next_token_logits = outputs["logits"][:, -1, :]
            print(f"  Logits shape: {tf.shape(next_token_logits)}")
            print(f"  Logits min/max: {tf.reduce_min(next_token_logits)}/{tf.reduce_max(next_token_logits)}")

            # Generate next token
            if do_sample:
                next_token = sample_top_p(next_token_logits, temperature=temperature, top_p=top_p)
            else:
                max_logit = tf.reduce_max(next_token_logits, axis=-1, keepdims=True)
                stable_logits = next_token_logits - max_logit
                next_token = tf.math.argmax(stable_logits, axis=-1)

            next_token = tf.reshape(next_token, (1, 1))
            print(f"  Generated token ID: {next_token.numpy().item()}")

            generated_tokens.append(next_token)

            if next_token.numpy().item() == stop_token:
                print(f"  BREAKING: Generated EOS token")
                break

            print(f"  Total tokens so far: {len(generated_tokens)}")
    # --- FINAL DECODING ---
    generated_tokens = tf.concat(generated_tokens, axis=-1)
    generated_tokens_list = generated_tokens.numpy().flatten().tolist()

    # Decode properly
    decoded = processor.tokenizer.decode(generated_tokens_list, skip_special_tokens=True)
    print("Generated text:", decoded)

    print(prompt)
    print(decoded)


def main(
        model_path: str = None,
        prompt: str = None,
        image_file_path: str = None,
        max_tokens_to_generate: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        only_cpu: bool = False
):
    device = "cpu"
    print(f"Device is {device}")

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(PaliGemmaConfig())
    # Build model by calling it once
    vocab_size = model.config.vocab_size
    image_size = model.config.vision_config.image_size
    # Simulate a real image: random values in [0, 255]
    dummy_image = tf.random.uniform(
        (1, image_size, image_size, 3),
        minval=0.0,
        maxval=255.0,
        dtype=tf.float32
    )
    # Apply the SAME normalization as process_images
    pixel_values = (dummy_image / 255.0 - 0.5) / 0.5
    _ = model(
        input_ids=tf.random.uniform(
            (1, 10),
            minval=1,  # Avoid padding token 0
            maxval=min(1000, vocab_size),
            dtype=tf.int32
        ),

        pixel_values=pixel_values,
        attention_mask=tf.ones((1, 10), dtype=tf.int32),  # All tokens valid
        kv_cache=KVCache.KVCache(),
        training=False)
    tokenizer, model = load_gemma_tf_model(model)
    # After: tokenizer, model = load_gemma_tf_model(model)
    print("\n=== TENSORFLOW EMBEDDINGS ===")
    tf_embed_cats = model.language_model.model.embed_tokens.embeddings.numpy()[34371]
    tf_embed_comma = model.language_model.model.embed_tokens.embeddings.numpy()[235269]

    print(f"TF embedding for token 34371 ('cats')[:10]: {tf_embed_cats[:10]}")
    print(f"TF embedding for token 235269 (',')[:10]: {tf_embed_comma[:10]}")
    print(f"TF embedding norm for 'cats': {np.linalg.norm(tf_embed_cats):.6f}")
    print(f"TF embedding norm for ',': {np.linalg.norm(tf_embed_comma):.6f}")

    print("\n=== COMPARISON ===")
    print(
        f"PyTorch 'cats' [:10]: [ 0.33346865  0.00137039 -0.05751126  0.12262455 -0.02193636  0.11118821  0.09834561  0.10248051 -0.07931319 -0.16288337]")
    print(f"TF 'cats' [:10]:      {tf_embed_cats[:10]}")
    print(
        f"Match? {np.allclose(tf_embed_cats[:10], [0.33346865, 0.00137039, -0.05751126, 0.12262455, -0.02193636, 0.11118821, 0.09834561, 0.10248051, -0.07931319, -0.16288337], atol=1e-5)}")
    # img_input = tf.zeros((1, 448,448, 3),
    #                      dtype=tf.float32)
    #
    # try:
    #     vit_output = model.vision_tower(img_input)
    #
    #     visual_tokens = model.multi_modal_projector(vit_output)
    #
    # except Exception as e:
    #     print(f"DIAGNOSIS: Forward pass failed in vision path: {e}")
    num_image_tokens = model.config.vision_config.num_image_tokens
    processor = PaligemmaProcessor(tokenizer, num_image_tokens, image_size=model.config.vision_config.image_size)
    print("\n=== PRE-INFERENCE POST_NORM CHECK ===")
    print(f"Post norm gamma[:5]: {model.vision_tower.vision_model.post_layernorm.gamma.numpy()[:5]}")
    print(f"Post norm beta[:5]: {model.vision_tower.vision_model.post_layernorm.beta.numpy()[:5]}")

    test_inference(model,
                   processor,
                   device,
                   prompt,
                   image_file_path,
                   max_tokens_to_generate,
                   temperature,
                   top_p,
                   do_sample)


if __name__ == "__main__":
    fire.Fire(main)
