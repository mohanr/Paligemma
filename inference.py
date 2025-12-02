from PIL import Image
import tensorflow as tf
import tensorflow_probability as tfp
import fire
from processing_paligemma import PaligemmaProcessor
from KVCache import  KVCache
from Utils import  load_gemma_tf_model
import KVCache
from modelling_gemma import PaliGemmaForConditionalGeneration
from PaliGemmaConfig import PaliGemmaConfig


def get_model_inputs( processor,
                      prompt,
                      image_file_path):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts,images=images)
    return  model_inputs


def sample_top_p(logits, temperature=1.0, top_p=0.9):
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
    sorted_probs = sorted_probs / tf.reduce_sum(sorted_probs, axis=-1, keepdims=True)

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

    model_inputs = get_model_inputs(processor,prompt,image_file_path)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache.KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache
    )

    kv_cache = outputs["kv_cache"]
    next_token_logits = outputs["logits"][:, -1, :]
    for _ in range(max_tokens_to_generate):

        if do_sample:
            next_token = sample_top_p(next_token_logits, temperature=temperature, top_p=top_p)
        else:
            next_token = tf.math.argmax(next_token_logits, axis=-1)


        next_token = tf.reshape(next_token, (1,1))
        assert tf.shape(next_token)[0] == 1
        assert tf.shape(next_token)[1] == 1
        generated_tokens.append(next_token)
        if next_token.numpy().item() == stop_token:
            break

        input_ids = next_token
        attention_mask = tf.concat([attention_mask, tf.ones((1,1),dtype=tf.int32)],axis=-1)
        outputs = model(
            input_ids=input_ids,
            pixel_values=None,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

    generated_tokens = tf.concat(generated_tokens, axis=-1)
    print("token_ids:", generated_tokens.numpy().tolist())
    # After generation
    generated_tokens = tf.concat(generated_tokens, axis=-1)  # shape: (1, seq_len)
    # Convert to Python list of ints
    generated_tokens_list = generated_tokens.numpy().flatten().tolist()
    # Decode properly
    decoded = processor.tokenizer.decode(generated_tokens_list, skip_special_tokens=True)
    print("Generated text:", decoded)

    decoded = processor.tokenizer.decode(generated_tokens.numpy().flatten(), skip_special_tokens=True)
    print( prompt )
    print( decoded)

def main(
        model_path : str = None,
        prompt : str = None,
        image_file_path : str = None,
        max_tokens_to_generate : int = 100,
        temparature : float = 0.8,
        top_p : float = 0.9,
        do_sample : bool = True,
        only_cpu : bool = False
):
    device = "cpu"
    print(f"Device is {device}")

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(PaliGemmaConfig())
    # Build model by calling it once
    _ = model(
        input_ids=tf.zeros((1, 10)),
        pixel_values=tf.zeros((1, 224, 224, 3)),
        attention_mask=tf.zeros((1, 1)),
        kv_cache=KVCache.KVCache(),
        training=False
    )
    # for k in model.variables:
    #     print("Model :", k.name,k.shape)

    tokenizer,model = load_gemma_tf_model(model)
    num_image_tokens = model.config.vision_config.num_image_tokens
    processor = PaligemmaProcessor(tokenizer,num_image_tokens,image_size=model.config.vision_config.image_size)

    test_inference(model,
                   processor,
                   device,
                   prompt,
                   image_file_path,
                   max_tokens_to_generate,
                   temparature,
                   top_p,
                   do_sample)

if __name__ == "__main__":
    fire.Fire(main)