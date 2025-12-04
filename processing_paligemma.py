import tensorflow as tf
import numpy as np
from PIL import Image

IMAGENET_STD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD_STD = [0.5, 0.5, 0.5]
IMAGE_TOKEN = "<image>"

def add_image_token_to_prompt(
        prefix_prompt,
        bos_token,
        image_seq_len,
        image_token
):
    return f"{ image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def rescale(
        image,
        scale
):
    rescaled_image = image * scale
    return rescaled_image.astype(np.float32)

def resize(
        image,
        size,
        resample,
        reducing_gap
):
    height,width = size
    resized_image = image.resize((width,height),resample=resample,reducing_gap=reducing_gap)
    return resized_image

def normalize(
        image,
        mean,
        std
):
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image
def process_images(
    images,
    size,
    resample,
    rescale_factor,
    image_mean,
    image_std
):
    height, width = size
    images = [resize(image=image, size=(height,width), resample=resample, reducing_gap=None) for image in images]
    images = [np.array(image).astype(np.float32) for image in images]
    images = [image * rescale_factor for image in images]
    images = [(image - image_mean) / image_std for image in images]  # normalize manually
    images = [tf.convert_to_tensor(image, dtype=tf.float32) for image in images]
    # print ("Stacked image ", tf.stack(images, axis=0).shape)
    return tf.stack(images, axis=0)

# def process_images(
#     images,
#     size,
#     resample,
#     rescale_factor,
#     image_mean,
#     image_std):
#     height,width = size[0],size[1]
#     images = [resize( image=image,size =(height,width),resample=resample,reducing_gap=None) for image in images]
#     images = [np.array(image) for image in images]
#     images = [rescale( image, scale=rescale_factor) for image in images]
#     normalization_layer = tf.keras.layers.Normalization(mean=image_mean, variance=tf.pow(image_std, 2))
#     images = [normalization_layer(image) for image in images]
#     images = [tf.transpose(image,perm=[2, 0, 1]) for image in images]
#     return images


class PaligemmaProcessor():

    def __init__(self, tokenizer, num_image_tokens, image_size):
        super(PaligemmaProcessor,self).__init__()
        self.image_seq_len = num_image_tokens
        self.image_size = image_size
        tokens_to_add = {"additional_special_tokens": [IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer
        self.image_token_string = IMAGE_TOKEN
        self.tokenizer = tokenizer

    def __call__(self, text, images, padding="max_length", truncation=True):
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts"
        pixel_values = process_images(
            images,
            size=(self.image_size,self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean=IMAGENET_STD_MEAN,
            image_std=IMAGENET_STD_STD
        )
        input_strings = [
            add_image_token_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_len,
                image_token=self.image_token_string,

            ) for prompt in text
        ]
        inputs = self.tokenizer(
            input_strings,
            return_tensors="tf",
            padding=padding,
            truncation=truncation
        )
        INCORRECT_ID = 257152
        CORRECT_ID = 256000

        inputs['input_ids'] = tf.where(
            tf.equal(inputs['input_ids'], INCORRECT_ID),
            CORRECT_ID,
            inputs['input_ids']
        )
        # Assuming pixel_values is a tensor and inputs is a dictionary of tensors
        return_data = {"pixel_values": pixel_values}
        return_data.update(inputs)
        return return_data

    # def __call__(self, text, images, padding="max_length", truncation=True):
    #     assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts"
    #
    #     pixel_values = process_images(
    #         images,
    #         size=(self.image_size, self.image_size),
    #         resample=Image.Resampling.BICUBIC,
    #         rescale_factor=1 / 255.0,
    #         image_mean=IMAGENET_STD_MEAN,
    #         image_std=IMAGENET_STD_STD
    #     )
    #
    #     text_prompt = [f"{self.tokenizer.bos_token}{text[0]}\n"]
    #
    #     text_inputs = self.tokenizer(
    #         text_prompt,
    #         return_tensors="tf",
    #         padding=False,
    #         truncation=truncation
    #     )
    #
    #     V_TOKEN_ID = 256000
    #     v_tokens = tf.fill((1, self.image_seq_len), V_TOKEN_ID)  # Shape (1, 256)
    #
    #     input_ids = tf.concat([v_tokens, text_inputs['input_ids']], axis=-1)
    #
    #     attention_mask = tf.ones_like(input_ids, dtype=tf.int32)
    #
    #     return_data = {
    #         "pixel_values": pixel_values,
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask
    #     }
    #     return return_data