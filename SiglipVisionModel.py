import tensorflow as tf
from SiglipVisionTransformer import  SiglipVisionTransformer


class SiglipVisionModel(tf.keras.Model):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer( config )

    def call(self, pixel_values):
        #Takes a batch of images and returns a batch of list of embeddings
        return self.vision_model( pixel_values)