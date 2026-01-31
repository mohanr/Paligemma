import tensorflow as tf
# https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/pali_gemma/pali_gemma_causal_lm_preprocessor.py

class PaliGemmaMultiModalProjector(tf.keras.layers.Layer):
    def __init__(self, config):
        super(PaliGemmaMultiModalProjector, self).__init__()
        self.config = config

        # Use zeros initializer to ensure no random initialization interference
        # Explicitly disable constraints and regularizers
        self.linear = tf.keras.layers.Dense(
            2048,
            activation=None,
            use_bias=True,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        )

    def call(self, image_features):
        print(f"Projector input dtype: {image_features.dtype}")

        hidden_states = self.linear(image_features)

        print(f"Projector output dtype: {hidden_states.dtype}")
        return hidden_states