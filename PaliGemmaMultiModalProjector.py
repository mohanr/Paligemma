import tensorflow as tf


class PaliGemmaMultiModalProjector(tf.keras.layers.Layer):
    def __init__(self, config):
        super(PaliGemmaMultiModalProjector, self).__init__()
        self.config = config

        self.linear = tf.keras.layers.Dense(
            2048,
            activation=None,
            use_bias=True
        )

    def call(self, image_features):
        print(f"\n=== PROJECTOR DEBUG ===")
        print(f"Input std: {tf.math.reduce_std(image_features).numpy():.6f}")
        print(f"Input [0,0,:5]: {image_features[0, 0, :5].numpy()}")

        # Check if layer is built and has weights
        print(f"Layer built: {self.linear.built}")
        if hasattr(self.linear, 'kernel'):
            print(f"Kernel exists: True")
            print(f"Kernel shape: {self.linear.kernel.shape}")
            print(f"Kernel std: {tf.math.reduce_std(self.linear.kernel).numpy():.6f}")
            print(f"Bias std: {tf.math.reduce_std(self.linear.bias).numpy():.6f}")
        else:
            print(f"Kernel exists: False - LAYER NOT BUILT!")

        # Dense layer
        hidden_states = self.linear(image_features)
        print(f"Dense output std: {tf.math.reduce_std(hidden_states).numpy():.6f}")
        print(f"Dense output [0,0,:5]: {hidden_states[0, 0, :5].numpy()}")

        return hidden_states