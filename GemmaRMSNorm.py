import tensorflow as tf

class GemmaRMSNorm(tf.keras.Model):
    def __init__(self,
                 dim,
                 eps : 1e-6):
        super().__init__()
        self.eps = 1e-6
        self.weight = tf.Variable(tf.ones((dim,)))

    def _norm(self,x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)


    def call(self,x):
        output = self._norm(tf.cast(x,tf.float32))
        output = output * (1 + tf.cast(self.weight, tf.float32))
        return tf.cast(output, x.dtype)