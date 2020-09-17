import tensorflow as tf
from tensorflow.keras import layers


def orthogonal_regularizer(scale=0.0001):
    def regularizer(w):
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        identity = tf.eye(c)

        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        loss = tf.nn.l2_loss(reg)

        return scale * loss

    return regularizer


def orthogonal_regularizer_fully(scale=0.0001):
    def regularizer_fully(w):
        _, c = w.get_shape().as_list()

        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        loss = tf.nn.l2_loss(reg)

        return scale * loss

    return regularizer_fully


class ResBlock(tf.keras.Model):
    def __init__(self, channels, weight_init, use_bias=True):
        super(ResBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv0 = layers.Conv2D(channels, 3, 1, 'SAME', use_bias=use_bias, kernel_initializer=weight_init)
        self.conv1 = layers.Conv2D(channels, 3, 1, 'SAME', use_bias=use_bias, kernel_initializer=weight_init)

    def __call__(self, inputs, training=False, **kwargs):
        # res1
        x = self.conv0(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        # res2
        x = self.conv1(x)
        x = self.bn(x)
        return x + inputs


class ResBlockUp(tf.keras.Model):
    def __init__(self, channels, weight_init, use_bias=True):
        super(ResBlockUp, self).__init__()
        self.batch_normalization1 = layers.BatchNormalization(fused=True)
        self.batch_normalization2 = layers.BatchNormalization(fused=True)
        self.transpose = tf.keras.layers.UpSampling2D()
        self.conv1 = layers.Conv2D(channels, 3, 1, 'SAME', use_bias=use_bias, kernel_initializer=weight_init,
                                   kernel_regularizer=orthogonal_regularizer(0.0001))
        self.conv2 = layers.Conv2D(channels, 3, 1, 'SAME', use_bias=use_bias, kernel_initializer=weight_init,
                                   kernel_regularizer=orthogonal_regularizer(0.0001))
        self.skip_conv = layers.Conv2D(channels, 1, 1, 'SAME', use_bias=use_bias, kernel_initializer=weight_init,
                                       kernel_regularizer=orthogonal_regularizer(0.0001))

    def call(self, inputs, training=False, **kwargs):
        # res1
        x = self.batch_normalization1(inputs)
        x = tf.nn.relu(x)
        x = self.transpose(x)
        x = self.conv1(x)

        # res2
        x = self.batch_normalization2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # skip
        x_init = self.transpose(inputs)
        x_init = self.skip_conv(x_init)

        return x + x_init


class ResBlockDown(tf.keras.Model):
    def __init__(self, channels, weight_init, use_bias=True):
        super(ResBlockDown, self).__init__()
        self.conv0 = layers.Conv2D(channels, 3, 1, 'SAME', use_bias=use_bias, kernel_initializer=weight_init)
        self.conv1 = layers.Conv2D(channels, 3, 1, 'SAME', use_bias=use_bias, kernel_initializer=weight_init)
        self.skip_conv = layers.Conv2D(channels, 1, 1, 'SAME', use_bias=use_bias, kernel_initializer=weight_init)
        self.avg_pooling = tf.keras.layers.AveragePooling2D(padding='SAME')

    def call(self, inputs, training=False, **kwargs):
        # res1
        x = tf.nn.relu(inputs)
        x = self.conv0(x)
        # res2
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.avg_pooling(x)
        # skip
        x_init = self.skip_conv(inputs)
        x_init = self.avg_pooling(x_init)

        return x + x_init


class Attention(tf.keras.Model):
    def __init__(self, ch):
        super(Attention, self).__init__()
        self.filters_f_g_h = ch // 8
        self.filters_v = ch
        self.f_l = tf.keras.layers.Conv2D(self.filters_f_g_h, 1, 1, use_bias=True)
        self.g_l = tf.keras.layers.Conv2D(self.filters_f_g_h, 1, 1, use_bias=True)
        self.h_l = tf.keras.layers.Conv2D(self.filters_f_g_h, 1, 1, use_bias=True)
        self.v_l = tf.keras.layers.Conv2D(self.filters_v, 1, 1, use_bias=True)
        self.gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)

    def call(self, inputs, training=False, **kwargs):
        def hw_flatten(x):
            return tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[-1]])

        f = self.f_l(inputs)

        g = self.g_l(inputs)

        h = self.h_l(inputs)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        v = tf.matmul(beta, hw_flatten(h))
        v = tf.reshape(v, shape=[inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[1],
                                 inputs.get_shape().as_list()[2], -1])  # [bs, h, w, C]

        o = self.v_l(v)

        output = self.gamma * o + inputs

        return output
